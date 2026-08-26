"""
gefl_f/engine_f.py

GeFL-F training loop — Algorithm 3 from Kang et al. 2025.

Three-stage federated training with a common feature extractor:

Stage (i)  — Warm up common FE (T_FE rounds):
    Each client trains {θ_f, θ_h_m} JOINTLY on real (x,y), T_w local epochs.
    Server: θ_f ← FlatAvg(all clients)       # every client contributes
    Server: θ_h_m ← FlatAvg(clients with header m)  # grouped, as usual

Stage (ii) — Generative knowledge aggregation (T_KA rounds):
    FE is FROZEN from stage (i). Each client:
      1. Extracts features = FE(x) for real local (x,y)
      2. Trains feature-generator G_F on (features, y) pairs, T_g local epochs
    Server: w_g ← FlatAvg(all clients)

Stage (iii) — Target header training (T_TN rounds):
    FE stays FROZEN. Each client, each round:
      1. T_s epochs on SYNTHETIC features ~ G_F(z|y, w_g), header-only gradient
      2. T_r epochs on REAL data forwarded through frozen FE, header-only gradient
    Server: θ_h_g,m ← GroupedFlatAvg (per header architecture)

Key: θ_f is trained ONLY in stage (i) then frozen. Stages (ii)+(iii)
use it in eval/no-grad mode.
"""
import copy
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from gefl_f.feature_extractor import CommonFeatureExtractor
from gefl_f.headers import HEADER_REGISTRY
from utils.avg import FedAvg, model_wise_FedAvg, aggregate_generator
from utils.checkpoint import ckpt_path, save_checkpoint, save_results
from utils.evaluate import (average_client_bucketed_accuracy, bucketed_accuracy, gap_report,
                            train_centralized_upper_bound, mnd_ratio, make_held_out_val_split,
                            ConvergenceTracker, classification_report_from_cm,
                            compute_confusion_matrix, per_client_accuracy)
from utils.label_sampler import build_label_sampler
from utils.localUpdateGen import get_local_gen_update
from utils.logger import ExperimentLogger
from utils.seed import set_seed
from utils.user_sampling import ClientSampler



def _client_loader(exp, client_id: int, args) -> DataLoader:
    idxs = exp.dict_users[client_id]
    return DataLoader(Subset(exp.dataset_train, idxs), batch_size=args.local_bs, shuffle=True)


def _build_fe(args) -> CommonFeatureExtractor:
    """Build the common feature extractor."""
    fe_channels = getattr(args, 'fe_channels', 32)
    # in_channels is determined by dataset — set in run_gefl_f
    return CommonFeatureExtractor(
        in_channels=args._fe_in_channels,
        fe_channels=fe_channels,
    ).to(args.device)


def _build_header(header_name: str, fe_channels: int, num_classes: int, args):
    """Build one header network."""
    cls = HEADER_REGISTRY.get(header_name)
    return cls(fe_channels=fe_channels, num_classes=num_classes).to(args.device)


def _build_feature_generator(num_classes, fe_channels, fe_spatial, args):
    """Build the feature-space generator.

    Output activation: ReLU for VAE/GAN because the frozen FE ends in
    ReLU+MaxPool so real features are non-negative — matching that range
    keeps synthetic and real features on the same manifold. DDPM predicts
    the noise epsilon (unbounded); wrapping its output in ReLU is a
    semantic error, so the DDPM path leaves it unbounded.
    """
    from generators.base import GEN_REGISTRY
    # Import specific generators so they register
    import generators.ccvae
    import generators.ccgan
    import generators.cddpm
    gen_name = args.gen_model
    gen_cls = GEN_REGISTRY.get(gen_name)
    out_act = "relu" if gen_name in ("vae", "gan") else "none"
    return gen_cls(
        num_classes=num_classes,
        in_channels=fe_channels,
        img_size=fe_spatial,
        args=args,
        output_activation=out_act,
    ).to(args.device)


# ============================================================
#  Stage (i): FE Warm-up — joint FE + header training
# ============================================================

def _stage_i_round(exp, args, fe_state, header_states, client_ids):
    """One round of stage (i): each client trains FE+header jointly on real data."""
    fe_updates = []
    per_group_header_states = [[] for _ in range(args.num_models)]
    losses = []

    fe_channels = getattr(args, 'fe_channels', 32)
    t_w = getattr(args, 'gen_local_ep', 5)  # paper: T_w local epochs

    for cid in client_ids:
        loader = _client_loader(exp, cid, args)
        if len(loader.dataset) == 0:
            continue

        # Build FE + header for this client
        fe = _build_fe(args)
        fe.load_state_dict(fe_state)
        fe.train()

        group = exp.dev_spec_idx[cid]
        header = _build_header(
            args.header_models_list[group], fe_channels,
            exp.meta.num_classes, args
        )
        header.load_state_dict(header_states[group])
        header.train()

        # Joint optimizer over both FE and header
        params = list(fe.parameters()) + list(header.parameters())
        if args.optimizer == "adam":
            opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        else:
            opt = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.weight_decay)

        total_loss, n_batches = 0.0, 0
        for _ in range(t_w):
            for x, y in loader:
                x, y = x.to(args.device), y.to(args.device)
                features = fe(x)
                logits = header(features)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1

        fe_updates.append(fe.state_dict())
        per_group_header_states[group].append(header.state_dict())
        losses.append(total_loss / max(n_batches, 1))

    # Aggregate FE across ALL clients (flat FedAvg)
    if fe_updates:
        new_fe_state = FedAvg(fe_updates)
    else:
        new_fe_state = fe_state

    # Aggregate headers per group
    new_header_states = model_wise_FedAvg(header_states, per_group_header_states)

    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    return new_fe_state, new_header_states, avg_loss


# ============================================================
#  Stage (ii): Feature-generator training (FE frozen)
# ============================================================

def _stage_ii_round(exp, args, fe_state, gen_state, gen_opt_states, client_ids, fe_channels, fe_spatial):
    """One round of stage (ii): train feature generator on FE(x) features."""
    client_states, client_counts, client_class_counts_list = [], [], []
    gen_losses = []

    local_gen_update_fn = get_local_gen_update(args.gen_model + "_f")

    for cid in client_ids:
        loader = _client_loader(exp, cid, args)
        if len(loader.dataset) == 0:
            continue

        # Frozen FE to extract features
        fe = _build_fe(args)
        fe.load_state_dict(fe_state)
        fe.eval()

        # Create feature-label dataset from this client's data
        features_list, labels_list = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(args.device)
                feat = fe(x)
                features_list.append(feat.cpu())
                labels_list.append(y)

        feat_dataset = torch.utils.data.TensorDataset(
            torch.cat(features_list), torch.cat(labels_list)
        )
        feat_loader = DataLoader(feat_dataset, batch_size=args.local_bs, shuffle=True)

        # Train feature generator
        gen_net = _build_feature_generator(exp.meta.num_classes, fe_channels, fe_spatial, args)
        if gen_state is not None:
            gen_net.load_state_dict(gen_state)

        new_state, loss, new_opt_state = local_gen_update_fn(gen_net, feat_loader, args, gen_opt_states.get(cid))
        gen_opt_states[cid] = new_opt_state
        client_states.append(new_state)
        client_counts.append(len(loader.dataset))
        client_class_counts_list.append(exp.client_class_counts[cid])
        gen_losses.append(loss)

    if not client_states:
        return gen_state, float("nan")

    # Aggregate feature generator (flat FedAvg for baseline, Mechanism A if enabled)
    conditioning_keys = _build_feature_generator(
        exp.meta.num_classes, fe_channels, fe_spatial, args
    ).conditioning_parameter_names()

    new_state = aggregate_generator(
        client_states, client_counts, client_class_counts_list, exp.meta.num_classes,
        conditioning_keys, mechanism_a=bool(args.mechanism_a), beta=args.mech_a_beta,
        support_floor=getattr(args, "mech_a_support_floor", 0),
    )
    mean_gen_loss = sum(gen_losses) / len(gen_losses) if gen_losses else float("nan")
    return new_state, mean_gen_loss


# ============================================================
#  Stage (iii): Header training (FE frozen, sequential Ts/Tr)
# ============================================================

def _stage_iii_round(exp, args, fe_state, gen_state, header_states, client_ids,
                     fe_channels, fe_spatial, label_samplers):
    """One round of stage (iii): sequential synthetic/real training, header-only gradients."""
    per_group_states = [[] for _ in range(args.num_models)]
    losses = []

    # Build frozen FE and feature generator
    fe = _build_fe(args)
    fe.load_state_dict(fe_state)
    fe.eval()

    gen_net = None
    if gen_state is not None:
        gen_net = _build_feature_generator(exp.meta.num_classes, fe_channels, fe_spatial, args)
        gen_net.load_state_dict(gen_state)
        gen_net.eval()

    for cid in client_ids:
        loader = _client_loader(exp, cid, args)
        if len(loader.dataset) == 0:
            continue

        group = exp.dev_spec_idx[cid]
        header = _build_header(
            args.header_models_list[group], fe_channels,
            exp.meta.num_classes, args
        )
        header.load_state_dict(header_states[group])
        header.train()

        def _make_header_opt():
            if args.optimizer == "adam":
                return torch.optim.Adam(header.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            return torch.optim.SGD(header.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        total_loss, n_batches = 0.0, 0
        # Accumulate fidelity across the whole synthetic phase, apply once at
        # the end (matches GEFL's utils/localUpdateTarget.py behaviour so both
        # variants respond to Mechanism B identically).
        fid_sum = defaultdict(float)
        fid_n = defaultdict(int)

        # Phase 1: Synthetic features (T_s epochs) — fresh optimizer per phase.
        if gen_net is not None:
            opt = _make_header_opt()
            for _ in range(args.target_ts):
                for x_real, _ in loader:
                    batch_size = x_real.size(0)
                    syn_y = label_samplers[cid].sample(batch_size).to(args.device) \
                        if cid in label_samplers else torch.randint(0, exp.meta.num_classes, (batch_size,), device=args.device)
                    with torch.no_grad():
                        syn_feat = gen_net.sample(syn_y)

                    opt.zero_grad()
                    logits = header(syn_feat)
                    loss = F.cross_entropy(logits, syn_y)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                    n_batches += 1

                    # Collect fidelity feedback for Mechanism B (single update at end).
                    if args.mechanism_b and cid in label_samplers:
                        with torch.no_grad():
                            probs = F.softmax(logits.detach(), dim=1)
                            conf = probs.gather(1, syn_y.unsqueeze(1)).squeeze(1)
                        for c in syn_y.unique().tolist():
                            mask = syn_y == c
                            fid_sum[c] += conf[mask].sum().item()
                            fid_n[c] += int(mask.sum())

            if args.mechanism_b and cid in label_samplers and fid_sum:
                feedback = {c: fid_sum[c] / fid_n[c] for c in fid_sum}
                label_samplers[cid].update_fidelity(feedback)

        # Phase 2: Real features through frozen FE (T_r epochs) — fresh optimizer.
        opt = _make_header_opt()
        for _ in range(args.target_tr):
            for x, y in loader:
                x, y = x.to(args.device), y.to(args.device)
                with torch.no_grad():
                    real_feat = fe(x)

                opt.zero_grad()
                logits = header(real_feat)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1

        per_group_states[group].append(header.state_dict())
        losses.append(total_loss / max(n_batches, 1))

    new_header_states = model_wise_FedAvg(header_states, per_group_states)
    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    return new_header_states, avg_loss


# ============================================================
#  Combined model for evaluation (FE + header)
# ============================================================

class _FEHeaderModel(torch.nn.Module):
    """Wraps frozen FE + header into a single module for evaluation."""

    def __init__(self, fe, header):
        super().__init__()
        self.fe = fe
        self.header = header

    def forward(self, x):
        with torch.no_grad():
            feat = self.fe(x)
        return self.header(feat)


def _build_eval_model(fe_state, header_state, header_name, fe_channels, num_classes, args):
    """Build a combined FE+header model for evaluation."""
    fe = _build_fe(args)
    fe.load_state_dict(fe_state)
    fe.eval()
    header = _build_header(header_name, fe_channels, num_classes, args)
    header.load_state_dict(header_state)
    header.eval()
    return _FEHeaderModel(fe, header)


def _build_all_eval_models(exp, args, fe_state, header_states):
    """Build evaluation models for all clients."""
    fe_channels = getattr(args, 'fe_channels', 32)
    models = {}
    for cid in range(args.num_users):
        group = exp.dev_spec_idx[cid]
        model = _build_eval_model(
            fe_state, header_states[group],
            args.header_models_list[group], fe_channels,
            exp.meta.num_classes, args
        )
        models[cid] = model
    return models


# ============================================================
#  Main entry point
# ============================================================

def run_gefl_f(args) -> dict:
    """Run the full GeFL-F three-stage training loop."""
    from datasets.get_dataset import get_dataset
    from sampling.partition import (class_counts_long_tailed, client_class_counts,
                                     frequency_bucket_labels, partition_dataset)

    set_seed(args.seed)
    logger = ExperimentLogger(args)
    plots_dir = os.path.join(args.out_dir, "plots", args.name)
    os.makedirs(plots_dir, exist_ok=True)

    # ---- Setup ----
    dataset_train, dataset_test, meta, train_labels = get_dataset(args)
    num_classes = meta.num_classes
    in_channels = meta.in_channels
    img_size = meta.native_img_size

    # Stash in args for _build_fe
    args._fe_in_channels = in_channels

    dict_users = partition_dataset(train_labels, num_classes, args)
    class_counts = class_counts_long_tailed(train_labels, num_classes, args.imbalance_factor, args.max_per_class)
    buckets = frequency_bucket_labels(class_counts)
    per_client_counts = {cid: client_class_counts(train_labels, idxs, num_classes)
                          for cid, idxs in dict_users.items()}

    # Store experiment state in a simple namespace
    class Exp:
        pass
    exp = Exp()
    exp.dataset_train = dataset_train
    exp.dataset_test = dataset_test
    exp.meta = meta
    exp.train_labels = train_labels
    exp.dict_users = dict_users
    exp.class_counts = class_counts
    exp.buckets = buckets
    exp.client_class_counts = per_client_counts

    # Header setup
    header_list = [h.strip() for h in args.header_models.split(',') if h.strip()]
    if len(header_list) == 1:
        header_list = header_list * args.num_models
    args.header_models_list = header_list
    exp.dev_spec_idx = [i % args.num_models for i in range(args.num_users)]

    fe_channels = getattr(args, 'fe_channels', 32)

    # Initialize FE
    fe_init = _build_fe(args)
    fe_state = fe_init.state_dict()

    # Compute FE spatial output size
    with torch.no_grad():
        dummy = torch.randn(1, in_channels, img_size, img_size, device=args.device)
        fe_out = fe_init(dummy)
        fe_spatial = fe_out.shape[-1]

    # Initialize headers
    header_states = []
    for name in header_list:
        h = _build_header(name, fe_channels, num_classes, args)
        header_states.append(h.state_dict())

    client_sampler = ClientSampler(args.num_users, args.frac, args.seed)
    global_natural_counts = np.array([exp.class_counts[c] for c in range(num_classes)])
    label_samplers = {cid: build_label_sampler(args, num_classes, global_natural_counts)
                       for cid in range(args.num_users)}

    logger.console.info(
        "GeFL-F: dataset=%s classes=%d clients=%d fe_channels=%d fe_spatial=%d "
        "gen_model=%s_f T_FE=%d T_KA=%d T_TN=%d",
        meta.name, num_classes, args.num_users, fe_channels, fe_spatial,
        args.gen_model, args.fe_rounds, args.gen_wu_epochs, args.epochs
    )

    t_start = time.time()
    history = []
    convergence_tracker = ConvergenceTracker(patience=5)

    # ============================================================
    #  STAGE (i): Feature extractor warm-up
    # ============================================================
    logger.console.info("=== Stage (i): FE warm-up for %d rounds ===", args.fe_rounds)
    for rnd in range(args.fe_rounds):
        client_ids = client_sampler.select()
        fe_state, header_states, wu_loss = _stage_i_round(exp, args, fe_state, header_states, client_ids)
        if (rnd + 1) % max(1, args.fe_rounds // 10) == 0:
            logger.console.debug("FE warm-up round %d/%d loss=%.4f", rnd + 1, args.fe_rounds, wu_loss)

    # ============================================================
    #  STAGE (ii): Feature-generator training (FE frozen)
    # ============================================================
    logger.console.info("=== Stage (ii): Feature-generator training for %d rounds ===", args.gen_wu_epochs)
    gen_state = None
    gen_opt_states = {}

    # Initialize generator
    gen_init = _build_feature_generator(num_classes, fe_channels, fe_spatial, args)
    gen_state = gen_init.state_dict()

    # Register -F generator update functions
    from utils.localUpdateGen import LOCAL_GEN_UPDATE_REGISTRY

    # -F variants use the same training logic as their full counterparts
    gen_model_f = args.gen_model + "_f"
    if gen_model_f not in LOCAL_GEN_UPDATE_REGISTRY:
        base_fn = get_local_gen_update(args.gen_model)
        LOCAL_GEN_UPDATE_REGISTRY.register(gen_model_f)(base_fn)

    local_gen_update_fn = get_local_gen_update(gen_model_f)

    for rnd in range(args.gen_wu_epochs):
        client_ids = client_sampler.select()
        gen_state, gen_loss = _stage_ii_round(
            exp, args, fe_state, gen_state, gen_opt_states, client_ids, fe_channels, fe_spatial
        )
        if (rnd + 1) % max(1, args.gen_wu_epochs // 10) == 0:
            logger.console.debug("Gen training round %d/%d gen_loss=%.4f", rnd + 1, args.gen_wu_epochs, gen_loss)

    # ============================================================
    #  STAGE (iii): Header training (FE frozen, sequential Ts/Tr)
    # ============================================================
    logger.console.info("=== Stage (iii): Header training for %d rounds ===", args.epochs)
    for rnd in range(args.epochs):
        client_ids = client_sampler.select()

        header_states, avg_loss = _stage_iii_round(
            exp, args, fe_state, gen_state, header_states, client_ids,
            fe_channels, fe_spatial, label_samplers
        )

        # Evaluation
        if (rnd + 1) % args.sample_test == 0 or rnd == args.epochs - 1:
            client_models = _build_all_eval_models(exp, args, fe_state, header_states)
            from utils.evaluate import average_client_metrics
            scores = average_client_metrics(client_models, dataset_test, num_classes, buckets, args.device)

            row = {
                "round": rnd + 1,
                "train_loss": avg_loss,
                **{f"acc_{k}": v for k, v in scores.items() if k in ["overall", "head", "medium", "tail"]},
                "elapsed_s": round(time.time() - t_start, 1),
            }

            conv_info = convergence_tracker.update(scores.get("overall", 0))
            row["best_acc"] = conv_info["best_acc"]
            
            # Additional metrics
            row["macro_f1"] = scores.get("macro_f1", 0)
            row["weighted_f1"] = scores.get("weighted_f1", 0)
            row["class_balanced_accuracy"] = scores.get("class_balanced_accuracy", 0)
            row["macro_precision"] = scores.get("macro_precision", 0)
            row["macro_recall"] = scores.get("macro_recall", 0)

            logger.log(row, step=rnd + 1)
            history.append(row)

    # ---- Final results ----
    results = {"history": history, "final_scores": history[-1] if history else {}}
    results["total_time_s"] = round(time.time() - t_start, 1)

    logger.console.info("GeFL-F training complete. Total time: %.1fs", results["total_time_s"])
    if history:
        logger.console.info("Final scores: %s",
                             {k: v for k, v in history[-1].items() if k.startswith("acc_")})

    save_results(os.path.join(args.out_dir, f"{args.name}_results.pkl"), results)
    logger.close()
    return results
