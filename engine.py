"""
engine.py

The federated training loop itself. Every GeFL_*.py top-level script
(CVAE / GAN / DDPM) calls `run_gefl(args)` after setting `args.gen_model`
-- the loop below never branches on which generator type is active; that
is entirely delegated to the registries in generators/ and
utils/localUpdateGen.py. This is what keeps GeFL_CVAE.py, GeFL_GAN.py,
and GeFL_DDPM.py thin, near-duplicate-free wrappers instead of three
copies of the same 200-line loop (a deliberate improvement over having
three independently-maintained scripts).

One communication round:
  1. (if args.aid_by_gen and not args.freeze_gen) selected clients each
     run a local generator update; the server aggregates with either
     GeFL's flat FedAvg or Mechanism A's frequency-weighted rule.
  2. selected clients each train their own (architecturally heterogeneous)
     target network on real local data, optionally augmented with
     generator-sampled synthetic data drawn from their label sampler
     (uniform under the baseline, Mechanism B's fidelity-gated
     distribution otherwise); the server aggregates each architecture
     group separately with model_wise_FedAvg.
  3. if Mechanism B is active, each client's fidelity state is updated
     from this round's synthetic-sample confidence feedback.
  4. periodic evaluation + logging + checkpointing.
"""
import copy
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from targetNetModels.nets import NET_REGISTRY
from utils.avg import aggregate_generator, model_wise_FedAvg
from utils.checkpoint import ckpt_path, save_checkpoint, save_results
from utils.evaluate import (average_client_bucketed_accuracy, bucketed_accuracy, gap_report,
                             make_held_out_val_split, mnd_ratio, train_centralized_upper_bound)
from utils.label_sampler import build_label_sampler
from utils.localUpdateGen import get_local_gen_update
from utils.localUpdateTarget import LocalUpdate
from utils.logger import ExperimentLogger
from utils.seed import set_seed
from utils.setup import build_generator, build_target_net, setup_experiment
from utils.user_sampling import ClientSampler


def _client_loader(exp, client_id: int, args) -> DataLoader:
    idxs = exp.dict_users[client_id]
    return DataLoader(Subset(exp.dataset_train, idxs), batch_size=args.local_bs, shuffle=True)


def _generator_round(exp, args, gen_global_state, gen_opt_states, client_ids, local_gen_update_fn):
    client_states, client_counts, client_class_counts_list = [], [], []
    for cid in client_ids:
        net = build_generator(exp, args)
        if gen_global_state is not None:
            net.load_state_dict(gen_global_state)
        loader = _client_loader(exp, cid, args)
        if len(loader.dataset) == 0:
            continue
        new_state, _loss, new_opt_state = local_gen_update_fn(net, loader, args, gen_opt_states.get(cid))
        gen_opt_states[cid] = new_opt_state
        client_states.append(new_state)
        client_counts.append(len(loader.dataset))
        client_class_counts_list.append(exp.client_class_counts[cid])

    if not client_states:
        return gen_global_state

    conditioning_keys = build_generator(exp, args).conditioning_parameter_names()
    return aggregate_generator(
        client_states, client_counts, client_class_counts_list, exp.meta.num_classes,
        conditioning_keys, mechanism_a=bool(args.mechanism_a), beta=args.mech_a_beta,
    )


def _target_net_round(exp, args, gen_global_state, client_ids, label_samplers):
    per_group_states = [[] for _ in range(args.num_models)]
    per_group_counts = [[] for _ in range(args.num_models)]
    losses = []

    gennet = None
    if args.aid_by_gen and gen_global_state is not None:
        gennet = build_generator(exp, args)
        gennet.load_state_dict(gen_global_state)
        gennet.eval()

    for cid in client_ids:
        loader = _client_loader(exp, cid, args)
        if len(loader.dataset) == 0:
            continue
        net = build_target_net(exp, cid, args)
        updater = LocalUpdate(args, loader)
        new_state, loss, fidelity_feedback = updater.train(net, gennet=gennet, label_sampler=label_samplers.get(cid))
        group = exp.dev_spec_idx[cid]
        per_group_states[group].append(new_state)
        per_group_counts[group].append(len(loader.dataset))
        losses.append(loss)

        if args.mechanism_b and fidelity_feedback and cid in label_samplers:
            label_samplers[cid].update_fidelity(fidelity_feedback)

    new_ws_glob = model_wise_FedAvg(exp.ws_glob, per_group_states, per_group_counts)
    return new_ws_glob, (sum(losses) / len(losses) if losses else float("nan"))


def run_gefl(args) -> dict:
    set_seed(args.seed)
    logger = ExperimentLogger(args)
    exp = setup_experiment(args)
    logger.console.info("dataset=%s classes=%d clients=%d imbalance_factor=%.3f dir_param=%.3f "
                         "gen_model=%s mechanism_a=%d mechanism_b=%d",
                         exp.meta.name, exp.meta.num_classes, args.num_users, args.imbalance_factor,
                         args.dir_param, args.gen_model, args.mechanism_a, args.mechanism_b)

    client_sampler = ClientSampler(args.num_users, args.frac, args.seed)
    natural_counts = {cid: np.array([exp.client_class_counts[cid].get(c, 0) for c in range(exp.meta.num_classes)])
                       for cid in range(args.num_users)}
    global_natural_counts = np.array([exp.class_counts[c] for c in range(exp.meta.num_classes)])
    label_samplers = {cid: build_label_sampler(args, exp.meta.num_classes, global_natural_counts)
                       for cid in range(args.num_users)} if args.aid_by_gen else {}

    gen_global_state = None
    gen_opt_states = {}
    local_gen_update_fn = get_local_gen_update(args.gen_model) if args.aid_by_gen else None

    if args.aid_by_gen:
        init_gen = build_generator(exp, args)
        gen_global_state = init_gen.state_dict()
        logger.console.info("warm-up: training shared %s generator for %d rounds", args.gen_model, args.gen_wu_epochs)
        for rnd in range(args.gen_wu_epochs):
            client_ids = client_sampler.select()
            gen_global_state = _generator_round(exp, args, gen_global_state, gen_opt_states,
                                                 client_ids, local_gen_update_fn)

    history = []
    t_start = time.time()
    for rnd in range(args.epochs):
        client_ids = client_sampler.select()

        if args.aid_by_gen and not args.freeze_gen:
            gen_global_state = _generator_round(exp, args, gen_global_state, gen_opt_states,
                                                 client_ids, local_gen_update_fn)

        exp.ws_glob, avg_loss = _target_net_round(exp, args, gen_global_state, client_ids, label_samplers)

        if (rnd + 1) % args.sample_test == 0 or rnd == args.epochs - 1:
            client_models = {}
            for cid in range(args.num_users):
                net = build_target_net(exp, cid, args)
                client_models[cid] = net
            scores = average_client_bucketed_accuracy(client_models, exp.dataset_test, exp.buckets, args.device)
            row = {"round": rnd + 1, "train_loss": avg_loss, **{f"acc_{k}": v for k, v in scores.items()},
                   "elapsed_s": round(time.time() - t_start, 1)}
            if args.mechanism_b and label_samplers:
                mean_fid = np.mean([s.state().get("mean_fidelity", float("nan")) for s in label_samplers.values()])
                row["mean_fidelity"] = float(mean_fid)
            logger.log(row, step=rnd + 1)
            history.append(row)

        if args.save_ckpt:
            save_checkpoint(ckpt_path(args), rnd, exp.ws_glob, gen_global_state, gen_opt_states,
                             {cid: s.state() for cid, s in label_samplers.items()}, args)

    results = {"history": history, "final_scores": history[-1] if history else {}}

    if args.eval_centralized_upper_bound:
        net_cls = exp.target_net_classes[0]
        central_model = train_centralized_upper_bound(
            net_cls, exp.dataset_train, exp.meta.num_classes, exp.meta.in_channels, exp.meta.native_img_size, args
        )
        central_scores, _ = bucketed_accuracy(central_model, exp.dataset_test, exp.buckets, args.device)
        results["centralized_upper_bound"] = central_scores
        federated_scores = {k.replace("acc_", ""): v for k, v in history[-1].items() if k.startswith("acc_")}
        results["gap"] = gap_report(federated_scores, central_scores)
        logger.console.info("centralized upper bound: %s", central_scores)
        logger.console.info("centralized-vs-federated gap: %s", results["gap"])

    if args.aid_by_gen and gen_global_state is not None:
        gen_for_eval = build_generator(exp, args)
        gen_for_eval.load_state_dict(gen_global_state)
        gen_for_eval.eval()
        val_ds, remaining_train_idx = make_held_out_val_split(exp.dataset_train, val_fraction=0.1, seed=args.seed)
        with torch.no_grad():
            labels = torch.randint(0, exp.meta.num_classes, (200,), device=args.device)
            synth = gen_for_eval.sample(labels)
            train_sample = torch.stack([exp.dataset_train[i][0] for i in remaining_train_idx[:2000]]).to(args.device)
            val_sample = torch.stack([val_ds[i][0] for i in range(min(len(val_ds), 2000))]).to(args.device)
        results["mnd_ratio"] = mnd_ratio(synth, train_sample, val_sample)
        logger.console.info("MND privacy ratio: %.4f (near 1 = no detectable memorization signal)",
                             results["mnd_ratio"])

    save_results(os.path.join(args.out_dir, f"{args.name}_results.pkl"), results)
    logger.close()
    return results
