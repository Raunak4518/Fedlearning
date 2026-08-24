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
  4. periodic evaluation + logging + checkpointing + visualization.
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
                             make_held_out_val_split, mnd_ratio, train_centralized_upper_bound,
                             ConvergenceTracker, classification_report_from_cm,
                             compute_confusion_matrix, per_class_accuracy, per_client_accuracy)
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
    """Returns (new_global_state, mean_gen_loss)."""
    client_states, client_counts, client_class_counts_list = [], [], []
    gen_losses = []
    for cid in client_ids:
        net = build_generator(exp, args)
        if gen_global_state is not None:
            net.load_state_dict(gen_global_state)
        loader = _client_loader(exp, cid, args)
        if len(loader.dataset) == 0:
            continue
        new_state, loss, new_opt_state = local_gen_update_fn(net, loader, args, gen_opt_states.get(cid))
        gen_opt_states[cid] = new_opt_state
        client_states.append(new_state)
        client_counts.append(len(loader.dataset))
        client_class_counts_list.append(exp.client_class_counts[cid])
        gen_losses.append(loss)

    if not client_states:
        return gen_global_state, float("nan")

    conditioning_keys = build_generator(exp, args).conditioning_parameter_names()
    new_state = aggregate_generator(
        client_states, client_counts, client_class_counts_list, exp.meta.num_classes,
        conditioning_keys, mechanism_a=bool(args.mechanism_a), beta=args.mech_a_beta,
    )
    mean_gen_loss = sum(gen_losses) / len(gen_losses) if gen_losses else float("nan")
    return new_state, mean_gen_loss


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


def _build_all_client_models(exp, args):
    """Build and return all client models with current global weights."""
    client_models = {}
    for cid in range(args.num_users):
        net = build_target_net(exp, cid, args)
        client_models[cid] = net
    return client_models


def run_gefl(args) -> dict:
    set_seed(args.seed)
    logger = ExperimentLogger(args)
    exp = setup_experiment(args)
    plots_dir = os.path.join(args.out_dir, "plots", args.name)
    os.makedirs(plots_dir, exist_ok=True)

    logger.console.info(
        "dataset=%s classes=%d clients=%d imbalance_factor=%.3f dir_param=%.3f "
        "gen_model=%s mechanism_a=%d mechanism_b=%d",
        exp.meta.name, exp.meta.num_classes, args.num_users, args.imbalance_factor,
        args.dir_param, args.gen_model, args.mechanism_a, args.mechanism_b
    )

    client_sampler = ClientSampler(args.num_users, args.frac, args.seed)
    natural_counts = {cid: np.array([exp.client_class_counts[cid].get(c, 0) for c in range(exp.meta.num_classes)])
                       for cid in range(args.num_users)}
    global_natural_counts = np.array([exp.class_counts[c] for c in range(exp.meta.num_classes)])
    label_samplers = {cid: build_label_sampler(args, exp.meta.num_classes, global_natural_counts)
                       for cid in range(args.num_users)} if args.aid_by_gen else {}

    gen_global_state = None
    gen_opt_states = {}
    local_gen_update_fn = get_local_gen_update(args.gen_model) if args.aid_by_gen else None

    # ---- Plot data distribution at startup --------------------------------
    try:
        from utils.visualize import plot_class_distribution
        plot_class_distribution(
            exp.class_counts, exp.client_class_counts,
            exp.meta.num_classes,
            os.path.join(plots_dir, "data_distribution.png"),
            title=f"Data Distribution — {exp.meta.name} (IF={args.imbalance_factor})"
        )
        logger.console.info("Saved data distribution plot")
    except Exception as e:
        logger.console.warning("Could not generate data distribution plot: %s", e)

    # ---- Generator warm-up ------------------------------------------------
    if args.aid_by_gen:
        init_gen = build_generator(exp, args)
        gen_global_state = init_gen.state_dict()
        logger.console.info("warm-up: training shared %s generator for %d rounds", args.gen_model, args.gen_wu_epochs)
        for rnd in range(args.gen_wu_epochs):
            client_ids = client_sampler.select()
            gen_global_state, wu_gen_loss = _generator_round(exp, args, gen_global_state, gen_opt_states,
                                                              client_ids, local_gen_update_fn)
            logger.console.debug("warm-up round %d/%d gen_loss=%.4f", rnd + 1, args.gen_wu_epochs, wu_gen_loss)

    # ---- Main training loop -----------------------------------------------
    history = []
    per_class_history = []
    fidelity_history = []
    convergence_history = []
    timing_history = []
    convergence_tracker = ConvergenceTracker(patience=5)
    t_start = time.time()

    for rnd in range(args.epochs):
        logger.log_round_start(rnd)
        client_ids = client_sampler.select()
        gen_loss = float("nan")

        # Generator round
        if args.aid_by_gen and not args.freeze_gen:
            logger.round_timer.start("gen")
            gen_global_state, gen_loss = _generator_round(exp, args, gen_global_state, gen_opt_states,
                                                           client_ids, local_gen_update_fn)
            logger.round_timer.stop()

        # Target network round
        logger.round_timer.start("target")
        exp.ws_glob, avg_loss = _target_net_round(exp, args, gen_global_state, client_ids, label_samplers)
        logger.round_timer.stop()

        # Evaluation
        if (rnd + 1) % args.sample_test == 0 or rnd == args.epochs - 1:
            logger.round_timer.start("eval")

            client_models = _build_all_client_models(exp, args)
            from utils.evaluate import average_client_metrics
            scores = average_client_metrics(client_models, exp.dataset_test, exp.meta.num_classes, exp.buckets, args.device)

            # Core row
            row = {
                "round": rnd + 1,
                "train_loss": avg_loss,
                **{f"acc_{k}": v for k, v in scores.items() if k in ["overall", "head", "medium", "tail"]},
                "elapsed_s": round(time.time() - t_start, 1),
            }

            # Generator loss
            if args.aid_by_gen and not args.freeze_gen:
                row["gen_loss"] = gen_loss

            # Per-class accuracy and classification report
            if getattr(args, "log_per_class", 1):
                # Use the first client model for per-class metrics history
                first_model = list(client_models.values())[0]
                cm = compute_confusion_matrix(
                    first_model, exp.dataset_test, exp.meta.num_classes, args.device
                )
                pca = per_class_accuracy(cm)
                per_class_history.append(pca)

                # Classification report values populated from scores object (averaged across clients)
                row["macro_f1"] = scores["macro_f1"]
                row["weighted_f1"] = scores["weighted_f1"]
                row["class_balanced_accuracy"] = scores["class_balanced_accuracy"]
                row["macro_precision"] = scores["macro_precision"]
                row["macro_recall"] = scores["macro_recall"]

            # Per-client accuracy
            if getattr(args, "log_per_client", 0):
                client_accs = per_client_accuracy(client_models, exp.dataset_test, args.device)
                row["client_acc_mean"] = float(np.mean(list(client_accs.values())))
                row["client_acc_std"] = float(np.std(list(client_accs.values())))
                row["client_acc_min"] = float(min(client_accs.values()))
                row["client_acc_max"] = float(max(client_accs.values()))

            # Generator quality metrics
            if args.aid_by_gen and gen_global_state is not None:
                try:
                    gennet_eval = build_generator(exp, args)
                    gennet_eval.load_state_dict(gen_global_state)
                    gennet_eval.eval()
                    gen_quality = generator_quality_metrics(
                        gennet_eval, first_model if getattr(args, "log_per_class", 1) else list(client_models.values())[0],
                        exp.dataset_train, exp.meta.num_classes, args.device
                    )
                    row["gen_mean_confidence"] = gen_quality["gen_mean_confidence"]
                    row["gen_label_accuracy"] = gen_quality["gen_label_accuracy"]
                except Exception as e:
                    logger.console.debug("Generator quality metrics failed: %s", e)

            # Fidelity tracking (Mechanism B)
            if args.mechanism_b and label_samplers:
                mean_fid = np.mean([s.state().get("mean_fidelity", float("nan")) for s in label_samplers.values()])
                row["mean_fidelity"] = float(mean_fid)
                # Collect fidelity state from the first client for per-class tracking
                first_sampler = list(label_samplers.values())[0]
                fidelity_state = first_sampler.state()
                fidelity_history.append({
                    "mean_fidelity": float(mean_fid),
                    "fidelity_per_class": fidelity_state.get("fidelity_per_class", []),
                })

            # Convergence tracking
            conv_info = convergence_tracker.update(scores.get("overall", 0))
            row["acc_delta"] = conv_info["acc_delta"]
            row["best_acc"] = conv_info["best_acc"]
            convergence_history.append(conv_info)
            if conv_info["is_plateau"]:
                logger.console.warning("Plateau detected at round %d (no improvement for %d eval steps)",
                                        rnd + 1, conv_info["rounds_without_improvement"])
            if conv_info["is_diverging"]:
                logger.console.warning("Possible divergence detected at round %d", rnd + 1)

            logger.round_timer.stop()

            # Timing
            timing_history.append(logger.round_timer.get_timings())

            # Log to all sinks
            logger.log(row, step=rnd + 1)
            history.append(row)

            # Log synthetic images to TensorBoard
            if args.aid_by_gen and gen_global_state is not None and logger.tb_writer is not None:
                try:
                    gennet_tb = build_generator(exp, args)
                    gennet_tb.load_state_dict(gen_global_state)
                    gennet_tb.eval()
                    sample_labels = torch.arange(min(exp.meta.num_classes, 10), device=args.device).repeat(2)
                    with torch.no_grad():
                        sample_imgs = gennet_tb.sample(sample_labels)
                    logger.log_image_grid("synthetic_samples", sample_imgs, rnd + 1)
                except Exception:
                    pass

            # Intermediate plots
            if (getattr(args, "plot_every", 0) > 0 and
                    (rnd + 1) % args.plot_every == 0 and len(history) > 1):
                _generate_intermediate_plots(history, per_class_history, convergence_history,
                                             timing_history, exp, args, plots_dir, rnd + 1)

        else:
            # Non-eval round: still track timing
            timing_history.append(logger.round_timer.get_timings())

        logger.log_round_end(rnd)

        if args.save_ckpt:
            save_checkpoint(ckpt_path(args), rnd, exp.ws_glob, gen_global_state, gen_opt_states,
                             {cid: s.state() for cid, s in label_samplers.items()}, args)

    # ---- Final evaluation & visualization ---------------------------------
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
            # Generate enough synthetic samples for MND reference set (paper: |S|=600)
            labels = torch.randint(0, exp.meta.num_classes, (600,), device=args.device)
            synth = gen_for_eval.sample(labels)
            train_sample = torch.stack([exp.dataset_train[i][0] for i in remaining_train_idx[:2000]]).to(args.device)
            val_sample = torch.stack([val_ds[i][0] for i in range(min(len(val_ds), 600))]).to(args.device)
        results["mnd_ratio"] = mnd_ratio(train_sample, synth, val_sample,
                                          n_query=1000, n_ref=600, device=args.device,
                                          dataset_mean=exp.meta.mean, dataset_std=exp.meta.std)
        logger.console.info("MND privacy ratio: %.4f (< 1 = good generalization, > 1 = potential memorization)",
                             results["mnd_ratio"])

    # ---- Generate final plots ---------------------------------------------
    _generate_final_plots(history, per_class_history, fidelity_history,
                          convergence_history, timing_history, exp, args,
                          plots_dir, gen_global_state, results)

    # ---- Final summary ----------------------------------------------------
    summary = {}
    if history:
        final = history[-1]
        for k in ("acc_overall", "acc_head", "acc_medium", "acc_tail",
                  "macro_f1", "class_balanced_accuracy", "train_loss"):
            if k in final:
                summary[k] = final[k]
    if "mnd_ratio" in results:
        summary["mnd_ratio"] = results["mnd_ratio"]
    if "gap" in results:
        summary["centralized_gap"] = results["gap"]
    summary["total_time_s"] = round(time.time() - t_start, 1)
    logger.log_summary(summary)

    save_results(os.path.join(args.out_dir, f"{args.name}_results.pkl"), results)
    logger.close()

    # Auto-display plots if requested
    if getattr(args, "show_plots", 0):
        try:
            from utils.visualize import show_all_plots
            show_all_plots(plots_dir)
        except Exception:
            pass

    return results


def _generate_intermediate_plots(history, per_class_history, convergence_history,
                                  timing_history, exp, args, plots_dir, current_round):
    """Generate plots at intermediate checkpoints."""
    try:
        from utils.visualize import (plot_training_curves, plot_convergence,
                                      plot_timing_breakdown)

        eval_rounds = [r["round"] for r in history]

        plot_training_curves(
            history,
            os.path.join(plots_dir, f"training_curves_r{current_round}.png"),
            title=f"Training Progress (Round {current_round})"
        )

        if convergence_history:
            plot_convergence(
                convergence_history, eval_rounds,
                os.path.join(plots_dir, f"convergence_r{current_round}.png"),
            )

        if timing_history:
            all_rounds = list(range(1, current_round + 1))
            plot_timing_breakdown(
                timing_history[:len(all_rounds)], all_rounds,
                os.path.join(plots_dir, f"timing_r{current_round}.png"),
            )
    except Exception as e:
        pass  # don't crash training for a plot failure


def _generate_final_plots(history, per_class_history, fidelity_history,
                           convergence_history, timing_history, exp, args,
                           plots_dir, gen_global_state, results):
    """Generate all final visualizations."""
    try:
        from utils import visualize as viz
    except ImportError:
        return

    eval_rounds = [r["round"] for r in history]
    logger_name = args.name

    # 1. Training curves
    try:
        if len(history) > 1:
            viz.plot_training_curves(
                history,
                os.path.join(plots_dir, "training_curves.png"),
                title=f"{logger_name} — Training Progress"
            )
    except Exception:
        pass

    # 2. Per-class accuracy heatmap
    try:
        if per_class_history:
            viz.plot_per_class_accuracy_heatmap(
                per_class_history, eval_rounds[:len(per_class_history)],
                exp.meta.num_classes,
                os.path.join(plots_dir, "per_class_accuracy_heatmap.png"),
            )
    except Exception:
        pass

    # 3. Confusion matrix
    try:
        if history:
            client_models = _build_all_client_models(exp, args)
            from utils.evaluate import compute_confusion_matrix as _cm
            import numpy as np
            cm = np.zeros((exp.meta.num_classes, exp.meta.num_classes), dtype=np.int64)
            for m in client_models.values():
                cm += _cm(m, exp.dataset_test, exp.meta.num_classes, args.device)
            viz.plot_confusion_matrix(
                cm, os.path.join(plots_dir, "confusion_matrix.png"),
                title=f"{logger_name} — Final Confusion Matrix"
            )

            # Classification report bar chart
            from utils.evaluate import classification_report_from_cm
            report = classification_report_from_cm(cm)
            viz.plot_classification_report(
                report, os.path.join(plots_dir, "classification_report.png"),
                title=f"{logger_name} — Precision / Recall / F1"
            )
            results["classification_report"] = report
    except Exception:
        pass

    # 4. Fidelity evolution (Mechanism B)
    try:
        if fidelity_history:
            viz.plot_fidelity_evolution(
                fidelity_history, eval_rounds[:len(fidelity_history)],
                exp.meta.num_classes,
                os.path.join(plots_dir, "fidelity_evolution.png"),
            )
    except Exception:
        pass

    # 5. Convergence
    try:
        if convergence_history:
            viz.plot_convergence(
                convergence_history, eval_rounds[:len(convergence_history)],
                os.path.join(plots_dir, "convergence.png"),
            )
    except Exception:
        pass

    # 6. Timing breakdown
    try:
        if timing_history:
            all_rounds = list(range(1, args.epochs + 1))
            viz.plot_timing_breakdown(
                timing_history[:len(all_rounds)], all_rounds[:len(timing_history)],
                os.path.join(plots_dir, "timing_breakdown.png"),
            )
    except Exception:
        pass

    # 7. Per-client accuracy
    try:
        if getattr(args, "log_per_client", 0) and history:
            client_models = _build_all_client_models(exp, args)
            from utils.evaluate import per_client_accuracy
            client_accs = per_client_accuracy(client_models, exp.dataset_test, args.device)
            viz.plot_per_client_accuracy(
                client_accs,
                os.path.join(plots_dir, "per_client_accuracy.png"),
            )
    except Exception:
        pass

    # 8. Synthetic sample grid
    try:
        if (getattr(args, "save_synthetic_samples", 1) and
                args.aid_by_gen and gen_global_state is not None):
            gennet = build_generator(exp, args)
            gennet.load_state_dict(gen_global_state)
            gennet.eval()
            n_per_class = 5
            sample_labels = torch.arange(exp.meta.num_classes, device=args.device).repeat(n_per_class)
            with torch.no_grad():
                sample_imgs = gennet.sample(sample_labels)
            viz.plot_synthetic_samples(
                sample_imgs.cpu().numpy(), sample_labels.cpu().numpy(),
                os.path.join(plots_dir, "synthetic_samples.png"),
                num_classes=exp.meta.num_classes,
                samples_per_class=n_per_class,
                title=f"{logger_name} — Generated Samples"
            )
    except Exception:
        pass

    # 9. Bucket comparison (if centralized upper bound available)
    try:
        if "centralized_upper_bound" in results and history:
            federated = {k.replace("acc_", ""): v for k, v in history[-1].items() if k.startswith("acc_")}
            method_scores = {
                "Federated (GeFL)": federated,
                "Centralized Upper Bound": results["centralized_upper_bound"],
            }
            viz.plot_bucket_comparison(
                method_scores,
                os.path.join(plots_dir, "bucket_comparison.png"),
                title=f"{logger_name} — Federated vs. Centralized"
            )
    except Exception:
        pass
