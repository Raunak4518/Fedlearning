#!/usr/bin/env python3
"""
Baseline_CReFF.py

Runs the CReFF classifier-side baseline (see baselines/creff.py) on the
same dataset / partition / long-tail settings as the GeFL_*.py scripts,
for a like-for-like comparison table.

    python Baseline_CReFF.py --config configs/cifar10_lt.yaml --name creff_cifar10lt
"""
from args import parse_args, save_args
from baselines.creff import run_creff
from utils.checkpoint import save_results
from utils.evaluate import bucketed_accuracy, gap_report, train_centralized_upper_bound
from utils.logger import ExperimentLogger
from utils.seed import set_seed
from utils.setup import setup_experiment


def main():
    args = parse_args()
    args.aid_by_gen = 0  # CReFF never uses a generator
    args.name = args.name if args.name != "run" else "creff_baseline"
    set_seed(args.seed)
    save_args(args, f"{args.out_dir}/{args.name}_args.json")

    logger = ExperimentLogger(args)
    exp = setup_experiment(args)
    logger.console.info("CReFF baseline | dataset=%s classes=%d clients=%d imbalance_factor=%.3f dir_param=%.3f",
                         exp.meta.name, exp.meta.num_classes, args.num_users, args.imbalance_factor, args.dir_param)

    out = run_creff(exp, args)
    scores, per_class = bucketed_accuracy(out["combined"], exp.dataset_test, exp.buckets, args.device)
    logger.log({"phase": "creff_final", **{f"acc_{k}": v for k, v in scores.items()}})

    results = {"final_scores": scores, "per_class_accuracy": per_class}

    if args.eval_centralized_upper_bound:
        net_cls = exp.target_net_classes[0]
        central_model = train_centralized_upper_bound(
            net_cls, exp.dataset_train, exp.meta.num_classes, exp.meta.in_channels, exp.meta.native_img_size, args
        )
        central_scores, _ = bucketed_accuracy(central_model, exp.dataset_test, exp.buckets, args.device)
        results["centralized_upper_bound"] = central_scores
        results["gap"] = gap_report(scores, central_scores)
        logger.console.info("centralized-vs-CReFF gap: %s", results["gap"])

    save_results(f"{args.out_dir}/{args.name}_results.pkl", results)
    logger.close()
    return results


if __name__ == "__main__":
    main()
