"""
experiments/reference_check/run_reference.py

Runs the AUTHORS' GeFL-F code (github.com/honggkang/hetero-model-fl-gen,
GeFL_CVAE-F.py) unmodified, to check that our re-implementation's GeFL-F
baseline matches theirs. Only three things are injected from outside:

  1. device: their args.py hardcodes args.device = 'cpu'; we set cuda:0.
  2. evaluation: their evaluate_models() is wrapped so that, besides their
     best-over-rounds accuracy, the final-round accuracy and per-class
     recall of every architecture are recorded (same test_img forward).
  3. --setting lt: their dict_iid() split is replaced by our long-tail
     Dirichlet split (experiments/core.py make_partition, same seed), i.e.
     the identical client indices our F01 runs used.

    python run_reference.py --ref_dir <copy of their repo> --setting iid --seed 0 --avg_FE 1
    python run_reference.py --ref_dir <copy> --setting lt  --seed 0 --avg_FE 1

Results are appended to reference_runs.jsonl next to this file.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--setting", choices=["iid", "lt"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--avg_FE", type=int, default=1)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()

    os.chdir(a.ref_dir)
    sys.path.insert(0, a.ref_dir)
    # Paper Table XIV/XV values for MNIST; everything else is their default.
    sys.argv = ["GeFL_CVAE-F.py", "--dataset", "mnist", "--models", "cnn", "--num_classes", "10",
                "--orig_img_size", "32", "--partial_data", "0.1", "--num_users", "10", "--local_bs", "64",
                "--latent_size", "16", "--aid_by_gen", "1", "--num_experiment", "1", "--rs", str(a.seed),
                "--avg_FE", str(a.avg_FE)] + a.extra
    spec = importlib.util.spec_from_file_location("gefl_cvae_f", os.path.join(a.ref_dir, "GeFL_CVAE-F.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # defines main(); the __main__ block does not run
    args = mod.parse_args()
    args.gen_model = "vaef"
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mod.args = args

    if a.setting == "lt":
        sys.path.insert(0, os.path.dirname(HERE))
        import core  # our partitioner, identical indices to the F01 runs
        import utils.setup as ref_setup
        orig_get = ref_setup.getDataset

        def lt_dict_iid(dataset, _num_shards):
            part = core.make_partition(dataset.targets.clone(), args.num_users, 0.1, 0.01, 0.5, a.seed)
            return {k: set(p.tolist()) for k, p in enumerate(part["parts"])}

        ref_setup.dict_iid = lt_dict_iid
        _ = orig_get

    hist = []
    orig_eval = mod.evaluate_models

    def eval_and_record(local_models, ws_glob, dataset_test, args_, it, best_perf):
        from torch.utils.data import DataLoader
        accs, recalls = [], []
        loader = DataLoader(dataset_test, batch_size=1000)
        for i in range(args_.num_models):
            net = local_models[i]
            net.load_state_dict(ws_glob[i])
            net.to(args_.device)
            net.eval()
            correct = torch.zeros(10)
            total = torch.zeros(10)
            with torch.no_grad():
                for x, y in loader:
                    pred = net(x.to(args_.device))[0].argmax(1).cpu()
                    for c in range(10):
                        m = y == c
                        total[c] += m.sum()
                        correct[c] += (pred[m] == c).sum()
            rec = (correct / total.clamp(min=1)).numpy()
            recalls.append(rec)
            accs.append(float(correct.sum() / total.sum()))
        rec = np.mean(recalls, 0)
        hist.append(dict(round=it, acc=float(np.mean(accs)), bal=float(rec.mean()), tail=float(rec[6:].mean()),
                         recall=rec.round(4).tolist()))
        return orig_eval(local_models, ws_glob, dataset_test, args_, it, best_perf)

    mod.evaluate_models = eval_and_record
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    np.random.seed(a.seed)
    import random
    random.seed(a.seed)
    t0 = time.time()
    best = mod.main()
    row = dict(implementation="authors' GeFL_CVAE-F.py", setting=a.setting, seed=a.seed, avg_FE=a.avg_FE,
               best_mean_acc=float(best) / 100.0, final_acc=hist[-1]["acc"], final_bal=hist[-1]["bal"],
               final_tail=hist[-1]["tail"], final_recall=hist[-1]["recall"], time_s=round(time.time() - t0),
               history=[{k: v for k, v in h.items() if k != "recall"} for h in hist], argv=sys.argv)
    with open(os.path.join(HERE, "reference_runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print("REFERENCE RESULT", json.dumps({k: row[k] for k in ["setting", "seed", "avg_FE", "best_mean_acc",
                                                              "final_acc", "final_bal", "final_tail", "time_s"]}))


if __name__ == "__main__":
    main()
