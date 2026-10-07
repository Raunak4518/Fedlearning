"""Print one line per finished run: python experiments/peek.py [results_dir]"""
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
for f in sorted(glob.glob(os.path.join(root, "*", "runs.jsonl"))):
    print(os.path.basename(os.path.dirname(f)))
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        print(f"  {r['label'][:34]:34s} {r['dataset']:8s} IF={r['IF']:<5} a={r['alpha']} K={r['K']:<3} s={r['seed']} "
              f"bal={r['final_bal'] * 100:5.1f} tail={r['final_tail'] * 100:5.1f} best={r['best_mean_acc'] * 100:5.1f} "
              f"orc={r.get('oracle_bal', float('nan')) * 100:5.1f} fid_t={r.get('fidelity_tail', float('nan')):.2f} "
              f"nrm={r.get('cond_norm_tail_over_head_end', float('nan')):.2f} t={r['time_s']:.0f}s")
