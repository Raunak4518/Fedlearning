"""K15 - Kaggle run: the existing fixes for conditioning-row collapse, against HWA (MNIST / FMNIST long tail, K = 10/50/100).

Theorem 1 (paper section 3.1): under flat FedAvg a class row contracts when the holders' gain is
smaller than the non-holders' loss; under Adam with COUPLED decay the non-holders' loss is a step of
order lr for any lambda > 0 (Lemma 2). Fixes that already exist outside this paper:
  ADAMW : decoupled weight decay (Loshchilov & Hutter 2019): a non-holder's row shrinks only by
          (1 - lr * wd) per step -> removes the collapse term, keeps flat averaging (dilution m_c / K).
  LCD   : lazy decay, as in lazy Adam for embedding rows: decay and update only the rows present in
          the batch -> non-holders leave the row unchanged; flat averaging still dilutes it.
  HWA   : (ours; the same idea as count-weighted sparse aggregation in federated recommendation)
          average the row over holders only -> removes both the collapse term and the dilution.
Prediction: ADAMW and LCD restore the rows at every K, but their data step is diluted by m_c / K,
so HWA's advantage over them GROWS with K (fewer holders per tail class).

Heads use LA in every arm, so only the generator differs. Reference arms are NOT re-run (same splits
and seeds, merge their runs.jsonl): GeFL-F, +LA, +HWA+LA from F01 (K = 10) and F04 (MNIST K = 50/100),
and from K03 (FMNIST K = 50/100). Check: python ledger.py --check K15_kaggle_collapse_fixes.

Typical use (rough time on T4 x2: ~2 h per dataset):
    python K15.py --datasets mnist  --out_dir /kaggle/working/results
    python K15.py --datasets fmnist --out_dir /kaggle/working/results
"""
ARMS = [("+ADAMW+LA", "ADAMW+LA"), ("+LCD+LA", "LCD+LA")]
EXPERIMENT = dict(
    name="K15_kaggle_collapse_fixes",
    title="Existing fixes for row collapse (AdamW, lazy decay) against HWA, K = 10 / 50 / 100",
    hypothesis="ADAMW and LCD restore the tail rows (ratio ~1) but trail +HWA+LA, by more as K grows.",
    reference="+ADAMW+LA",
    split_by_gen=True,
    metrics=["final_bal", "final_tail", "best_mean_acc", "cond_norm_tail_over_head_end", "fidelity_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c), IF=0.01, alpha=0.5, K=K)
                      for ds in ["mnist", "fmnist"] for K in [10, 50, 100] for s in cfg["seeds"] for lab, c in ARMS],
)
