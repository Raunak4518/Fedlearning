"""K14 - Kaggle run: K09 ablation extended to 5 seeds, plus MC-B in the final method (MNIST / FMNIST long tail).

Why: K09 (3 seeds) found every removal lowers balanced accuracy (30/30 seed-level differences),
but no single component survives a Holm correction with n = 3; MC, KH and PC on FMNIST are
0.3-1 point effects. Seeds 3 and 4 give 10/10 paired differences per component (sign test
p = 0.002). MC-B (exact class covariance where the class's sample size determines it, the
generator's own shape where not; E32) beat Ours-A on FMNIST LT in all 3 seeds (+1.2 to +2.2)
but lost 2.7 on MNIST seed 2; this job tests it paired against Ours-A on seeds 0-4.

Arms, seeds 3 and 4 only: the six K09 arms (Ours-A, -LA, -CSL, -MC, -KH, -PC) and Ours-A-B
(MC-B instead of MC). Seeds 0-2 already exist and are NOT re-run: Ours-A and the ablations in
K09, Ours-A-B in E32 (same splits and seeds) - merge those runs.jsonl files for the analysis
(`python ledger.py --check K14_kaggle_ablation_seeds_mcb` shows 0 duplicates).
All PC arms of one seed share one generator, so Ours-A / Ours-A-B / the head ablations cost
only their head stage. Long tail IF = 100, Dirichlet(0.5), K = 10 - the same splits as K09.

Typical use (rough time on T4 x2: ~2 h per dataset):
    python K14.py --datasets mnist  --out_dir /kaggle/working/results
    python K14.py --datasets fmnist --out_dir /kaggle/working/results
"""
ABLATION = [("Ours-A Ts=10", "PC+LA+CSL+MC+KH"), ("-LA", "PC+CSL+MC+KH"), ("-CSL", "PC+LA+MC+KH"),
            ("-MC", "PC+LA+CSL+KH"), ("-KH", "PC+LA+CSL+MC"), ("-PC (CVAE+HWA)", "HWA+LA+CSL+MC+KH")]
MCB = ("Ours-A-B Ts=10", "PC+LA+CSL+MCB+KH")


def _plan(cfg):
    runs = []
    for ds in ["mnist", "fmnist"]:
        for s in cfg["seeds"]:
            runs += [run_spec(lab, ds, s, compose(c, head=dict(ts=10)), IF=0.01, alpha=0.5)
                     for lab, c in ABLATION + [MCB]]
    return runs


EXPERIMENT = dict(
    name="K14_kaggle_ablation_seeds_mcb",
    title="K09 ablation and MC-B at seeds 3-4 (MNIST / FMNIST long tail)",
    hypothesis="every removal lowers balanced accuracy in 5/5 seeds; Ours-A-B >= Ours-A in 5/5 seeds (with K09 / E32 seeds 0-2).",
    reference="Ours-A Ts=10",
    seeds=[3, 4],
    split_by_gen=True,
    metrics=["final_bal", "final_tail", "bbc_bal", "best_mean_acc", "final_worst", "fidelity_tail_kh", "spread_tail_kh"],
    plan=_plan,
)
