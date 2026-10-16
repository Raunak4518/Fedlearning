"""K20 - Kaggle run: mechanism controls for kernel herding and moment correction (action plan C2).

Reviewers asked whether KH's *selection* matters or any subset of the candidate pool would do, and which of MC's two
moments matters. One generator per seed is shared by every arm (cached within the job), so only the head stage differs:
  Ours (reference)      GATE + MC + KH + LA + CSL (within-job reference: same generator for every arm)
  KH -> random subset   the same 4n-candidate pool, a uniformly random n instead of herding
  no KH                 MC samples used directly
  MC mean only          MC moves the class mean, no spread rescaling
  MC spread only        MC rescales the spread, keeps the generator's own mean
  KH, RFF dim 1536      (default 6144)
  KH, pool 8n           (default 4n)
FashionMNIST (T_s = 10; KH and MC effects were measured there) and CIFAR-10 (T_s = 1), long tail, seeds 3-4.
The reference arm repeats K14 / K17 configurations at seeds 3-4 on purpose: a within-job reference shares the
generator, which a 0.3-1 point mechanism effect needs.
PRE-REGISTERED (17 Oct 2026): M1 herding beats the random subset by > 0 on both datasets (if not, KH's benefit is the
larger pool or nothing); M2 MC's gain comes mainly from the spread (spread-only within 0.5 of full MC, mean-only not);
M3 results are flat in RFF dimension and pool size (within 0.5).
"""
TS = {"fmnist": 10, "cifar10": 1}
B = "GATE+MC+KH+LA+CSL"
ARMS = [("Ours (reference)", B), ("KH -> random subset", B + "+KHRAND"), ("no KH", "GATE+MC+LA+CSL"),
        ("MC mean only", B + "+MCMEAN"), ("MC spread only", B + "+MCSPREAD"), ("KH, RFF dim 1536", B + "+KHD1536"),
        ("KH, pool 8n", B + "+KHM8")]
EXPERIMENT = dict(
    name="K20_kaggle_mechanism_controls",
    title="Mechanism controls for kernel herding and moment correction (shared generator per seed)",
    hypothesis="M1-M3 in the docstring.",
    reference="Ours (reference)",
    seeds=[3, 4],
    metrics=["final_bal", "er_bal", "final_tail", "fidelity_tail_kh", "spread_tail_kh", "val_final_bal"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c, head=dict(ts=TS[ds])), IF=0.01, alpha=0.5, K=10)
                      for ds in ["fmnist", "cifar10"] for s in cfg["seeds"] for lab, c in ARMS],
)
