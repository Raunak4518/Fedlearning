"""F11 - Our method on the paper's strongest feature generator, DDPM-F (full rounds, 3 seeds).

The best FMNIST number of any GeFL variant in the paper is 84.28, from
GeFL-F with the feature diffusion model DDPM-F (Fig. 4b, w = 0); CVAE-F
gives 83.14. HWA, LA and CSL do not depend on the generator: HWA averages
the class-conditioning columns of the context embeddings over holders, LA
and CSL act on the heads. This runs GeFL-F, +CSL and Ours, all on the
authors' DDPM-F (ported from DDPM/ddpm16.py, same schedule and optimiser),
in the paper's IID setting and under the long tail.
"""
DD = "DDPM"
LABELS_IID = [("GeFL-F (DDPM-F)", DD), ("+CSL (DDPM-F)", DD + "+CSL"), ("Ours (DDPM-F)", DD + "+HWA+LA+CSL")]
LABELS_LT = [("GeFL-F (DDPM-F)", DD), ("+LA (DDPM-F)", DD + "+LA"), ("Ours (DDPM-F)", DD + "+HWA+LA+CSL")]
EXPERIMENT = dict(
    name="F11_ddpm_generator",
    title="Our method on DDPM-F, the paper's best feature generator",
    hypothesis="IID FMNIST: Ours (DDPM-F) > 84.28 (paper's best). LT: Ours (DDPM-F) >> GeFL-F (DDPM-F).",
    reference="GeFL-F (DDPM-F)",
    metrics=["best_mean_acc", "final_bal", "final_tail", "final_ens_acc", "fidelity_tail", "cond_norm_tail_over_head_end"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c), IF=IF, alpha=a)
                      for ds in ["fmnist", "mnist"] for IF, a, L in [(1.0, None, LABELS_IID), (0.01, 0.5, LABELS_LT)]
                      for s in cfg["seeds"] for lab, c in L],
)
