"""E30 - MC-S: anchoring the generator to the exact class covariance, where class means carry no class identity (SVHN, 3 seeds).

Finding: on SVHN the anchored stack Ours-A loses to CVAE + HWA (55.9 vs 62.5 under the
long tail). Cause (Proposition 5): its class identity comes from the class mean, and SVHN's
class means carry none - nearest-class-mean 10 %, LDA 21 % in pixel space. The class
COVARIANCES do: QDA reaches 54 %. Proxy (class-agnostic samples, a head trained only on
corrected synthetic data, tested on real data): mean + spread correction (MC) 12 %
(long tail) / 11 % (balanced), mean + covariance correction (MC-S) 29 % / 44 %.
MC-S maps each generated class to the exact class mean and class covariance (top-256
principal subspace; exact spread outside it) with the Gelbrich map - the
minimum-displacement map achieving both (Proposition 6). Both arms share FE and
generator per seed, so the comparison is exactly paired.
"""
ARMS = [("Ours-A (PC+MC+KH+LA+CSL)", "PC+LA+CSL+MC+KH"), ("Ours-A-S (PC+MCS+KH+LA+CSL)", "PC+LA+CSL+MCS+KH")]
EXPERIMENT = dict(
    name="E30_covariance_anchoring",
    title="MC-S: exact class-covariance anchoring on SVHN (long tail + IID)",
    hypothesis="Ours-A-S > Ours-A on SVHN in both regimes, closing most of the gap to CVAE + HWA under the long tail.",
    reference="Ours-A (PC+MC+KH+LA+CSL)",
    metrics=["final_bal", "final_tail", "best_mean_acc", "bbc_bal", "fidelity_tail_mc", "fidelity_tail_kh", "spread_tail_mc",
             "class_sep_F", "ncm_acc"],
    plan=lambda cfg: [run_spec(lab, "svhn", s, compose(c), IF=IF, alpha=a)
                      for IF, a in [(0.01, 0.5), (1.0, None)] for s in cfg["seeds"] for lab, c in ARMS],
)
