"""E20 - Prototype-conditioned residual CVAE (PC-VAE): class identity from exact statistics, not parameters.

Diagnosis (docs/diagnostics.md, sections 2-3): the CVAE-F's only class-specific
parameter is one learned row per class, fitted from that class's few samples and
then diluted or collapsed by FedAvg; even with HWA, MNIST tail fidelity is 27%.
FSG uses the EXACT class mean (secure-aggregated sums) and never degrades with
K, but its Gaussian shape fails on multi-modal data (SVHN).
PC-VAE: x = ReLU(mu_y + dec(z, mu_y)), mu_y the exact federated class mean, fed
through a class-shared projection. No class rows (HWA unnecessary), tail classes
borrow within-class variation from all classes; a linear decoder recovers FSG.
ZP: ex-post latent prior N(m, S) from exact sums of encoder means (prior hole).
MC: sampling-time W2 projection onto the exact class mean and spread (sum h,
sum ||h||^2, n per class) - fixes the measured VAE under-dispersion
(spread ratio 0.1-0.3) for any generator.
Predictions: tail fidelity PC >> CVAE+HWA; accuracy PC >= max(CVAE+HWA, FSG)
across regimes; ZP adds realism; MC raises spread to ~1 and helps both generators.
"""
LABELS = [("GeFL-F", ""), ("Ours (HWA+LA+CSL)", "HWA+LA+CSL"), ("Ours+MC", "HWA+LA+CSL+MC"),
          ("Ours-PC (PC+LA+CSL)", "PC+LA+CSL"), ("Ours-PC+MC", "PC+LA+CSL+MC"),
          ("Ours-PC+ZP (PC+ZP+LA+CSL)", "PC+ZP+LA+CSL"), ("FSG+LA+CSL", "GAUSS+LA+CSL")]
SETTINGS = [("mnist", 0.01, 0.5), ("fmnist", 0.01, 0.5), ("fmnist", 1.0, None)]
EXPERIMENT = dict(
    name="E20_proto_generator",
    title="PC-VAE: prototype-conditioned residual feature generator",
    hypothesis="tail fidelity PC >> CVAE+HWA; accuracy PC >= max(CVAE+HWA, FSG); ZP helps.",
    reference="Ours (HWA+LA+CSL)",
    metrics=["final_bal", "final_tail", "best_mean_acc", "fidelity_tail", "fidelity_head"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c), IF=IF, alpha=a)
                      for ds, IF, a in SETTINGS for s in cfg["seeds"] for lab, c in LABELS],
)
