"""E18 - Bayesian consensus soft labels (full rounds, 3 seeds, K = 10).

CSL labels a generated feature with the MIXTURE t = (1-b) e_y + b p(.|x)
(b = 0.5 fixed). Under symmetric generator label noise
q(c|y) = rho 1{c=y} + (1-rho)/C, the Bayes posterior of the feature's class
given both sources is the PRODUCT t_c ~ p(c|x) q(c|y). rho is estimated each
round, without real data, from the ensemble-label agreement
a = mean p(y|x): rho = (a - 1/C) / (1 - 1/C). Limits: rho = 1 -> hard labels,
rho = 0 -> pure ensemble. No hyperparameter.
Predictions: BCSL >= CSL; and (Prop. 3) BCSL lowers the label bias b, so the
larger budget T_s = 10 helps more - most clearly on MNIST long tail, where
the generator's tail fidelity is lowest and CSL's budget gain vanished.
"""
FM_IID = [("+CSL Ts=10", "CSL", 10), ("+BCSL", "BCSL", 1), ("+BCSL Ts=10", "BCSL", 10),
          ("Ours-B (HWA+LA+BCSL) Ts=10", "HWA+LA+BCSL", 10)]
LT = [("Ours (HWA+LA+CSL)", "HWA+LA+CSL", 1), ("Ours (HWA+LA+CSL) Ts=10", "HWA+LA+CSL", 10),
      ("Ours-B (HWA+LA+BCSL)", "HWA+LA+BCSL", 1), ("Ours-B (HWA+LA+BCSL) Ts=10", "HWA+LA+BCSL", 10)]
EXPERIMENT = dict(
    name="E18_bayesian_csl",
    title="Bayesian consensus soft labels (product of label prior and ensemble)",
    hypothesis="BCSL >= CSL at equal budget; BCSL makes T_s = 10 pay off where CSL did not (MNIST long tail).",
    reference="+CSL Ts=10",
    metrics=["best_mean_acc", "final_bal", "final_tail", "fidelity_tail"],
    plan=lambda cfg: [run_spec(lab, "fmnist", s, compose(c, head=dict(ts=ts))) for s in cfg["seeds"] for lab, c, ts in FM_IID]
                     + [run_spec(lab, ds, s, compose(c, head=dict(ts=ts)), IF=0.01, alpha=0.5)
                        for ds in ["mnist"] for s in cfg["seeds"] for lab, c, ts in LT],
)
