"""E19 - Response surface of the synthetic signal: CSL weight beta x budget T_s (full rounds, 2 seeds, FMNIST, K = 10).

Why: a method's failure pattern across its knobs exposes its mechanism.
Prop. 3 model: error = sigma^2/(n_r + lam n_s) + w(lam)^2 b(beta)^2, where the
label bias b depends on beta only and the budget T_s only scales how much of
that bias is paid. Predictions:
  (P1) the best beta is the SAME at T_s = 1 and T_s = 10;
  (P2) the best T_s depends strongly on beta: at beta = 0 (hard labels) more
       budget barely helps (paper Fig. 10), at the best beta it helps most;
  (P3) beta = 1 (pure ensemble) is worse than an interior beta, because the
       ensemble's own error is a second bias source.
If (P1) fails, the stylised model is wrong in a way the surface localises.
Regimes: the paper's IID setting and the 100:1 long tail (with HWA + LA).
"""
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
TS = [1, 10]


def _m(base, beta, ts):
    return compose(base, head=dict(csl=beta, ts=ts))


EXPERIMENT = dict(
    name="E19_csl_response_surface",
    title="Response surface: CSL weight beta x synthetic budget T_s",
    hypothesis="argmax_beta is the same at both budgets; the budget gain grows with the right beta; beta = 1 < interior beta.",
    reference="beta=0 Ts=1",
    seeds=[0, 1],
    metrics=["best_mean_acc", "final_bal", "final_tail"],
    plan=lambda cfg: [run_spec(f"beta={b:g} Ts={t}", "fmnist", s, _m("", b, t)) for s in cfg["seeds"] for t in TS for b in BETAS]
                     + [run_spec(f"beta={b:g} Ts={t}", "fmnist", s, _m("HWA+LA", b, t), IF=0.01, alpha=0.5)
                        for s in cfg["seeds"] for t in TS for b in BETAS],
)
