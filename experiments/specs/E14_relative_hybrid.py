"""E14 - Relative-rarity hybrid: FSG for rare classes, CVAE-F for the rest.

E13 showed a bias-variance split. Under a long tail, FSG (exact Gaussian
from aggregated sums) beats every CVAE-F method, by up to +5 pp. In the
paper's IID setting it loses about 1 pp, because a Gaussian is too crude
for classes with plenty of data. A fixed threshold (n < 100) was too low
on FMNIST. The rule tested here is scale-free:

    class c uses FSG  iff  n_c < rho * N / C      (N/C = average class size)

In IID data every n_c = N/C, so no class switches and the method reduces
exactly to GeFL-F + HWA + LA: do-no-harm holds by construction, not by
luck. Under a long tail the classes below rho times the average get FSG.
rho in {0.5, 1}. Settings as in E12, 3 seeds.
"""
LABELS = [("GeFL-F", ""), ("+HWA+LA", "HWA+LA"), ("FSG+LA", "GAUSS+LA"),
          ("RHYB(0.5)+HWA+LA", "RHYB+HWA+LA"),
          ("RHYB(1.0)+HWA+LA", "HWA+LA", dict(gen=dict(type="hybrid", n_min_rel=1.0)))]


def _m(spec):
    return compose(spec[1], **(spec[2] if len(spec) > 2 else {}))


def _plan(cfg):
    runs = []
    for ds in ["mnist", "fmnist"]:
        for s in cfg["seeds"]:
            for IF, a in [(0.01, 0.5), (1.0, 0.5), (1.0, None)]:
                for spec in LABELS:
                    runs.append(run_spec(spec[0], ds, s, _m(spec), IF=IF, alpha=a))
    for s in cfg["seeds"]:
        for spec in LABELS:
            runs.append(run_spec(spec[0], "mnist", s, _m(spec), IF=0.01, alpha=0.5, K=50))
    return runs


EXPERIMENT = dict(
    name="E14_relative_hybrid",
    title="Relative-rarity hybrid generator (FSG for rare classes, CVAE-F otherwise)",
    hypothesis="RHYB+HWA+LA >= FSG+LA under the long tail and == GeFL-F+HWA+LA in IID (no harm).",
    reference="GeFL-F",
    metrics=["final_bal", "final_tail", "final_worst", "best_mean_acc", "n_gauss_classes", "fidelity_tail"],
    plan=_plan,
)
