"""F08 - Ablations of the final method (full rounds, 2 seeds).

Each component removed or varied one at a time around Ours on MNIST and
FMNIST under the long tail: the rarity threshold rho (which classes get
FSG), LA's temperature tau, HWA removed, LA removed, and FSG for every
class (rho = infinity) vs none (rho = 0, i.e. +HWA+LA).
"""


def _variants():
    return [(FINAL_LABEL, final_method()),
            ("rho=0.25", final_method(gen=dict(n_min_rel=0.25))),
            ("rho=0.5", final_method(gen=dict(n_min_rel=0.5))),
            ("rho=2.0", final_method(gen=dict(n_min_rel=2.0))),
            ("tau=1.5", final_method(head=dict(tau=1.5))),
            ("tau=2.0", final_method(head=dict(tau=2.0))),
            ("no HWA", final_method(gen=dict(agg="flat"))),
            ("no LA", final_method(head=dict(loss="ce")))]


EXPERIMENT = dict(
    name="F08_ablations",
    title="Ablations of the final method",
    hypothesis="every component matters (removing it costs accuracy); rho near 0.5-1 and tau near 1 are best.",
    reference=FINAL_LABEL,
    seeds=[0, 1],
    plan=lambda cfg: [run_spec(lab, ds, s, m, IF=0.01, alpha=0.5)
                      for ds in ["mnist", "fmnist"] for s in cfg["seeds"] for lab, m in _variants()],
)
