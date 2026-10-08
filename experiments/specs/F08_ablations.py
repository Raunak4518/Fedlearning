"""F08 - Ablations of the final method (full rounds, 2 seeds).

One change at a time around Ours, on MNIST and FMNIST under the long tail:
  - each component removed (HWA -> flat averaging, LA -> plain CE, CSL toggled);
  - HWA's holder weights: effective number E(n) (default, beta = 0.999),
    raw counts n, and uniform over holders (beta = 0, E(n) = 1 for n > 0);
  - LA temperature tau in {1.5, 2};
  - CSL weight beta in {0.25, 0.75} (beta = 0.5 in Ours / the +CSL variant).
"""


def _variants():
    has_csl = "CSL" in FINAL_PARTS.split("+")
    v = [(FINAL_LABEL, final_method()),
         ("no HWA (flat averaging)", final_method(gen=dict(agg="flat"))),
         ("no LA (plain CE)", final_method(head=dict(loss="ce"))),
         ("no CSL" if has_csl else "with CSL (beta=0.5)", final_method(head=dict(csl=0.0 if has_csl else 0.5))),
         ("HWA weights = counts n", final_method(gen=dict(weight="linear"))),
         ("HWA weights = uniform over holders", final_method(gen=dict(beta=0.0))),
         ("LA tau=1.5", final_method(head=dict(tau=1.5))),
         ("LA tau=2.0", final_method(head=dict(tau=2.0))),
         ("CSL beta=0.25", final_method(head=dict(csl=0.25))),
         ("CSL beta=0.75", final_method(head=dict(csl=0.75)))]
    return v


EXPERIMENT = dict(
    name="F08_ablations",
    title="Ablations of the final method",
    hypothesis="removing HWA or LA costs several points; E(n) >= n >= uniform weights; tau = 1 and beta = 0.5 near-optimal.",
    reference=FINAL_LABEL,
    seeds=[0, 1],
    metrics=["final_bal", "final_tail", "final_worst", "fidelity_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, m, IF=0.01, alpha=0.5)
                      for ds in ["mnist", "fmnist"] for s in cfg["seeds"] for lab, m in _variants()],
)
