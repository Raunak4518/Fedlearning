"""F04 - More clients (full rounds, 3 seeds): K = 50 and 100, the paper's Figure 4 axis.

Total data fixed (footnote 5): 120 / 60 images per client. Several clients
share each architecture, so header averaging is active. FSG's sums do not
degrade with K (exact aggregation), whereas weight averaging of CVAE-F
dilutes rare classes further (Theorem 1).
"""
EXPERIMENT = dict(
    name="F04_clients",
    title="Client scaling K = 50, 100 under the long tail",
    hypothesis="Ours > +LA > GeFL-F at K=50 and K=100.",
    reference="GeFL-F",
    plan=lambda cfg: [run_spec(lab, "mnist", s, m, IF=0.01, alpha=0.5, K=K)
                      for K in [50, 100] for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()), ("+LA", compose("LA")),
                                     ("FSG+LA", compose("GAUSS+LA")), ("+HWA+LA", compose("HWA+LA")),
                                     ("+HWA+LA+CSL", compose("HWA+LA+CSL"))]],
)
