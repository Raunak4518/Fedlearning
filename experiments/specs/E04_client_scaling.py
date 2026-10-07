"""E04 - Does the advantage grow with the number of clients?  (plan E3.1)

Theorem 1: flat averaging scales the class-r conditioning update by m_r/K
(holders over clients). With the data budget fixed (paper footnote 5), more
clients means fewer holders per class, so dilution and collapse worsen and
HWA+LCD's gain should RISE with K - a falsifiable prediction on the same
axis as the paper's Figure 4 (K = 10, 50, 100). At K=50/100 several clients
share each architecture, so header averaging is active too.
"""
EXPERIMENT = dict(
    name="E04_client_scaling",
    title="Client scaling K = 10, 50, 100 under a long tail",
    hypothesis="(+HWA+LCD) - GeFL-F and Ours - GeFL-F both increase with K.",
    reference="GeFL-F",
    plan=lambda cfg: [run_spec(lab, "mnist", s, compose(c), IF=0.01, alpha=0.5, K=K)
                      for K in [10, 50, 100] for s in cfg["seeds"]
                      for lab, c in [("GeFL-F", ""), ("+HWA+LCD", "HWA+LCD"), ("+LA", "LA"),
                                     ("Ours (HWA+LCD+PCM+LA)", OURS),
                                     ("Ours-private (LCD+PCM+LA)", OURS_PRIVATE)]],
)
