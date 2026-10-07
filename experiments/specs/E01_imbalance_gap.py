"""E01 - How much does class imbalance cost GeFL-F?  (plan E1.1)

Why: the paper only tests IID balanced clients (footnote 5). This measures
GeFL-F, and the generator-free LG-FedAvg, across four regimes of increasing
imbalance at the paper's data budget: IID; Dirichlet(0.5) with balanced
classes; long tail IF=10 + Dir(0.5); long tail IF=100 + Dir(0.5).
Hypothesis: tail and balanced accuracy fall monotonically with imbalance,
and the generator's advantage over LG-FedAvg shrinks on the tail, because
the tail classes' conditioning collapses (plan section 2.1).
"""
REGIMES = [(1.0, None), (1.0, 0.5), (0.1, 0.5), (0.01, 0.5)]
EXPERIMENT = dict(
    name="E01_imbalance_gap",
    title="GeFL-F accuracy gap by imbalance regime",
    hypothesis="final_bal and final_tail decrease monotonically from IID to IF=100; "
               "the GeFL-F minus LG-FedAvg gap on the tail shrinks as imbalance grows.",
    reference="GeFL-F",
    plan=lambda cfg: [run_spec(lab, ds, s, m, IF=IF, alpha=a)
                      for ds in ["mnist", "fmnist"] for IF, a in REGIMES for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()), ("LG-FedAvg", baseline_method("LG-FedAvg"))]],
)
