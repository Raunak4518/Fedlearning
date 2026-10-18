"""K24 - Kaggle run: does rare-class collapse need coupled decay? (action plan X12 = D-5), DCGAN-F under the long
tail, MNIST and FashionMNIST, K = 10 and 50, new seeds 3-5.

The paper's DCGAN-F (like DDPM-F) is trained without weight decay. Lemma 4 / §3.6 then predict no collapse — a
non-holder's gradient on a class's parameters is exactly zero — only the m_c / K dilution of flat averaging, which
HWA removes. DDPM-F is not run (about 1 GPU-hour per run; F11, K04).
Arms: GeFL-F (DCGAN-F) + LA, and the same with HWA on its class embeddings (g_label / d_label rows).

PRE-REGISTERED (18 Oct 2026, before any run):
  G1  tail / head conditioning-row norm ratio >= 0.7 for flat averaging at K = 10 (no collapse without decay).
  G2  HWA - flat (final balanced accuracy and ER) is within +-1 point at K = 10 and larger at K = 50 than at K = 10 on
      both datasets (dilution grows with K, §3.6).
"""
ARMS = [("GeFL-F (DCGAN-F) + LA", "DCGAN+LA"), ("GeFL-F (DCGAN-F) + HWA + LA", "DCGAN+HWA+LA")]
EXPERIMENT = dict(
    name="K24_kaggle_dcgan_longtail",
    title="DCGAN-F under the long tail: flat averaging vs HWA without weight decay (MNIST, FMNIST; K = 10, 50)",
    hypothesis="G1-G2 in the docstring.",
    reference="GeFL-F (DCGAN-F) + LA",
    seeds=[3, 4, 5],
    split_by_gen=True,
    readouts=["er", "ccvr"],
    metrics=["final_bal", "er_bal", "final_tail", "cond_norm_tail_over_head_end", "fidelity_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c), IF=0.01, alpha=0.5, K=K)
                      for ds in ["mnist", "fmnist"] for K in [10, 50] for s in cfg["seeds"] for lab, c in ARMS],
)
