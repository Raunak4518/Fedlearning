"""E24 - KME-Gen: a federated generator that is never trained federatedly (full head rounds).

Every failure measured so far is a failure of training the generator by FedAvg
(row collapse, m_c/K dilution, few local steps for rare classes, K-dependence).
KME-Gen removes federated generator training altogether: each client uploads,
once, its per-class sums of random Fourier features phi(h) (secure aggregation;
bounded features, so Gaussian-mechanism DP is cheap). The server obtains the
EXACT pooled kernel mean embedding mu_c of every class and trains the
prototype-anchored generator by MMD: min sum_c ||E phi(G(z,c)) - mu_c||^2.
The objective is identical to the centralised one for any partition (partition
invariance), the clients train no generator, and one ~150 KB upload replaces
100 rounds of generator FedAvg.
Arms: KME with LA + CSL (+MC / +KH); and KH (kernel herding of samples toward the exact
federated class embeddings) on the best federated generator, PC + MC. References (same seeds/splits): E20.
Decision rule: full test only if KME >= the best E20 arm on the long tail.
"""
LABELS = [("Ours-KME (KME+LA+CSL)", "KME+LA+CSL"), ("Ours-KME+MC", "KME+LA+CSL+MC"), ("Ours-KME+KH", "KME+LA+CSL+KH"),
          ("Ours-PC+MC+KH", "PC+LA+CSL+MC+KH")]
SETTINGS = [("mnist", 0.01, 0.5), ("fmnist", 0.01, 0.5), ("fmnist", 1.0, None)]
EXPERIMENT = dict(
    name="E24_kme_generator",
    title="KME-Gen: server-trained generator from exact federated kernel mean embeddings",
    hypothesis="KME-Gen >= PC+MC (E20) under the long tail, with no client-side generator training.",
    reference="Ours-KME (KME+LA+CSL)",
    metrics=["final_bal", "final_tail", "best_mean_acc", "fidelity_tail", "spread_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c), IF=IF, alpha=a)
                      for ds, IF, a in SETTINGS for s in cfg["seeds"] for lab, c in LABELS],
)
