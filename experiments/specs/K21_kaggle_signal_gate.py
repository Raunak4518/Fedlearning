"""K21 - Kaggle run: the signal-fraction gate (P1 of the math report, action plan X7) and the class-balanced Stage-I
loss (P2, X8), long tail, new seeds 3-5, paired with K17 (same splits, same seeds; K17 provides every reference arm,
none is re-run here).

Why: Proposition 9 of the draft shows that synthetic features without label information are an exact penalty on the
spread of the logits, and K17 measured the synthetic phase hurting where the generator's signal fraction s is near 0
(CIFAR-10 GeFL-F -0.8, CIFAR-100 GeFL-F -2.2, decay-fixed GeFL-F -1.2). P1 estimates s per class at the server from
existing exact sums (class-balanced LDA on the frozen features, its agreement with the conditioning label of
generated samples, normalised by its recall on real data; core.signal_fraction) and either
  SGATE  draws synthetic labels only from classes with s_c >= 0.1, or
  SCSL   uses consensus soft labels with a per-class weight beta_c = 1 - s_c (pure ensemble label where s_c = 0).
P2 (BALFE) weights the Stage-I loss by N / (C n_y) from the exact global counts, clipped at 20.

Arms (K = 10, IF = 100, Dirichlet 0.5; all readouts, as in K17):
  GeFL-F + AdamW + LA + SGATE / + SCSL      vs K17 "GeFL-F + AdamW + LA"
  Ours + SGATE / + SCSL (T_s as K17's validation selection: 10, except 1 on CIFAR-10)   vs K17 "Ours Ts=<T_s>"
  Ours + BALFE                                   vs K17 "Ours Ts=<T_s>"
Datasets: CIFAR-10 and CIFAR-100 (where s is low) for every P1 arm; MNIST and FashionMNIST (s high) for the
no-harm check with one P1 arm per method; P2 on FashionMNIST, CIFAR-10 and CIFAR-100 (+ MNIST no-harm).
Every run logs sfrac_hat (per-class s_c estimate) next to the referee's per-class fidelity (fidelity, fidelity_kh):
that comparison is X5 (does the label-free estimate track the referee's s?).

PRE-REGISTERED (18 Oct 2026, before any run). Synthetic gain = ER(arm) - ER(K17 real-only heads), same seeds.
  Q1 (P1, decay fix)  on CIFAR-10 and CIFAR-100 the better of SGATE / SCSL moves the decay-fixed GeFL-F's synthetic
     gain from <= 0.1 to >= 0 on both, i.e. its ER is >= K17's "GeFL-F + AdamW + LA" ER (paired, seeds 3-5).
  Q2 (P1, ours)       ours + SCSL changes ER by >= 0 on CIFAR-100 and by 0 +- 0.3 on MNIST / FashionMNIST.
  Q3 (X5)             Spearman(sfrac_hat_c, referee s_c) >= 0.6 pooled over classes, runs and datasets.
  Q4 (P2)             ours + BALFE: ER +1 +- 1 on FashionMNIST, 0 +- 0.5 on MNIST, <= 0 on CIFAR-100.
Selection rule if a gate arm is adopted: SGATE vs SCSL chosen by mean val_er_bal per dataset (validation only).
"""
TS = {"mnist": 10, "fmnist": 10, "svhn": 10, "cifar10": 1, "cifar100": 10}  # K17's validation-selected T_s
OURS = "GATE+MC+KH+LA+CSL"


def _arms(ds):
    a = []
    hard = ds in ("cifar10", "cifar100")
    a.append(("GeFL-F + AdamW + LA + SGATE", "ADAMW3+LA+SGATE", 1))
    if hard:
        a.append(("GeFL-F + AdamW + LA + SCSL", "ADAMW3+LA+SCSL", 1))
    a.append((f"Ours Ts={TS[ds]} + SCSL", OURS + "+SCSL", TS[ds]))
    if hard:
        a.append((f"Ours Ts={TS[ds]} + SGATE", OURS + "+SGATE", TS[ds]))
    a.append((f"Ours Ts={TS[ds]} + BALFE", OURS + "+BALFE", TS[ds]))
    return a


EXPERIMENT = dict(
    name="K21_kaggle_signal_gate",
    title="Signal-fraction gate (P1) and class-balanced Stage-I loss (P2), seeds 3-5, paired with K17",
    hypothesis="Q1-Q4 in the docstring.",
    reference=None,
    seeds=[3, 4, 5],
    split_by_gen=True,
    readouts=["er", "er_own", "fed3r", "fed3r_own", "ncm", "ncm_own", "ccvr", "ccvr_own"],
    metrics=["final_bal", "er_bal", "ccvr_bal", "val_final_bal", "val_er_bal", "final_tail", "final_worst",
             "sfrac_hat_mean", "sgate_kept"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c, head=dict(ts=ts)), IF=0.01, alpha=0.5, K=10)
                      for ds in ["mnist", "fmnist", "cifar10", "cifar100"] for s in cfg["seeds"]
                      for lab, c, ts in _arms(ds)],
)
