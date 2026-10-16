"""K18 - Kaggle run: more clients on hard data (SVHN long tail, K = 50 and 100, seeds 0-2).

Replaces K12's IID 50/100-client part, which predates the readout and optimiser-fix results. Two open points:
  (1) K15 found HWA's advantage over the decay fix grows with the number of clients (MNIST); does it on harder data?
  (2) At K = 50 / 100 each architecture has 5 / 10 clients, so the published per-architecture readouts
      (CCVR / Fed3R / FedNCM over an architecture's own clients) are no longer one-client degenerate: the fair
      setting for them.
Arms: GeFL-F; GeFL-F + AdamW + LA (decay fix); GeFL-F + HWA + LA; Ours (gated; on SVHN the gate picks the HWA
class rows, so its generator is shared with the HWA arm). Every run is scored under every readout (test and val).
PRE-REGISTERED (17 Oct 2026): P1 HWA + LA - AdamW + LA is larger at K = 100 than at K = 10 (K16/K15 reference);
P2 Ours + ER beats GeFL-F + ER by >= 2 at both K; P3 own-client CCVR is within 2 points of all-client CCVR at K = 100.
"""
OURS = "GATE+MC+KH+LA+CSL"
ARMS = [("GeFL-F", ""), ("GeFL-F + AdamW + LA", "ADAMW3+LA"), ("GeFL-F + HWA + LA", "HWA+LA"), ("Ours (gated)", OURS)]
EXPERIMENT = dict(
    name="K18_kaggle_more_clients_svhn",
    title="SVHN long tail with 50 and 100 clients: decay fix vs HWA vs ours, under every readout",
    hypothesis="P1-P3 in the docstring.",
    reference="GeFL-F",
    split_by_gen=True,
    save_gen=True,
    readouts=["er", "er_own", "fed3r", "fed3r_own", "ncm", "ncm_own", "ccvr", "ccvr_own"],
    metrics=["final_bal", "er_bal", "ccvr_bal", "ccvr_own_bal", "bbc_bal", "final_tail", "final_worst",
             "cond_norm_tail_over_head_end", "val_er_bal"],
    plan=lambda cfg: [run_spec(lab, "svhn", s, compose(c), IF=0.01, alpha=0.5, K=K)
                      for K in [50, 100] for s in cfg["seeds"] for lab, c in ARMS],
)
