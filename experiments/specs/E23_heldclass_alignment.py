"""E23 - Held-class real-synthetic alignment (RSA): shift transfer from held to unheld classes.

View: under the long tail each client is a generalised zero-shot learner - it
learns the classes it lacks ONLY from synthetic features, but is tested on real
ones. The synthetic->real gap is unobservable for unheld classes yet observable
on held ones, and because one shared decoder produces every class, the
generator's error is largely class-invariant. RSA pulls each head's embedding
mean of class-matched synthetic features onto the (stop-gradient) real one for
the classes in the real batch; the shared body carries the correction to the
unheld classes. Bound: R_real(c) <= R_syn(c) + L * W1(phi#q_c, phi#p_c).
Mechanism probe: gap_unheld (synthetic vs REAL test embeddings of classes the
client never saw) must shrink, not only gap_held.
References (same seeds, splits): E20 (Ours, Ours+MC, Ours-PC+MC at T_s = 1).
"""
LABELS = [("Ours+RSA", "HWA+LA+CSL+RSA"), ("Ours-PC+MC+RSA", "PC+LA+CSL+MC+RSA"),
          ("Ours-PC+MC+RSA(0.3)", ("PC+LA+CSL+MC", dict(head=dict(rsa=0.3))))]
SETTINGS = [("mnist", 0.01, 0.5), ("fmnist", 0.01, 0.5), ("fmnist", 1.0, None)]


def _m(c):
    return compose(c[0], **c[1]) if isinstance(c, tuple) else compose(c)


EXPERIMENT = dict(
    name="E23_heldclass_alignment",
    title="Held-class real-synthetic alignment (shift transfer to unheld classes)",
    hypothesis="RSA lowers gap_unheld and raises tail recall over the same method without RSA (E20).",
    reference="Ours+RSA",
    metrics=["final_bal", "final_tail", "best_mean_acc", "gap_held", "gap_unheld"],
    plan=lambda cfg: [run_spec(lab, ds, s, _m(c), IF=IF, alpha=a)
                      for ds, IF, a in SETTINGS for s in cfg["seeds"] for lab, c in LABELS],
)
