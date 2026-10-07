"""E12 - Confirmation before any full run: is the candidate method reliably better?

E03's one-seed quick pass picked HWA + LA (with or without LCD) as the best
method, and showed PCM hurting while the generator's tail fidelity is low.
This repeats the decisive comparisons with 3 seeds (paired t-tests) across
the federated settings the method must survive:

  long tail      IF=100 + Dirichlet(0.5)    the target regime
  label skew     IF=1   + Dirichlet(0.5)    non-IID without a global tail
  IID            the paper's own setting    do-no-harm (Proposition 2)
  more clients   K=50, IF=100 + Dir(0.5)    sparser holders (Theorem 1)

Candidates:  +HWA+LA            needs class histograms (secure-aggregation form, README)
             +HWA+LCD+LA        the same plus lazy decay
             +LCD+LA            privacy-maximal: no histograms leave any client
Baselines:   GeFL-F, +LA (classifier-side only), LG-FedAvg+LA (no generator).
Rule for proceeding to full runs: a candidate beats GeFL-F, +LA and
LG-FedAvg+LA on final_bal in the long-tail setting on both datasets with
p < 0.05, and is within noise or better in the IID setting.
"""
CANDIDATES = [("GeFL-F", ""), ("+LA", "LA"), ("+HWA", "HWA"), ("+HWA+LA", "HWA+LA"),
              ("+HWA+LCD+LA", "HWA+LCD+LA"), ("+LCD+LA", "LCD+LA")]


def _plan(cfg):
    runs = []
    for ds in ["mnist", "fmnist"]:
        for s in cfg["seeds"]:
            for IF, a in [(0.01, 0.5), (1.0, 0.5), (1.0, None)]:
                for lab, c in CANDIDATES:
                    runs.append(run_spec(lab, ds, s, compose(c), IF=IF, alpha=a))
                runs.append(run_spec("LG-FedAvg+LA", ds, s, baseline_method("LG-FedAvg", la=True), IF=IF, alpha=a))
            runs.append(run_spec("+HWA+LA (tau=1.5)", ds, s, compose("HWA+LA", head=dict(tau=1.5)), IF=0.01, alpha=0.5))
    for s in cfg["seeds"]:
        for lab, c in [("GeFL-F", ""), ("+LA", "LA"), ("+HWA+LA", "HWA+LA"), ("+HWA+LCD+LA", "HWA+LCD+LA"),
                       ("+LCD+LA", "LCD+LA")]:
            runs.append(run_spec(lab, "mnist", s, compose(c), IF=0.01, alpha=0.5, K=50))
    return runs


EXPERIMENT = dict(
    name="E12_confirmation",
    title="Multi-seed confirmation of the candidate method across federated settings",
    hypothesis="+HWA+LA (and +HWA+LCD+LA) > +LA > GeFL-F and > LG-FedAvg+LA under the long tail, p<0.05; "
               "no loss in IID.",
    reference="GeFL-F",
    plan=_plan,
)
