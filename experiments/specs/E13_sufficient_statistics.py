"""E13 - FSG: replace the learned feature generator by exactly-aggregated sufficient statistics.

Observation (E03, E09): every method sits ~13 pp below the classifier
oracle, and the gap is classifier knowledge that never crosses clients,
because at K=10 each architecture lives on one client and the only
cross-client channel, the CVAE-F, models rare classes poorly (tail
fidelity 0.15-0.40). Two facts point to a different channel:

 1. Aggregation. Weight averaging of a conditional generator is lossy for
    rare classes (Thm 1: dilution m_r/K; eq. 2.2: collapse). Sufficient
    statistics are not: with per-class counts n_c, sums S_c and one
    second-moment matrix M, the federated estimate of a Gaussian
    h | y ~ N(mu_y, Sigma) equals the centralised one exactly, and every
    quantity is a sum (secure-aggregation and DP friendly).
 2. Estimation. A tail class has ~24 samples federation-wide. The mean
    estimate has error tr(Sigma)/n; a 5.5M-parameter CVAE fitted to those
    samples through 100 rounds of averaging has far higher variance. The
    lower-capacity model is the better estimator for rare classes; the
    pooled covariance borrows strength from all classes (as in LDA).

Cost: one round, D^2 + C*D + C numbers per client (~600k for 3x16x16
features) instead of 5.5M parameters x 100 rounds - about 1000x less
communication. A Gaussian cannot reproduce individual samples.
HYB keeps CVAE-F for classes with >= 100 samples and FSG below that.
"""
LT = [("GeFL-F", ""), ("+HWA+LA", "HWA+LA"),
      ("FSG (seq uniform)", "GAUSS"), ("FSG+LA", "GAUSS+LA"), ("FSG+GF+LA", "GAUSS+GF+LA"),
      ("FSG+PCM+LA", "GAUSS+PCM+LA"), ("HYB+HWA+LA", "HYB+HWA+LA"), ("HYB+HWA+PCM+LA", "HYB+HWA+PCM+LA")]


def _plan(cfg):
    runs = []
    for ds in ["mnist", "fmnist"]:
        for s in cfg["seeds"]:
            for lab, c in LT:
                runs.append(run_spec(lab, ds, s, compose(c), IF=0.01, alpha=0.5))
            runs.append(run_spec("FSG+PCM+LA, DP eps=1", ds, s, compose("GAUSS+PCM+LA", gen=dict(dp_eps=1.0)),
                                 IF=0.01, alpha=0.5))
            for lab, c in [("GeFL-F", ""), ("FSG+LA", "GAUSS+LA"), ("FSG+PCM+LA", "GAUSS+PCM+LA")]:
                runs.append(run_spec(lab, ds, s, compose(c)))  # IID paper setting: do-no-harm
    return runs


EXPERIMENT = dict(
    name="E13_sufficient_statistics",
    title="FSG: exactly aggregated Gaussian feature generator vs CVAE-F",
    hypothesis="FSG has higher tail fidelity than CVAE-F and FSG+PCM+LA > +HWA+LA on final_bal under the long tail; "
               "no loss in IID.",
    reference="GeFL-F",
    metrics=["final_bal", "final_tail", "final_worst", "fidelity_tail", "best_mean_acc", "oracle_bal"],
    plan=_plan,
)
