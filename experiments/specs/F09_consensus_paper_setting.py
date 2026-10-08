"""F09 - Beat GeFL-F in its OWN setting: consensus soft labels (CSL) for synthetic features.

Setting: exactly the paper's Figure 4 protocol (IID equal clients, Table
XIV/XV, K = 10/50/100). Targets (GeFL-F, CVAE-F, Figure 4):
  MNIST 95.47 / 95.04 / 94.63   FMNIST 83.14 / 82.21 / 81.65
  SVHN  76.26 / 73.64 / 76.00   CIFAR-10 55.86 / 53.19 / 51.46

Diagnosis. At K=10 each architecture is trained on one client's data (600
MNIST images); the rest of the federation reaches it only through the
generator's synthetic phase, which uses hard conditioning labels although a
real-data classifier recognises only ~40-80% of CVAE-F samples as their
label. CSL: each round the server draws one shared batch of synthetic
features (the same budget GeFL-F uses), every architecture's header labels
it, and the averaged probabilities replace the hard labels:

    target = (1 - beta) * onehot(y) + beta * mean_g softmax(header_g(x))

Clients exchange only these probabilities (P x C numbers per round, no
weights, no architecture; summable under secure aggregation). The averaged
posterior of headers that jointly saw all the data is a lower-variance
estimate of p(y | h) than the generator's conditioning label (distillation
as variance reduction), and it is the only cheap channel that moves
real-data knowledge across architectures that cannot be weight-averaged.
Diagnostic: final_ens_acc = accuracy of the averaged-softmax ensemble of
the ten headers - the headroom CSL can transfer.
"""
EXPERIMENT = dict(
    name="F09_consensus_paper_setting",
    title="Paper setting (IID, Figure 4 protocol): consensus soft labels vs GeFL-F",
    hypothesis="+CSL > GeFL-F on best_mean_acc (the paper's metric), beating the Figure 4 value.",
    reference="GeFL-F",
    metrics=["best_mean_acc", "final_acc", "final_ens_acc", "final_bal"],
    plan=lambda cfg: [run_spec(lab, ds, s, m, K=K)
                      for ds in ["mnist", "fmnist", "svhn", "cifar10"] for K in [10, 50, 100] for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()), ("+CSL (beta=0.5)", compose("CSL")),
                                     ("+CSL (beta=1)", compose("CSL", head=dict(csl=1.0))),
                                     ("+CSLM (interleaved)", compose("CSLM"))]],
)
