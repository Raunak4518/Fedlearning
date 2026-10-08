"""K02 - Kaggle run: heterogeneity regimes, privacy cost and ablations (MNIST, FMNIST; full schedule).

Made for Kaggle (GPU T4 x2, Internet on); both GPUs are used. It is
F03 + F07 + F08 in one file, around the final method chosen by F10.
Runs shared between the parts (e.g. Ours under the long tail) are run once.

Part A - regimes (3 seeds): label skew without a global tail (IF=1, Dir 0.5)
          and a mild tail (IF=10, Dir 0.5): GeFL-F, +LA, Ours.
Part B - privacy (3 seeds, long tail): Ours with Laplace-noised class counts
          (epsilon-DP per client histogram, eps in {10, 1, 0.1}); feature-space
          memorisation (MND, > 1 = memorisation) for GeFL-F and Ours.
Part C - ablations (2 seeds, long tail): each component removed, HWA weights,
          LA temperature, CSL weight.

Typical use (finished runs are skipped when runs.jsonl is present, so a
cut-off session resumes):
    python K02.py --out_dir /kaggle/working/results
    python K02.py --datasets mnist --out_dir /kaggle/working/results
"""
LT = dict(IF=0.01, alpha=0.5)
DSETS = ["mnist", "fmnist"]


def _regimes(cfg):
    return [run_spec(lab, ds, s, m, IF=IF, alpha=0.5)
            for ds in DSETS for IF in [1.0, 0.1] for s in cfg["seeds"]
            for lab, m in [("GeFL-F", compose()), ("+LA", compose("LA")), (FINAL_LABEL, final_method())]]


def _privacy(cfg):
    return [run_spec(lab, ds, s, m, **LT)
            for ds in DSETS for s in cfg["seeds"]
            for lab, m in [("GeFL-F", compose()), (FINAL_LABEL, final_method()),
                           (FINAL_LABEL + ", DP eps=10", final_method(gen=dict(dp_eps=10.0))),
                           (FINAL_LABEL + ", DP eps=1", final_method(gen=dict(dp_eps=1.0))),
                           (FINAL_LABEL + ", DP eps=0.1", final_method(gen=dict(dp_eps=0.1)))]]


def _ablations(cfg):
    has_csl = "CSL" in FINAL_PARTS.split("+")
    v = [("no HWA (flat averaging)", final_method(gen=dict(agg="flat"))),
         ("no LA (plain CE)", final_method(head=dict(loss="ce"))),
         ("no CSL" if has_csl else "with CSL (beta=0.5)", final_method(head=dict(csl=0.0 if has_csl else 0.5))),
         ("HWA weights = counts n", final_method(gen=dict(weight="linear"))),
         ("HWA weights = uniform over holders", final_method(gen=dict(beta=0.0))),
         ("LA tau=1.5", final_method(head=dict(tau=1.5))),
         ("LA tau=2.0", final_method(head=dict(tau=2.0))),
         ("CSL beta=0.25", final_method(head=dict(csl=0.25))),
         ("CSL beta=0.75", final_method(head=dict(csl=0.75)))]
    return [run_spec(lab, ds, s, m, **LT) for ds in DSETS for s in cfg["seeds"][:2] for lab, m in v]


def _plan(cfg):
    seen, out = set(), []
    for r in _privacy(cfg) + _regimes(cfg) + _ablations(cfg):
        if run_id(r) not in seen:
            seen.add(run_id(r))
            out.append(r)
    return out


EXPERIMENT = dict(
    name="K02_kaggle_regimes_privacy_ablations",
    title="Regimes, DP class counts and memorisation, ablations (MNIST, FMNIST)",
    hypothesis="Ours > GeFL-F in every regime; most of the gain survives eps=1; MND < 1; removing HWA or LA costs several points.",
    reference="GeFL-F",
    mnd=True,
    metrics=["final_bal", "final_tail", "best_mean_acc", "feature_mnd", "fidelity_tail"],
    plan=_plan,
)
