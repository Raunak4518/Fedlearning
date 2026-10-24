"""Every classification metric of action plan section 14, computed from one model's test logits.

Definitions follow experiments/core.py where it defined them (recall, precision, macro-F1, G-mean with recalls
floored at 0.01, ECE with 15 equal-width bins, NLL with probabilities floored at 1e-6, Brier); the others use the
standard definitions (scikit-learn for AUC-ROC and average precision; tests/test_metrics.py checks the rest against
scikit-learn). Multi-architecture summaries average over architectures (each architecture counts once), as core.py.
"""
import numpy as np
import torch

from settings import class_groups

SCALARS = ("acc", "bal", "precision", "f1", "mcc", "kappa", "gmean", "worst", "head", "medium", "tail", "rec_std",
           "auc_roc", "auc_pr", "top5", "ece", "cw_ece", "nll", "brier", "local")
# Per-class versions of every metric that has one (the class-averaged metrics above are their means, except NLL and
# Brier, which average over images): recall, precision, F1, average precision (AUC-PR), AUC-ROC and MCC of class c
# against the rest, calibration error of p(c|x), NLL and Brier score on the images of class c, top-5 recall, and the
# share of all predictions that go to class c (pred_share_c against the true share shows the bias towards frequent
# classes).
PER_CLASS = ("recall_c", "precision_c", "f1_c", "ap_c", "auc_c", "mcc_c", "ece_c", "nll_c", "brier_c", "top5_c",
             "pred_share_c")


def _ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        sel = (conf > edges[i]) & (conf <= edges[i + 1])
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def _classwise_ece(prob, y, C, bins=15):
    """Class-wise ECE (Kull et al., 2019): per class, calibration of p(c|x) against 1[y = c], averaged over classes."""
    return float(np.mean(_ece_per_class(prob, y, C, bins)))


def _ece_per_class(prob, y, C, bins=15):
    return np.array([_ece(prob[:, c], (y == c).astype(float), bins) for c in range(C)])


def class_metrics(logits, y, C, prior=None, keep_probs=False):
    """All metrics of one model. logits (N, C) tensor, y (N,) tensor. prior: the model's own clients' label
    distribution (accuracy re-weighted to it = what its own clients would see)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    prob_t = logits.float().softmax(1)
    pred_t = prob_t.argmax(1)
    prob, pred, yy = prob_t.cpu().numpy().astype(np.float64), pred_t.cpu().numpy(), y.cpu().numpy()
    N = len(yy)
    cm = np.zeros((C, C), dtype=np.int64)
    np.add.at(cm, (yy, pred), 1)
    tp = np.diag(cm).astype(float)
    ny = cm.sum(1).astype(float)
    npred = cm.sum(0).astype(float)
    rec = tp / np.maximum(ny, 1)
    prec = tp / np.maximum(npred, 1)
    f1_c = 2 * tp / np.maximum(ny + npred, 1)
    acc = float(tp.sum() / N)
    # MCC and Cohen's kappa (multiclass, from the confusion matrix)
    t, p, c_, s = ny, npred, tp.sum(), float(N)
    mcc_den = np.sqrt((s ** 2 - (p ** 2).sum()) * (s ** 2 - (t ** 2).sum()))
    mcc = float((c_ * s - (t * p).sum()) / mcc_den) if mcc_den > 0 else 0.0
    pe = float((t * p).sum() / s ** 2)
    kappa = float((acc - pe) / (1 - pe)) if pe < 1 else 0.0
    present = ny > 0
    onehot = np.eye(C)[yy]
    try:
        auc_roc = float(roc_auc_score(onehot[:, present], prob[:, present], average="macro"))
    except ValueError:
        auc_roc = float("nan")
    ap_c = np.array([average_precision_score(onehot[:, c], prob[:, c]) if present[c] else np.nan for c in range(C)])
    auc_c = np.array([roc_auc_score(onehot[:, c], prob[:, c]) if present[c] and ny[c] < N else np.nan
                      for c in range(C)])
    k5 = min(5, C)
    in5 = (np.argsort(-prob, 1)[:, :k5] == yy[:, None]).any(1)
    top5 = float(in5.mean())
    # per class: MCC of "class c" against the rest; NLL, Brier and top-5 recall on the images of class c
    fp, fn = npred - tp, ny - tp
    tn = N - tp - fp - fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc_c = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)
    nll_i = -np.log(np.clip(prob[np.arange(N), yy], 1e-6, None))
    brier_i = ((prob - onehot) ** 2).sum(1)
    by_class = [yy == c for c in range(C)]
    nll_c = np.array([nll_i[s].mean() if s.any() else np.nan for s in by_class])
    brier_c = np.array([brier_i[s].mean() if s.any() else np.nan for s in by_class])
    top5_c = np.array([in5[s].mean() if s.any() else np.nan for s in by_class])
    conf = prob.max(1)
    hd, md, tl = class_groups(C)
    out = dict(
        acc=acc, bal=float(rec[present].mean()), precision=float(prec.mean()), f1=float(f1_c.mean()), mcc=mcc,
        kappa=kappa, gmean=float(np.exp(np.log(np.clip(rec[present], 0.01, None)).mean())), worst=float(rec[present].min()),
        head=float(rec[hd].mean()), medium=float(rec[md].mean()), tail=float(rec[tl].mean()), rec_std=float(rec.std()),
        auc_roc=auc_roc, auc_pr=float(np.nanmean(ap_c)), top5=top5,
        ece=_ece(conf, (pred == yy).astype(float)), cw_ece=_classwise_ece(prob, yy, C),
        nll=float(nll_i.mean()), brier=float(brier_i.mean()),
        recall_c=rec.tolist(), precision_c=prec.tolist(), f1_c=f1_c.tolist(), ap_c=ap_c.tolist(), auc_c=auc_c.tolist(),
        mcc_c=mcc_c.tolist(), ece_c=_ece_per_class(prob, yy, C).tolist(), nll_c=nll_c.tolist(),
        brier_c=brier_c.tolist(), top5_c=top5_c.tolist(), pred_share_c=(npred / N).tolist(), confusion=cm.tolist())
    if prior is not None:
        out["local"] = float((np.asarray(prior, dtype=float) * rec).sum())
    if keep_probs:
        out["probs"] = prob.astype(np.float32)
    return out


def log_view(m, digits=4):
    """What a per-round evaluation logs: every scalar and every per-class metric (rounded; no confusion matrix or
    probabilities, which only the final evaluation keeps)."""
    out = {k: (round(float(m[k]), digits) if m[k] is not None and not np.isnan(m[k]) else None)
           for k in SCALARS + ("worst_arch", "ens_acc", "ens_bal") if k in m}
    for k in PER_CLASS:
        if k in m:
            out[k] = [None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), digits)
                      for v in m[k]]
    return out


def summarize(per_arch, ens=None):
    """Mean over architectures of every scalar; class-wise vectors averaged; balanced accuracy of the mean recall
    (as core.summarize_eval); worst_arch = balanced accuracy of the weakest architecture."""
    vals = list(per_arch.values())
    out = {}
    for k in SCALARS:
        xs = [v[k] for v in vals if k in v and v[k] is not None and not (isinstance(v[k], float) and np.isnan(v[k]))]
        if xs:
            out[k] = float(np.mean(xs))
    rec = np.mean([v["recall_c"] for v in vals], 0)
    C = len(rec)
    hd, md, tl = class_groups(C)
    out.update(bal=float(rec.mean()), head=float(rec[hd].mean()), medium=float(rec[md].mean()),
               tail=float(rec[tl].mean()), worst=float(rec.min()), rec_std=float(rec.std()),
               worst_arch=float(min(np.mean(v["recall_c"]) for v in vals)), recall_c=rec.tolist())
    for k in PER_CLASS[1:]:
        if all(k in v for v in vals):
            out[k] = np.nanmean([v[k] for v in vals], 0).tolist()
    out["confusion"] = np.sum([v["confusion"] for v in vals], 0).tolist()
    if ens is not None:
        out["ens_acc"], out["ens_bal"] = ens["acc"], ens["bal"]
    return out


@torch.no_grad()
def evaluate_heads(heads, groups, X, y, C, priors=None, keep_probs=False, embed=False):
    """Per-architecture metrics of the heads on features X (chunks of 2000), plus the averaged-softmax ensemble."""
    per, prob_sum = {}, None
    emb = {}
    for g in groups:
        h = heads[g]
        was = h.training
        h.eval()
        logits = torch.cat([h(X[s:s + 2000]) for s in range(0, len(y), 2000)])
        if embed:
            emb[g] = torch.cat([h.embed(X[s:s + 2000]) for s in range(0, len(y), 2000)]).cpu().numpy()
        h.train(was)
        prob_sum = logits.softmax(1) if prob_sum is None else prob_sum + logits.softmax(1)
        per[g] = class_metrics(logits, y, C, prior=None if priors is None else priors.get(g), keep_probs=keep_probs)
    ens = None
    if len(groups) > 1:
        e = class_metrics(prob_sum.log(), y, C)
        ens = dict(acc=e["acc"], bal=e["bal"])
    return per, summarize(per, ens), emb


def arch_priors(counts, arch, groups):
    out = {}
    for g in groups:
        own = counts[[k for k in range(len(arch)) if arch[k] == g]].sum(0).astype(np.float64)
        out[g] = own / max(own.sum(), 1.0)
    return out
