"""t-SNE plots, latent scatter plots and confusion matrices from saved models (no training).

    python experiments/viz_latents.py --models <models/K17_kaggle_fair_retest> --dataset mnist --seed 3 \
        [--out docs/paper_assets/figures] [--n 150]

Needs the models a run saved with --save_gen (feature extractor, generator, ten heads per run). Writes:
  V1_real_features_<ds>.png       t-SNE of real test features from the frozen feature extractor, coloured by class
  V2_real_vs_fake_<ds>.png        t-SNE of real training features and generated features, one panel per method
  V3_generator_latent_<ds>.png    the generator's latent code of real training features (encoder means), t-SNE
  V4_classifier_embedding_<ds>.png  t-SNE of one architecture's body embedding of real test data, per method
  V5_confusion_<ds>.png           row-normalised confusion matrices (recall on the diagonal), heads as trained and with ER
  V_metrics_<ds>.md               per-class precision, recall and F1 for the same heads
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402

METHODS = [("GeFL-F", "GeFL-F"), ("GeFL-F + AdamW + LA", "GeFL-F, decoupled decay + LA"), ("Ours Ts=10", "Ours")]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 150})


def load_run(mdir, label, ds, seed):
    run = dict(label=label, dataset=ds, IF=0.01, alpha=0.5, K=10, seed=seed)
    path = os.path.join(mdir, "heads", core.safe_name(core.run_id(run)) + ".pt")
    if not os.path.exists(path):
        return None
    M = torch.load(path, map_location=core.DEV)
    return run, M


def setup(mdir, run, M, data_root):
    ds, seed = run["dataset"], run["seed"]
    spec = core.DATASETS[ds]
    core.set_num_classes(ds)
    data = core.load_dataset(ds, data_root)
    part = core.make_partition(data["ytr"], 10, spec["frac"], run["IF"], run["alpha"], seed)
    data, part = core.localize(data, part)
    ctx = core.Ctx(core.default_cfg(), data, ds, part, 10, seed)
    fe = core.PaperFE(spec["in_ch"], spec["fe"]).to(core.DEV)
    fe.load_state_dict(torch.load(os.path.join(mdir, M["fe_file"]), map_location=core.DEV))
    feats = [core.compute_features(fe, data["xtr"][p]) for p in part["parts"]]
    ys = [data["ytr"][p] for p in part["parts"]]
    feats_te = core.compute_features(fe, data["xte"])
    heads = core.build_headers(ctx, feats[0].shape[1])
    for h, sd in zip(heads, M["heads"]):
        h.load_state_dict(sd)
        h.eval()
    G = None
    m = M["method"]
    if M.get("gen_file") and m["gen"] is not None:
        proto = core.class_prototypes(feats, ys) if m["gen"]["type"] == "pcvae" else None
        G = core.make_gen(ctx, m["gen"], feats[0].shape[1], feats[0].shape[1:], proto)
        G.load_state_dict(torch.load(os.path.join(mdir, M["gen_file"]), map_location=core.DEV))
        G.eval()
        s1 = dict(fe=fe, feats=feats, ys=ys, feats_te=feats_te, ref=None)
        core.set_seed(seed + 2000)
        Gs = core.wrap_sampler(ctx, m, G, s1, {})  # what the heads trained on (MC / KH applied where the method uses them)
    else:
        Gs = None
    return dict(ctx=ctx, feats=feats, ys=ys, feats_te=feats_te, yte=data["yte"], heads=heads, G=G, Gs=Gs, method=m)


def subsample(X, y, n, C, rng):
    idx = np.concatenate([rng.choice(np.where(y == c)[0], min(n, int((y == c).sum())), replace=False) for c in range(C)])
    return X[idx], y[idx]


def tsne(X, seed=0):
    return TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(X)


def scatter(ax, Z, y, C, marker="o", s=5, alpha=0.7, label_classes=False):
    cmap = plt.get_cmap("tab10" if C <= 10 else "tab20")
    for c in range(C):
        sel = y == c
        ax.scatter(Z[sel, 0], Z[sel, 1], s=s, marker=marker, color=cmap(c % cmap.N), alpha=alpha, lw=0,
                   label=str(c) if label_classes else None)
    ax.set_xticks([])
    ax.set_yticks([])


@torch.no_grad()
def generator_latent(G, X, y):
    """Encoder means of real features: where the generator places real data in its latent space."""
    if hasattr(G, "encode"):
        mu, _ = G.encode(X, y)
        return mu
    ymap = y.float().view(-1, 1, 1, 1).expand(-1, 1, X.size(2), X.size(3))
    return G.mu(G.enc(torch.cat([X, ymap], 1)).flatten(1))


@torch.no_grad()
def per_class_prf(logits, y, C):
    pred = logits.argmax(1)
    cm = torch.zeros(C, C)
    for t, p in zip(y.cpu(), pred.cpu()):
        cm[t, p] += 1
    tp = cm.diag()
    rec = tp / cm.sum(1).clamp(min=1)
    prec = tp / cm.sum(0).clamp(min=1)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
    return cm, prec.numpy(), rec.numpy(), f1.numpy()


@torch.no_grad()
def er_logits(S, h, X):
    W, b = core.lda_readout(h, S["feats"], S["ys"], list(range(S["ctx"].K)))
    e = h.embed(X).double()
    return (e @ W.T.double() + b.double()).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--data_root", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "paper_assets", "figures"))
    ap.add_argument("--n", type=int, default=150, help="points per class in each t-SNE")
    ap.add_argument("--arch", type=int, default=0, help="architecture whose body embedding is plotted")
    ap.add_argument("--labels", nargs="+", default=None, help="run labels to plot (default: GeFL-F, decay fix, Ours Ts=10)")
    a = ap.parse_args()
    methods = [(lab, lab) for lab in a.labels] if a.labels else METHODS
    os.makedirs(a.out, exist_ok=True)
    ds, rng = a.dataset, np.random.RandomState(0)
    runs = [(name, load_run(a.models, lab, ds, a.seed)) for lab, name in methods]
    runs = [(name, r) for name, r in runs if r is not None]
    if not runs:
        sys.exit("no saved runs found in " + a.models)
    S = {name: setup(a.models, r[0], r[1], a.data_root) for name, r in runs}
    first = next(iter(S.values()))
    C = core.NUM_CLASSES
    counts = np.bincount(torch.cat(first["ys"]).cpu().numpy(), minlength=C)
    tag = f"{ds} (long tail, seed {a.seed}; class counts {counts.max()}..{counts.min()})"

    # V1: real test features of the frozen extractor
    Xte = first["feats_te"].flatten(1).cpu().numpy()
    yte = first["yte"].cpu().numpy()
    Xs, ys_ = subsample(Xte, yte, a.n, C, rng)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    scatter(ax, tsne(Xs), ys_, C, label_classes=True)
    ax.legend(markerscale=3, fontsize=6, ncol=2, title="class", loc="best")
    ax.set_title(f"Real test features (frozen extractor), t-SNE\n{tag}")
    fig.savefig(os.path.join(a.out, f"V1_real_features_{ds}.png"), bbox_inches="tight")
    plt.close(fig)

    # V2: real vs generated features, one panel per method with a generator
    gens = [(n, s) for n, s in S.items() if s["Gs"] is not None]
    Xtr = torch.cat(first["feats"]).flatten(1).cpu().numpy()
    ytr = torch.cat(first["ys"]).cpu().numpy()
    Xr, yr = subsample(Xtr, ytr, a.n, C, rng)
    fig, axs = plt.subplots(1, len(gens), figsize=(4 * len(gens), 4), squeeze=False)
    for ax, (name, s) in zip(axs[0], gens):
        core.set_seed(0)
        yf = torch.arange(C, device=core.DEV).repeat_interleave(a.n)
        Xf = s["Gs"].sample(yf).flatten(1).cpu().numpy()
        Z = tsne(np.concatenate([Xr, Xf]))
        scatter(ax, Z[:len(yr)], yr, C, marker="o", s=6, alpha=0.25)
        scatter(ax, Z[len(yr):], yf.cpu().numpy(), C, marker="x", s=9, alpha=0.9)
        ax.set_title(f"{name}\nreal (faint dots) vs generated (crosses)")
    fig.suptitle(f"Do generated features match real ones, class by class? {tag}", y=1.02)
    fig.savefig(os.path.join(a.out, f"V2_real_vs_fake_{ds}.png"), bbox_inches="tight")
    plt.close(fig)

    # V3: the generator's latent space on real data
    fig, axs = plt.subplots(1, len(gens), figsize=(4 * len(gens), 4), squeeze=False)
    Xr_t = torch.from_numpy(Xr).to(core.DEV).view(-1, *first["feats"][0].shape[1:])
    yr_t = torch.from_numpy(yr).to(core.DEV)
    for ax, (name, s) in zip(axs[0], gens):
        mu = generator_latent(s["G"], Xr_t, yr_t).cpu().numpy()
        scatter(ax, tsne(mu) if mu.shape[1] > 2 else mu, yr, C)
        ax.set_title(f"{name}: latent code of real features")
    fig.suptitle("Generator latent space (encoder means) on real training data; the class is given to the decoder "
                 "separately, so a class-free latent is the intended outcome", y=1.02)
    fig.savefig(os.path.join(a.out, f"V3_generator_latent_{ds}.png"), bbox_inches="tight")
    plt.close(fig)

    # V4: classifier body embedding of real test data, one architecture
    Xte_t, yte_t = first["feats_te"], first["yte"]
    fig, axs = plt.subplots(1, len(S), figsize=(4 * len(S), 4), squeeze=False)
    sel = np.concatenate([rng.choice(np.where(yte == c)[0], min(a.n, int((yte == c).sum())), replace=False) for c in range(C)])
    for ax, (name, s) in zip(axs[0], S.items()):
        with torch.no_grad():
            e = s["heads"][a.arch].embed(Xte_t[sel]).cpu().numpy()
        scatter(ax, tsne(e), yte[sel], C)
        ax.set_title(f"{name}: classifier embedding\n(architecture {a.arch + 1}, test data)")
    fig.savefig(os.path.join(a.out, f"V4_classifier_embedding_{ds}.png"), bbox_inches="tight")
    plt.close(fig)

    # V5 + metrics: confusion matrices and per-class precision / recall / F1, averaged over the ten heads
    lines = [f"### Per-class precision, recall and F1, {tag}", "",
             "Averaged over the ten architectures' heads; test set (balanced). Class 0 = most common, class "
             f"{C - 1} = rarest.", ""]
    fig, axs = plt.subplots(2, len(S), figsize=(3.6 * len(S), 7), squeeze=False)
    for j, (name, s) in enumerate(S.items()):
        for i, (mode, fn) in enumerate([("as trained", lambda h: h(Xte_t)), ("exact readout", lambda h: er_logits(s, h, Xte_t))]):
            cms, P, Rr, F1 = [], [], [], []
            for h in s["heads"]:
                cm, p, r, f = per_class_prf(fn(h), yte_t, C)
                cms.append(cm / cm.sum(1, keepdim=True).clamp(min=1))
                P.append(p); Rr.append(r); F1.append(f)
            cm = torch.stack(cms).mean(0).numpy()
            ax = axs[i, j]
            ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(C)); ax.set_yticks(range(C))
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_title(f"{name}, {mode}")
            if C <= 10:
                for t in range(C):
                    ax.text(t, t, f"{100 * cm[t, t]:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if cm[t, t] > 0.5 else "black")
            P, Rr, F1 = (100 * np.mean(x, 0) for x in (P, Rr, F1))
            lines += [f"**{name}, {mode}:** macro precision {P.mean():.1f}, macro recall {Rr.mean():.1f}, "
                      f"macro-F1 {F1.mean():.1f}", "",
                      "| | " + " | ".join(str(c) for c in range(C)) + " |", "|---|" + "---|" * C,
                      "| precision | " + " | ".join(f"{v:.0f}" for v in P) + " |",
                      "| recall | " + " | ".join(f"{v:.0f}" for v in Rr) + " |",
                      "| F1 | " + " | ".join(f"{v:.0f}" for v in F1) + " |", ""]
    fig.suptitle(f"Confusion matrices (row-normalised; diagonal = recall), mean of ten heads; {tag}", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"V5_confusion_{ds}.png"), bbox_inches="tight")
    plt.close(fig)
    open(os.path.join(a.out, f"V_metrics_{ds}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
