"""E40 / E41 - Does the rare-row collapse of a FedAvg-trained generator happen in federated embedding models in general?

The mechanism of paper §3 needs only three things: a parameter row that receives a data gradient only from samples
that touch it (an embedding row), clients that never touch it, and local Adam with coupled (L2) weight decay,
followed by plain FedAvg. A non-holder's gradient on the row is then exactly lambda * r, which Adam normalises to a
step of size ~eta toward zero on every local step; averaging puts those shrunk copies back into the global row.
Approximation 5 predicts collapse from counts alone: kappa_c = sum_{k holds c} tau_k f_kc / sum_{k lacks c} tau_k
(tau_k: local steps; f_kc: probability that a batch of client k contains item c). Nothing here uses GeFL code.

Task "mf" (E40): federated matrix factorisation on MovieLens-1M explicit ratings (FedMF-style): item embeddings and
item biases are shared and aggregated, user embeddings and biases stay on their client. Users are split at random
into K clients (cross-silo groups). Ratings: 90 / 10 train / test per user.

Arms (same split, same initialisation per seed):
  adam_l2       local torch.optim.Adam(weight_decay=lambda) (coupled L2: the common choice), FedAvg by n_k
  adamw         local AdamW, decoupled decay lambda                                         , FedAvg by n_k
  adam          local Adam without decay                                                    , FedAvg by n_k
  adam_l2_hwa   as adam_l2, but each item row averaged over the clients that hold the item, weighted by their count
  sgd_l2        local SGD with coupled L2 (no normalisation)                                , FedAvg by n_k
  adam_l2_reset as adam_l2 with each client's Adam state reset every round (another common implementation)
  central_l2    centralised Adam + L2 on the pooled data (same number of epochs)

Task "lm" (E41): federated next-word prediction on WikiText-2 (raw), clients = random groups of Wikipedia articles,
so rare words live in few clients. Model: the previous 3 words' input embeddings (shared table, the rows we study)
-> hidden layer -> output layer; trained with a sampled softmax (the batch's targets plus 1,024 uniform negatives;
every output row is touched often on every client), evaluated with the full softmax. Vocabulary: the 10,000 most
frequent training words (others -> <unk>). Rows: input embeddings; holder counts = windows whose context contains
the word. Arms as for "mf" (HWA averages each input row over the clients whose windows contain the word).
E1-E4 below apply to E41 with "item" -> "word" and test RMSE -> test perplexity on windows whose context contains a
word with kappa < 1.

PRE-REGISTERED (19 Oct 2026, before any run):
  E1  under adam_l2, every item row with kappa_c < 1 ends at <= 0.15 of its initial norm; Spearman(log kappa, final
      norm) >= 0.8 over items with a non-holder.
  E2  no item row with kappa_c < 1 collapses (<= 0.15) under adamw, adam, adam_l2_hwa or central_l2.
  E3  the collapse costs accuracy where it happens: test RMSE on ratings of tail items (the items with kappa_c < 1)
      is higher under adam_l2 than under adamw / adam_l2_hwa, at every K.
  E4  the fraction of collapsed items grows with K (more non-holders per item).
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.environ.get("GEFL_DATA_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def hyper_present(n, m, b):
    """P(a batch of b drawn without replacement from n items contains at least one of m marked items)."""
    if m <= 0:
        return 0.0
    if n - m < b:
        return 1.0
    return 1.0 - math.exp(math.lgamma(n - m + 1) - math.lgamma(n - m - b + 1) - math.lgamma(n + 1) + math.lgamma(n - b + 1))


def kappa(counts, tau, B, with_fbar=False):
    """counts: (K, V) per-client row usage; tau: (K,) local steps. Returns kappa per row (inf if no non-holder) and,
    with with_fbar, fbar per row: the share of all local optimiser steps whose batch carries data for the row (the
    step-frequency predictor that also covers centralised training, K = 1)."""
    K, V = counts.shape
    nk = counts.sum(1)
    out, fbar = np.empty(V), np.empty(V)
    for c in range(V):
        H = counts[:, c] > 0
        f = np.array([hyper_present(int(nk[k]), int(counts[k, c]), min(B, int(nk[k]))) for k in np.nonzero(H)[0]])
        fbar[c] = (tau[H] * f).sum() / tau.sum()
        out[c] = np.inf if H.all() else (tau[H] * f).sum() / tau[~H].sum()
    return (out, fbar) if with_fbar else out


# ------------------------------------------------------------------ MovieLens-1M federated MF
def fetch(path, url):
    """Download a dataset file once (MovieLens from GroupLens, WikiText-2 parquet from Hugging Face)."""
    if os.path.exists(path):
        return
    import urllib.request, zipfile
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.part{os.getpid()}"  # two notebook workers may download at once: private temp file, atomic rename
    urllib.request.urlretrieve(url, tmp)
    if url.endswith(".zip"):
        with zipfile.ZipFile(tmp) as z:
            z.extractall(os.path.dirname(os.path.dirname(path)))
        os.remove(tmp)
    else:
        os.replace(tmp, path)


def load_ml1m(seed):
    path = os.path.join(ROOT, "ml-1m", "ratings.dat")
    fetch(path, "https://files.grouplens.org/datasets/movielens/ml-1m.zip")
    a = np.array([list(map(int, l.split("::")[:3])) for l in open(path)], dtype=np.int64)
    u, i, r = a[:, 0], a[:, 1], a[:, 2].astype(np.float32)
    _, u = np.unique(u, return_inverse=True)
    _, i = np.unique(i, return_inverse=True)
    rng = np.random.RandomState(seed)
    test = np.zeros(len(u), bool)  # 10% of each user's ratings (rounded up) go to the test set, chosen at random
    order = rng.permutation(len(u))
    seen = {}
    us = u[order]
    cnt = np.bincount(u)
    take = np.ceil(0.1 * cnt).astype(int)
    for j, uu in zip(order, us):
        s = seen.get(uu, 0)
        if s < take[uu]:
            test[j] = True
            seen[uu] = s + 1
    return u, i, r, test


def run_mf(arm, K, seed, rounds=60, epochs=1, d=32, B=256, lr=1e-3, lam=1e-4, log_every=5):
    u, i, r, test = load_ml1m(seed)
    U, V = u.max() + 1, i.max() + 1
    mu = float(r[~test].mean())
    rng = np.random.RandomState(seed + 1)
    client_of_user = rng.randint(0, K, size=U)
    tr = ~test
    cl = client_of_user[u]
    torch.manual_seed(seed)
    item0 = 0.1 * torch.randn(V, d)
    user = 0.1 * torch.randn(U, d)
    ib0, ub = torch.zeros(V), torch.zeros(U)
    item, ib = item0.clone(), ib0.clone()
    U_t, I_t, R_t = torch.from_numpy(u), torch.from_numpy(i), torch.from_numpy(r)
    data = [np.nonzero(tr & (cl == k))[0] for k in range(K)]
    counts = np.zeros((K, V), np.int64)
    for k in range(K):
        counts[k] = np.bincount(i[data[k]], minlength=V)
    n_k = np.array([len(x) for x in data], float)
    tau = np.array([epochs * math.ceil(n / B) for n in n_k], float)
    kap, fbar = kappa(counts, tau, B, with_fbar=True)
    init_norm = item0.norm(dim=1).numpy()

    def make_opt(params):
        if arm in ("adam_l2", "adam_l2_hwa", "adam_l2_reset", "central_l2"):
            return torch.optim.Adam(params, lr=lr, weight_decay=lam)
        if arm == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=lam)
        if arm == "adam":
            return torch.optim.Adam(params, lr=lr)
        if arm == "sgd_l2":
            return torch.optim.SGD(params, lr=0.05, weight_decay=lam)
        raise ValueError(arm)

    def step(ii, uu, rr, it, ibb, us, ubb, opt):
        pred = mu + (it[ii] * us[uu]).sum(1) + ibb[ii] + ubb[uu]
        loss = F.mse_loss(pred, rr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    hist = []
    t0 = time.time()
    if arm == "central_l2":
        it, ibb = item.clone().requires_grad_(), ib.clone().requires_grad_()
        us, ubb = user.clone().requires_grad_(), ub.clone().requires_grad_()
        opt = make_opt([it, ibb, us, ubb])
        allidx = np.nonzero(tr)[0]
        for rd in range(rounds):
            for _ in range(epochs):
                perm = rng.permutation(allidx)
                for s in range(0, len(perm), B):
                    b = perm[s:s + B]
                    step(I_t[b], U_t[b], R_t[b], it, ibb, us, ubb, opt)
        item, ib, user, ub = it.detach(), ibb.detach(), us.detach(), ubb.detach()
    else:
        states = [None] * K  # persistent per-client optimiser state (except adam_l2_reset)
        for rd in range(rounds):
            acc_i, acc_b, wsum = torch.zeros(V, d), torch.zeros(V), 0.0
            hw_i, hw_b, hw_w = torch.zeros(V, d), torch.zeros(V), torch.zeros(V)
            for k in range(K):
                it, ibb = item.clone().requires_grad_(), ib.clone().requires_grad_()
                mine = client_of_user == k
                us, ubb = user.clone().requires_grad_(), ub.clone().requires_grad_()
                opt = make_opt([it, ibb, us, ubb])
                if states[k] is not None and arm != "adam_l2_reset":
                    opt.load_state_dict(states[k])
                for _ in range(epochs):
                    perm = rng.permutation(data[k])
                    for s in range(0, len(perm), B):
                        b = perm[s:s + B]
                        step(I_t[b], U_t[b], R_t[b], it, ibb, us, ubb, opt)
                states[k] = opt.state_dict()
                # user rows stay local: only this client's users change
                m = torch.from_numpy(mine)
                user[m], ub[m] = us.detach()[m], ubb.detach()[m]
                acc_i += n_k[k] * it.detach()
                acc_b += n_k[k] * ibb.detach()
                wsum += n_k[k]
                w = torch.from_numpy(counts[k]).float()
                hw_i += w[:, None] * it.detach()
                hw_b += w * ibb.detach()
                hw_w += w
            if arm == "adam_l2_hwa":
                held = hw_w > 0
                item = item.clone(); ib = ib.clone()
                item[held] = hw_i[held] / hw_w[held, None]
                ib[held] = hw_b[held] / hw_w[held]
            else:
                item, ib = acc_i / wsum, acc_b / wsum
            if rd % log_every == 0 or rd == rounds - 1:
                hist.append(dict(round=rd, median_norm_ratio=float(np.median(item.norm(dim=1).numpy() / init_norm))))
    # evaluation
    with torch.no_grad():
        te = np.nonzero(test)[0]
        pred = (mu + (item[I_t[te]] * user[U_t[te]]).sum(1) + ib[I_t[te]] + ub[U_t[te]]).clamp(1, 5)
        err = (pred - R_t[te]) ** 2
    norm = item.norm(dim=1).numpy() / init_norm
    ntr = counts.sum(0)
    fin = np.isfinite(kap)
    lo = kap < 1
    te_items = i[te]
    tail_mask = np.isin(te_items, np.nonzero(lo)[0])
    pop = np.isin(te_items, np.nonzero(~lo)[0])
    from scipy.stats import spearmanr
    res = dict(task="mf", arm=arm, K=K, seed=seed, rounds=rounds, lam=lam, lr=lr, d=d, B=B,
               rmse=float(err.mean().sqrt()), rmse_kappa_lt1=float(err[torch.from_numpy(tail_mask)].mean().sqrt())
               if tail_mask.any() else None, rmse_kappa_ge1=float(err[torch.from_numpy(pop)].mean().sqrt()),
               n_test_kappa_lt1=int(tail_mask.sum()), n_items=int(V), n_kappa_lt1=int(lo.sum()),
               collapsed_kappa_lt1=int((norm[lo] <= 0.15).sum()), collapsed_kappa_ge1=int((norm[~lo] <= 0.15).sum()),
               max_norm_kappa_lt1=float(norm[lo].max()) if lo.any() else None,
               median_norm_kappa_lt1=float(np.median(norm[lo])) if lo.any() else None,
               median_norm_kappa_ge1=float(np.median(norm[~lo])),
               spearman_logkappa_norm=float(spearmanr(np.log(kap[fin]), norm[fin])[0]) if fin.sum() > 2 else None,
               time_s=round(time.time() - t0, 1), history=hist,
               kappa=[round(float(x), 4) if np.isfinite(x) else None for x in kap],
               norm_ratio=[round(float(x), 4) for x in norm], train_count=ntr.tolist(),
               fbar=[float(f"{x:.4g}") for x in fbar])
    return res


# ------------------------------------------------------------------ WikiText-2 federated next-word prediction
def load_wt2(V=10000, ctx=3):
    import pyarrow.parquet as pq, re, collections
    def articles(split):
        path = os.path.join(ROOT, "wikitext2", f"{split}.parquet")
        fetch(path, "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/"
                    f"{split}-00000-of-00001.parquet")
        t = pq.read_table(path).to_pydict()["text"]
        arts, cur = [], []
        for line in t:
            if re.match(r"^ = [^=].* = \n$", line):
                if cur:
                    arts.append(cur)
                cur = []
            cur += line.split() + ["<eos>"] if line.strip() else []
        arts.append(cur)
        return arts
    tr, te = articles("train"), articles("test")
    cnt = collections.Counter(w for a in tr for w in a)
    vocab = ["<unk>"] + [w for w, _ in cnt.most_common(V - 1)]
    idx = {w: j for j, w in enumerate(vocab)}
    enc = lambda arts: [np.array([idx.get(w, 0) for w in a], np.int64) for a in arts]
    return enc(tr), enc(te)


def windows(art, ctx=3):
    if len(art) <= ctx:
        return np.zeros((0, ctx), np.int64), np.zeros(0, np.int64)
    X = np.stack([art[j:len(art) - ctx + j] for j in range(ctx)], 1)
    return X, art[ctx:]


def run_lm(arm, K, seed, rounds=20, d=64, h=128, B=256, lr=1e-3, lam=1e-4, V=10000, ctx=3, neg=1024, log_every=5):
    tr_arts, te_arts = load_wt2(V, ctx)
    rng = np.random.RandomState(seed)
    owner = rng.randint(0, K, size=len(tr_arts))
    Xs, Ys = [], []
    for k in range(K):
        parts = [windows(a, ctx) for a, o in zip(tr_arts, owner) if o == k]
        Xs.append(np.concatenate([p[0] for p in parts])); Ys.append(np.concatenate([p[1] for p in parts]))
    counts = np.zeros((K, V), np.int64)
    for k in range(K):  # windows whose context contains word w (each window counted once per distinct word)
        Xk = Xs[k]
        for j in range(ctx):
            first = np.ones(len(Xk), bool)
            for jj in range(j):
                first &= Xk[:, jj] != Xk[:, j]
            counts[k] += np.bincount(Xk[first, j], minlength=V)
    n_k = np.array([len(y) for y in Ys], float)
    tau = np.array([math.ceil(n / B) for n in n_k], float)
    kap, fbar = kappa(counts, tau, B, with_fbar=True)
    torch.manual_seed(seed)
    P0 = dict(E=0.1 * torch.randn(V, d), W1=torch.randn(ctx * d, h) / math.sqrt(ctx * d), b1=torch.zeros(h),
              W2=torch.randn(V, h) / math.sqrt(h), b2=torch.zeros(V))
    init_norm = P0["E"].norm(dim=1).numpy()
    P0 = {k_: v.to(DEV) for k_, v in P0.items()}
    Xs = [torch.from_numpy(x).to(DEV) for x in Xs]
    Ys = [torch.from_numpy(y).to(DEV) for y in Ys]
    gen = torch.Generator(device=DEV).manual_seed(seed + 5)
    P = {k_: v.clone() for k_, v in P0.items()}

    def make_opt(params):
        if arm in ("adam_l2", "adam_l2_hwa", "adam_l2_reset", "central_l2"):
            return torch.optim.Adam(params, lr=lr, weight_decay=lam)
        if arm == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=lam)
        if arm == "adam":
            return torch.optim.Adam(params, lr=lr)
        if arm == "sgd_l2":
            return torch.optim.SGD(params, lr=0.1, weight_decay=lam)
        raise ValueError(arm)

    def train(Q, opt, X, Y, g):
        perm = torch.randperm(len(Y), device=DEV, generator=gen)
        for s0 in range(0, len(perm), B):
            b = perm[s0:s0 + B]
            x, y = X[b], Y[b]
            cand = torch.unique(torch.cat([y, torch.randint(0, V, (neg,), device=DEV, generator=gen)]))
            pos = torch.searchsorted(cand, y)
            hid = torch.relu(Q["E"][x].flatten(1) @ Q["W1"] + Q["b1"])
            logit = hid @ Q["W2"][cand].T + Q["b2"][cand]
            loss = F.cross_entropy(logit, pos)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    t0 = time.time()
    hist = []
    if arm == "central_l2":
        Q = {k_: v.clone().requires_grad_() for k_, v in P.items()}
        opt = make_opt(list(Q.values()))
        X, Y = torch.cat(Xs), torch.cat(Ys)
        for rd in range(rounds):
            train(Q, opt, X, Y, rng)
        P = {k_: v.detach() for k_, v in Q.items()}
    else:
        states = [None] * K
        for rd in range(rounds):
            acc = {k_: torch.zeros_like(v) for k_, v in P.items()}
            hwE, hwW = torch.zeros(V, d, device=DEV), torch.zeros(V, device=DEV)
            for k in range(K):
                Q = {k_: v.clone().requires_grad_() for k_, v in P.items()}
                opt = make_opt(list(Q.values()))
                if states[k] is not None and arm != "adam_l2_reset":
                    opt.load_state_dict(states[k])
                train(Q, opt, Xs[k], Ys[k], rng)
                states[k] = opt.state_dict()
                for k_ in P:
                    acc[k_] += n_k[k] * Q[k_].detach()
                w = torch.from_numpy(counts[k]).float().to(DEV)
                hwE += w[:, None] * Q["E"].detach()
                hwW += w
            P = {k_: v / n_k.sum() for k_, v in acc.items()}
            if arm == "adam_l2_hwa":
                held = hwW > 0
                P["E"][held] = hwE[held] / hwW[held, None]
            if rd % log_every == 0 or rd == rounds - 1:
                hist.append(dict(round=rd, median_norm_ratio=float(np.median(P["E"].norm(dim=1).cpu().numpy() / init_norm))))
    # evaluation: full softmax on the test articles
    Xt = np.concatenate([windows(a, ctx)[0] for a in te_arts]); Yt = np.concatenate([windows(a, ctx)[1] for a in te_arts])
    nll = []
    with torch.no_grad():
        for s0 in range(0, len(Yt), 4096):
            x, y = torch.from_numpy(Xt[s0:s0 + 4096]).to(DEV), torch.from_numpy(Yt[s0:s0 + 4096]).to(DEV)
            hid = torch.relu(P["E"][x].flatten(1) @ P["W1"] + P["b1"])
            nll.append(F.cross_entropy(hid @ P["W2"].T + P["b2"], y, reduction="none"))
    nll = torch.cat(nll).cpu().numpy()
    norm = P["E"].norm(dim=1).cpu().numpy() / init_norm
    lo = kap < 1
    lo[0] = False  # <unk> is held everywhere anyway
    rare_ctx = lo[Xt].any(1)
    fin = np.isfinite(kap)
    from scipy.stats import spearmanr
    return dict(task="lm", arm=arm, K=K, seed=seed, rounds=rounds, lam=lam, lr=lr, d=d, h=h, B=B, V=V,
                ppl=float(np.exp(nll.mean())), ppl_rare_ctx=float(np.exp(nll[rare_ctx].mean())) if rare_ctx.any() else None,
                ppl_other_ctx=float(np.exp(nll[~rare_ctx].mean())), n_test_rare_ctx=int(rare_ctx.sum()),
                n_rows=int(V), n_kappa_lt1=int(lo.sum()), collapsed_kappa_lt1=int((norm[lo] <= 0.15).sum()),
                collapsed_kappa_ge1=int((norm[~lo] <= 0.15).sum()),
                max_norm_kappa_lt1=float(norm[lo].max()) if lo.any() else None,
                median_norm_kappa_lt1=float(np.median(norm[lo])) if lo.any() else None,
                median_norm_kappa_ge1=float(np.median(norm[~lo])),
                spearman_logkappa_norm=float(spearmanr(np.log(kap[fin]), norm[fin])[0]) if fin.sum() > 2 else None,
                time_s=round(time.time() - t0, 1), history=hist,
                kappa=[round(float(x), 4) if np.isfinite(x) else None for x in kap],
                norm_ratio=[round(float(x), 4) for x in norm], train_count=counts.sum(0).tolist(),
                fbar=[float(f"{x:.4g}") for x in fbar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mf")
    ap.add_argument("--arms", nargs="+", default=["adam_l2", "adamw", "adam", "adam_l2_hwa", "sgd_l2", "adam_l2_reset",
                                                  "central_l2"])
    ap.add_argument("--K", type=int, nargs="+", default=[10, 50])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("NT", "2")))
    out = a.out or os.path.join(OUT, "E40_fedembed_mf" if a.task == "mf" else "E41_fedembed_lm")
    os.makedirs(out, exist_ok=True)
    jl = os.path.join(out, "runs.jsonl")
    done = set()
    if os.path.exists(jl):
        done = {(json.loads(l)["arm"], json.loads(l)["K"], json.loads(l)["seed"]) for l in open(jl)}
    for seed in a.seeds:
        for K in a.K:
            for arm in a.arms:
                if arm == "central_l2" and K != a.K[0]:
                    continue  # one centralised run per seed
                if (arm, K, seed) in done:
                    continue
                res = (run_mf if a.task == "mf" else run_lm)(arm, K, seed, rounds=a.rounds)
                with open(jl, "a") as f:
                    f.write(json.dumps(res) + "\n")
                print(json.dumps({k: v for k, v in res.items() if k not in ("kappa", "norm_ratio", "train_count", "history")}),
                      flush=True)


if __name__ == "__main__":
    main()
