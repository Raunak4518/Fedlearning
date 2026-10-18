"""N3b: variance inflation of the saturated flat-FedAvg row fixed point (Prop. 2.4), SGD, actual MNIST s0 counts.

Row model x | c ~ N(m_c, s^2 I). Local training: SGD on the per-client mean-squared row loss until the local
row reaches the local class mean (saturated regime: many local steps). Aggregation:
  flat (no decay; non-holders return the row unchanged)  -> fixed point = unweighted mean of holders' local means
  HWA with E(n) weights (beta=.999)                        -> fixed point = E(n)-weighted mean
  pooled MLE                                               -> count-weighted mean
MSE of each estimator of the TRUE m_c over 2000 draws of the data, divided by s^2 d / n_c (the pooled MLE's MSE).
Prediction: flat ratio = V_c = (n_c/m_c^2) sum_{H_c} 1/n_kc, HWA ratio ~ 1 (E(n) ~ n for n << 1000).
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
import numpy as np
OUT = _HERE
cnt = np.load(f"{OUT}/mnist_s0_K10_counts.npy").astype(int)
K, C = cnt.shape
d, s, R = 8, 1.0, 2000
rng = np.random.default_rng(0)
E = (1 - 0.999 ** cnt) / (1 - 0.999)
mse = {k: np.zeros(C) for k in ("flat", "hwa", "pooled")}
for _ in range(R):
    for c in range(C):
        H = np.where(cnt[:, c] > 0)[0]
        locs = np.stack([rng.normal(0, s / np.sqrt(cnt[k, c]), d) for k in H])  # local means (true mean 0)
        mse["flat"][c] += (locs.mean(0) ** 2).sum()
        mse["hwa"][c] += (((E[H, c][:, None] * locs).sum(0) / E[H, c].sum()) ** 2).sum()
        mse["pooled"][c] += (((cnt[H, c][:, None] * locs).sum(0) / cnt[H, c].sum()) ** 2).sum()
nc = cnt.sum(0); m = (cnt > 0).sum(0)
V = np.array([(nc[c] / m[c] ** 2) * (1.0 / cnt[cnt[:, c] > 0, c]).sum() for c in range(C)])
for k in mse:
    print(f"{k:7s} MSE / (s^2 d / n_c):", np.round(mse[k] / R / (s * s * d / nc), 2).tolist())
print("predicted V_c (flat):          ", np.round(V, 2).tolist())
