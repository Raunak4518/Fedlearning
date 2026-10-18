"""N1: exact checks of the quadratic statements in the report (Props. 1.2, 1.3, 1.4, 2.6).

(a) FedAvg fixed point for quadratic clients F_k = 1/2 (x - x_k)^T A_k (x - x_k), tau_k full-batch GD steps, lr eta,
    aggregation weights p_k:   x_FA = (sum p_k Q_k)^-1 sum p_k Q_k x_k,   Q_k = I - (I - eta A_k)^tau_k.
    Compare with the pooled-ERM minimiser x* = (sum w_k A_k)^-1 sum w_k A_k x_k, w_k = n_k / N.
    With p_k = 1/K and tau_k proportional to n_k (Stage I of GeFL-F), x_FA -> x* as eta -> 0, error O(eta).
(b) SCAFFOLD-type control variates (full participation, exact gradients): fixed point = x* for every eta, tau.
(c) Alternation (S_s synthetic steps on H_s, then S_r real steps on H_r) fixed point
    theta* = (I - P_r P_s)^-1 [(I - P_r) th_r + P_r (I - P_s) th_s].
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
import numpy as np

rng = np.random.default_rng(0)
d, K = 6, 10


def rand_spd(scale):
    M = rng.normal(size=(d, d))
    U, _ = np.linalg.qr(M)
    ev = scale * np.exp(rng.uniform(-1.5, 0.5, d))
    return U @ np.diag(ev) @ U.T


def fedavg(A, xs, p, tau, eta, rounds=20000, control=False):
    x = np.zeros(d)
    for _ in range(rounds):
        g_glob = sum(pk * Ak @ (x - xk) for pk, Ak, xk in zip(p, A, xs)) / sum(p)
        upd = np.zeros(d)
        for pk, Ak, xk, tk in zip(p, A, xs, tau):
            y = x.copy()
            ck = Ak @ (x - xk)
            for _ in range(tk):
                g = Ak @ (y - xk)
                if control:
                    g = g - ck + g_glob
                y = y - eta * g
            upd += pk * (y - x)
        x_new = x + upd / sum(p)
        if np.linalg.norm(x_new - x) < 1e-13:
            x = x_new
            break
        x = x_new
    return x


A = [rand_spd(1.0) for _ in range(K)]
xs = [rng.normal(size=d) * 2 for _ in range(K)]
n = rng.integers(50, 1500, K)
w = n / n.sum()
xstar = np.linalg.solve(sum(wk * Ak for wk, Ak in zip(w, A)), sum(wk * Ak @ xk for wk, Ak, xk in zip(w, A, xs)))
p = np.ones(K) / K
print("(a) FedAvg, uniform p_k = 1/K, tau_k = ceil(n_k/64)*5 (Stage-I rule)")
tau = np.ceil(n / 64).astype(int) * 5
for eta in (0.1, 0.03, 0.01, 0.003, 0.001):
    Q = [np.eye(d) - np.linalg.matrix_power(np.eye(d) - eta * Ak, tk) for Ak, tk in zip(A, tau)]
    x_formula = np.linalg.solve(sum(pk * Qk for pk, Qk in zip(p, Q)), sum(pk * Qk @ xk for pk, Qk, xk in zip(p, Q, xs)))
    x_sim = fedavg(A, xs, p, tau, eta, rounds=4000)
    xbar_loc = np.mean(xs, 0)
    print(f"  eta={eta:6.3f} eta*tau_mean={eta*tau.mean():6.2f}  |x_sim - x_formula|={np.linalg.norm(x_sim-x_formula):.1e}"
          f"  |x_FA - x*|={np.linalg.norm(x_formula-xstar):.4f}  |x_FA - mean_k x_k|={np.linalg.norm(x_formula-xbar_loc):.4f}")
print("  (equal tau, uniform p: surrogate = unweighted client average objective, NOT pooled)")
tau_eq = np.full(K, int(tau.mean()))
for eta in (0.01, 0.001):
    Q = [np.eye(d) - np.linalg.matrix_power(np.eye(d) - eta * Ak, tk) for Ak, tk in zip(A, tau_eq)]
    x_formula = np.linalg.solve(sum(pk * Qk for pk, Qk in zip(p, Q)), sum(pk * Qk @ xk for pk, Qk, xk in zip(p, Q, xs)))
    x_unw = np.linalg.solve(sum(A), sum(Ak @ xk for Ak, xk in zip(A, xs)))
    print(f"  eta={eta}: |x_FA - x*|={np.linalg.norm(x_formula-xstar):.4f}  |x_FA - argmin sum_k F_k|={np.linalg.norm(x_formula-x_unw):.4f}")

print("(b) control variates (SCAFFOLD-type, exact): fixed point vs x*")
for eta in (0.1, 0.03):
    x_cv = fedavg(A, xs, w, tau, eta, rounds=3000, control=True)
    print(f"  eta={eta}: |x_cv - x*| = {np.linalg.norm(x_cv - xstar):.2e}")

print("(c) alternation fixed point")
Hr, Hs = rand_spd(1.0), rand_spd(1.0)
# make H_r rank-deficient: the client's real data do not constrain 2 directions (absent classes)
U, _ = np.linalg.qr(rng.normal(size=(d, d)))
ev = np.array([0, 0, 0.5, 1, 2, 3.0])
Hr = U @ np.diag(ev) @ U.T
thr, ths = rng.normal(size=d), rng.normal(size=d)
eta = 0.1
for Sr, Ss in ((50, 9), (50, 94), (5, 1), (1, 1)):
    Pr = np.linalg.matrix_power(np.eye(d) - eta * Hr, Sr)
    Ps = np.linalg.matrix_power(np.eye(d) - eta * Hs, Ss)
    th_formula = np.linalg.solve(np.eye(d) - Pr @ Ps, (np.eye(d) - Pr) @ thr + Pr @ (np.eye(d) - Ps) @ ths)
    th = np.zeros(d)
    for _ in range(20000):
        th = ths + Ps @ (th - ths)
        th = thr + Pr @ (th - thr)
    Om = np.linalg.solve(np.eye(d) - Pr @ Ps, Pr @ (np.eye(d) - Ps))
    # synthetic weight seen in the eigenbasis of H_r
    om_diag = np.diag(U.T @ Om @ U)
    joint = np.linalg.solve(Sr * Hr + Ss * Hs, Sr * Hr @ thr + Ss * Hs @ ths)
    print(f"  S_r={Sr:3d} S_s={Ss:3d}: |iter - formula|={np.linalg.norm(th-th_formula):.1e}; synthetic weight per H_r eigen-direction "
          f"(h_r={ev.tolist()}): {np.round(om_diag,3).tolist()};  |theta* - joint-objective minimiser|={np.linalg.norm(th_formula-joint):.3f}")

print("(a') tau_k exactly proportional to n_k (tau_k = n_k / 10): x_FA - x* should be O(eta)")
tau_p = (n // 10).astype(int)
n_p = tau_p * 10
xstar_p = np.linalg.solve(sum((nk / n_p.sum()) * Ak for nk, Ak in zip(n_p, A)), sum((nk / n_p.sum()) * Ak @ xk for nk, Ak, xk in zip(n_p, A, xs)))
for eta in (0.01, 0.003, 0.001, 0.0003, 0.0001):
    Q = [np.eye(d) - np.linalg.matrix_power(np.eye(d) - eta * Ak, tk) for Ak, tk in zip(A, tau_p)]
    x_formula = np.linalg.solve(sum(pk * Qk for pk, Qk in zip(p, Q)), sum(pk * Qk @ xk for pk, Qk, xk in zip(p, Q, xs)))
    print(f"  eta={eta:7.4f} eta*tau_mean={eta*tau_p.mean():6.3f}  |x_FA - x*| = {np.linalg.norm(x_formula - xstar_p):.5f}")
