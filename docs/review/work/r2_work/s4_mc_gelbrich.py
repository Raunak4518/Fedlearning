"""S4: Prop 4 (MC) and Prop 6 (Gelbrich / MC-S).
(a) Prop 4 optimality: T* vs other maps meeting the two constraints (exact, population).
(b) The deployed map adds ReLU (MomentCalibratedGen.sample): constraint violation after ReLU,
    on non-negative 'post-ReLU' features (D=768) with a VAE-like under-dispersed generator.
(c) Finite-sample plug-in (n=500 as in the code) and scale clamp [0.1,10].
(d) Prop 6: Gelbrich value vs T*, competitors; effect of eigenvalue clipping of A_c to [0.1,10]
    and of the ReLU on Cov T(X)."""
import numpy as np
rng = np.random.default_rng(0)

# (a) Prop 4, population check with a large sample used as the population
D, N = 20, 200000
X = rng.gamma(0.5, 1.0, size=(N, D)) * rng.uniform(0.2, 2, D)      # skewed, non-Gaussian
mt, Vt = X.mean(0), ((X - X.mean(0))**2).sum(1).mean()
m = rng.uniform(0, 1, D); V = 3.0 * Vt
Tstar = m + np.sqrt(V / Vt) * (X - mt)
disp = lambda Y: ((Y - X)**2).sum(1).mean()
best = disp(Tstar)
formula = V + Vt + ((m - mt)**2).sum() - 2 * np.sqrt(V * Vt)
alts = []
for _ in range(5):                                   # rotated competitors: m + s Q (x - mt) (same mean & spread)
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    alts.append(disp(m + np.sqrt(V / Vt) * (X - mt) @ Q.T))
perm = X[rng.permutation(N)]                          # randomised competitor (independent coupling)
alts.append(disp(m + np.sqrt(V / Vt) * (perm - perm.mean(0))))
print(f"(a) Prop 4: E||T*-X||^2 = {best:.4f}, closed form = {formula:.4f}; competitors min = {min(alts):.4f} (all >= T*: {min(alts) >= best})")

# (b) ReLU after the affine map, realistic non-negative features
D = 768
def feat(n, mu_lat, s):                               # post-ReLU features with ~50% zeros
    return np.maximum(mu_lat + s * rng.normal(size=(n, D)), 0)
mu_real = rng.normal(0, 1, D) * 0.5; mu_gen = mu_real + rng.normal(0, 0.3, D)
Xr = feat(200000, mu_real, 1.0)                       # "real" class: target moments
m_c, V_c = Xr.mean(0), ((Xr - Xr.mean(0))**2).sum(1).mean()
Xg = feat(200000, mu_gen, 0.5)                        # under-dispersed generator
mg, Vg = Xg.mean(0), ((Xg - Xg.mean(0))**2).sum(1).mean()
Y = m_c + np.sqrt(V_c / Vg) * (Xg - mg)
Yr = np.maximum(Y, 0)
mean_err = lambda Z: np.linalg.norm(Z.mean(0) - m_c) / np.sqrt(V_c)
spread = lambda Z: ((Z - Z.mean(0))**2).sum(1).mean() / V_c
print(f"(b) fraction zero in real feats {np.mean(Xr==0):.2f}; scale s={np.sqrt(V_c/Vg):.2f}")
print(f"    before ReLU: mean err/sqrt(V_c) = {mean_err(Y):.4f}, spread ratio = {spread(Y):.4f}, negative coords = {np.mean(Y<0):.3f}")
print(f"    after  ReLU: mean err/sqrt(V_c) = {mean_err(Yr):.4f}, spread ratio = {spread(Yr):.4f}  (Prop 4 constraints NOT met)")
print(f"    displacement E||T-X||^2: affine {((Y-Xg)**2).sum(1).mean():.1f}, after ReLU {((Yr-Xg)**2).sum(1).mean():.1f}")

# (c) finite-sample plug-in: moments of the generator from n=500 draws (code), map applied to fresh draws
errs_m, errs_v = [], []
for rep in range(20):
    S_ = feat(500, mu_gen, 0.5); mg5 = S_.mean(0); Vg5 = ((S_ - mg5)**2).sum(1).mean()
    F_ = feat(20000, mu_gen, 0.5); Y5 = m_c + np.sqrt(V_c / Vg5) * (F_ - mg5)
    errs_m.append(mean_err(Y5)); errs_v.append(spread(Y5))
print(f"(c) n=500 plug-in, no ReLU: mean err/sqrt(V_c) = {np.mean(errs_m):.4f} +- {np.std(errs_m):.4f}; spread ratio {np.mean(errs_v):.4f} +- {np.std(errs_v):.4f}")

# (d) Prop 6 (Gelbrich)
def psd_sqrt(A):
    e, U = np.linalg.eigh((A + A.T) / 2); return (U * np.sqrt(np.clip(e, 0, None))) @ U.T
k = 8
Lg = rng.normal(size=(k, k)); St = Lg @ Lg.T / k + 0.05 * np.eye(k)
Lt = rng.normal(size=(k, k)) * np.linspace(0.05, 3, k); Sig = Lt @ Lt.T / k + 1e-3 * np.eye(k)
N = 400000
Z = rng.standard_t(5, size=(N, k)); Z = (Z - Z.mean(0)); Z = Z @ np.linalg.inv(np.linalg.cholesky(np.cov(Z.T))).T
Xg = Z @ np.linalg.cholesky(St).T; Xg -= Xg.mean(0); St = np.cov(Xg.T)       # non-Gaussian, exact cov St
mt = np.zeros(k); mc = rng.normal(size=k)
Sh, Shi = psd_sqrt(St), np.linalg.inv(psd_sqrt(St))
A = Shi @ psd_sqrt(Sh @ Sig @ Sh) @ Shi
Yd = mc + Xg @ A.T
gel = np.sum((mc - mt)**2) + np.trace(Sig) + np.trace(St) - 2 * np.trace(psd_sqrt(Sh @ Sig @ Sh))
dd = ((Yd - Xg)**2).sum(1).mean()
print(f"(d) Gelbrich: E||T*-X||^2 = {dd:.4f}, Gelbrich formula = {gel:.4f}; ||Cov T* - Sig||_F/||Sig||_F = {np.linalg.norm(np.cov(Yd.T)-Sig)/np.linalg.norm(Sig):.2e}")
comp = []
for _ in range(5):
    Q, _ = np.linalg.qr(rng.normal(size=(k, k)))
    Aq = psd_sqrt(Sig) @ Q @ Shi                          # also gives Cov = Sig
    comp.append(((mc + Xg @ Aq.T - Xg)**2).sum(1).mean())
print(f"    competitors (Sig^1/2 Q St^-1/2): min {min(comp):.4f} >= T*: {min(comp) >= dd - 1e-9}")
e, U = np.linalg.eigh(A); Ac = (U * np.clip(e, 0.1, 10)) @ U.T
print(f"    eig(A_c) range [{e.min():.3f}, {e.max():.3f}]; after clip to [0.1,10]: ||Cov - Sig||_F/||Sig||_F = "
      f"{np.linalg.norm(Ac @ St @ Ac - Sig)/np.linalg.norm(Sig):.3f}")
