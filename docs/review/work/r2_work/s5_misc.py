"""S5: (a) Prop 3 variance model vs the actual lambda-weighted estimator (sympy + Monte Carlo);
(b) kernel herding as coded (HerdedGen): is MMD monotone, is it greedy-MMD (||phi||^2 not const);
(c) zCDP accounting (Bun & Steinke Prop 1.3 / Lemma 2.5);  (d) Lemma R1 numeric."""
import numpy as np, sympy as sp, math
rng = np.random.default_rng(0)

# (a)
lam, nr, ns, s2, b = sp.symbols("lambda n_r n_s sigma2 b", positive=True)
w = lam * ns / (nr + lam * ns)
paper = s2 / (nr + lam * ns) + w**2 * b**2
true = s2 * (nr + lam**2 * ns) / (nr + lam * ns)**2 + w**2 * b**2       # Var of (sum real + lam sum syn)/(n_r + lam n_s)
diff = sp.simplify(paper - true)
print("(a) paper MSE - true MSE =", sp.factor(diff))
W = sp.symbols("w", positive=True)
Etrue_w = s2 * ((1 - W)**2 / nr + W**2 / ns) + W**2 * b**2
wstar = sp.solve(sp.diff(Etrue_w, W), W)[0]
print("    true-model optimum w* =", sp.simplify(wstar), " (paper: sigma^2/(2 n_r b^2))")
vals = {nr: 100, ns: 1000, s2: 1.0}
for bb in [0.01, 0.05, 0.1, 0.3]:
    wp = min(float(1 / (2 * 100 * bb**2)), 1000 / 1100); wt = float(wstar.subs(vals).subs(b, bb))
    print(f"    b={bb}: paper w*={wp:.3f}  true w*={wt:.3f}  (w_max={1000/1100:.3f})")
# Monte Carlo of the estimator variance at lambda = 0.3, b = 0
L = 0.3; reps = 200000
R_ = rng.normal(0, 1, (reps, 100)).sum(1); S_ = rng.normal(0, 1, (reps, 1000)).sum(1)
est = (R_ + L * S_) / (100 + L * 1000)
print(f"    MC Var at lambda=0.3: {est.var():.5f}; paper formula {1/(100+300):.5f}; true formula {(100+0.09*1000)/(400**2):.5f}")

# (b) herding as coded: w <- mu; pick argmax <phi_i, w> w/o replacement; w += mu - phi_j
d, Dr = 5, 600
Wf = rng.normal(size=(Dr, d)) / 1.5; bf = rng.uniform(0, 2 * np.pi, Dr)
phi = lambda X: np.sqrt(2 / Dr) * np.cos(X @ Wf.T + bf)
real = rng.normal(0, 1, (5000, d)); mu = phi(real).mean(0)
n = 200; cand = rng.normal(0.4, 0.7, (4 * n, d)); P = phi(cand)
taken = np.zeros(len(P), bool); wv = mu.copy(); S = np.zeros(Dr); mm = []; picks = []
for t in range(n):
    sc = P @ wv; sc[taken] = -np.inf; j = int(sc.argmax()); taken[j] = True; picks.append(j)
    S += P[j]; wv += mu - P[j]; mm.append(np.linalg.norm(S / (t + 1) - mu))
inc = sum(mm[i + 1] > mm[i] for i in range(len(mm) - 1))
print(f"(b) herding MMD: t=1 {mm[0]:.4f}, t=50 {mm[49]:.4f}, t=200 {mm[-1]:.4f}; min over t {min(mm):.4f} at t={int(np.argmin(mm))+1};"
      f" increases in {inc}/{n-1} steps; random 200 of pool: {np.linalg.norm(P[rng.choice(4*n,n,replace=False)].mean(0)-mu):.4f};"
      f" all 800: {np.linalg.norm(P.mean(0)-mu):.4f}")
nrm = (P**2).sum(1); print(f"    ||phi(x)||^2 over candidates: min {nrm.min():.3f} max {nrm.max():.3f} (herding == greedy MMD only if constant)")
# greedy-MMD with the -||phi||^2/2 term for comparison
taken = np.zeros(len(P), bool); S = np.zeros(Dr)
for t in range(n):
    sc = P @ ((t + 1) * mu - S) - 0.5 * nrm; sc[taken] = -np.inf; j = int(sc.argmax()); taken[j] = True; S += P[j]
print(f"    exact greedy-MMD (same pool, w/o replacement) final MMD {np.linalg.norm(S/n-mu):.4f} vs herding {mm[-1]:.4f}")

# (c) zCDP
for eps in [8, 2, 1]:
    delta = 1e-5; Lg = math.log(1 / delta); rho = (math.sqrt(Lg + eps) - math.sqrt(Lg))**2
    back = rho + 2 * math.sqrt(rho * Lg)
    sig = math.sqrt(2 / rho)                    # per release: rho/4 = 1/(2 sig^2)
    print(f"(c) eps={eps}: rho={rho:.4f}, rho+2sqrt(rho log1/d)={back:.4f} (=eps), sigma multiplier={sig:.2f}, per-release rho={1/(2*sig**2):.4f}=rho/4")

# (d) Lemma R1: offsets can undo a common shift, not class-specific shifts
C, d = 4, 6
A = rng.normal(size=(d, d)); Sig = A @ A.T + np.eye(d); Si = np.linalg.inv(Sig); mus = rng.normal(0, 1.2, (C, d))
X = np.concatenate([rng.multivariate_normal(m, Sig, 3000) for m in mus]); Y = np.repeat(np.arange(C), 3000)
def lda(m): return lambda E: E @ (Si @ m.T) - 0.5 * np.einsum('cd,de,ce->c', m, Si, m)
def best_offset_acc(m):
    f = lda(m)(X); best = 0
    off = np.zeros(C)
    for _ in range(300):                         # Saerens-style balancing of predictions (BBC)
        q = np.bincount((f + off).argmax(1), minlength=C) / len(Y); off -= 0.5 * np.log(np.maximum(q, 1e-3) * C)
    return np.mean((f + off).argmax(1) == Y)
base = np.mean(lda(mus)(X).argmax(1) == Y)
common = mus + rng.normal(0, 1.0, d)
specific = mus + rng.normal(0, 1.0, (C, d)) * np.linalg.norm(common - mus) / np.sqrt(d)
print(f"(d) LDA exact {base:.3f}; common shift -> with offsets {best_offset_acc(common):.3f}; class-specific shift -> with offsets {best_offset_acc(specific):.3f}")
