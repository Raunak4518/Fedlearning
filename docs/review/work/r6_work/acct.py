import math
def rho(eps,d): L=math.log(1/d); return (math.sqrt(L+eps)-math.sqrt(L))**2
def eps_of(r,d): return r+2*math.sqrt(r*math.log(1/d))
for e in (8,2):
    r=rho(e,1e-5); s=math.sqrt(2/r)
    print(f"eps={e}: rho={r:.4f} back-conv eps={eps_of(r,1e-5):.4f} sig1={s:.3f} per-release rho={1/(2*s*s):.4f} (x4={4/(2*s*s):.4f})")
    for D,name in ((768,'S1/MC'),(6144,'RFF/KH')):
        nmin=2*s*math.sqrt(D)
        print(f"   gate n_c >= {nmin:.0f} for {name}")
    for n in (2417,1449,868,520,312,187,112,67,40,24):
        print(f"     n={n}: rel noise norm of mean (S1) = {s*math.sqrt(768)/n:.2f} clip; RFF {s*math.sqrt(6144)/n:.2f}; count sd {s:.2f}")
# Laplace counts K02: per-client histogram eps; if re-drawn per round?
