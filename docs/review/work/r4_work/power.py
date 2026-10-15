import numpy as np
from scipy import stats, optimize
def power(dz,n,alpha=0.05):
    df=n-1; tc=stats.t.ppf(1-alpha/2,df); nc=dz*np.sqrt(n)
    return stats.nct.sf(tc,df,nc)+stats.nct.cdf(-tc,df,nc)
for n in (3,5,10):
    dz=optimize.brentq(lambda d: power(d,n)-0.8,0.01,min(5,7/np.sqrt(n)))
    print(f"n={n}: dz needed for 80% power (two-sided 0.05 paired t) = {dz:.2f}")
# observed paired-difference sds for small-effect claims
cases={"KH (K09 MNIST)":(-0.71,np.std([-0.81,-0.95,-0.36],ddof=1)),
"KH (K09 FMNIST)":(-0.30,np.std([-0.23,-0.17,-0.49],ddof=1)),
"MC (K09 FMNIST)":(-0.49,np.std([-0.76,-0.52,-0.18],ddof=1)),
"Ours-A vs minimal Ts10 FMNIST LT":(0.45,np.std([0.26,0.41,0.67],ddof=1)),
"Ours-A vs minimal Ts10 FMNIST IID":(-0.38,np.std([-0.75,-0.20,-0.18],ddof=1)),
"PCR vs Ours-A FMNIST LT (E29)":(-0.07,None),
"rows+MC+KH vs hybrid PCR+MC-S SVHN LT":(2.71,np.std([2.15,1.78,4.22],ddof=1))}
dz3=optimize.brentq(lambda d: power(d,3)-0.8,0.01,min(5,7/np.sqrt(3)))
dz5=optimize.brentq(lambda d: power(d,5)-0.8,0.01,min(5,7/np.sqrt(5)))
for k,(m,s) in cases.items():
    if s is None: continue
    print(f"{k}: mean {m:+.2f}, sd_d {s:.2f}, observed dz {abs(m)/s:.2f}, MDE(80%) n=3: {dz3*s:.2f} pts, n=5: {dz5*s:.2f} pts, power at observed effect n=3: {power(abs(m)/s,3):.2f}")
# K09-reported claim: 'Five seeds (10/10 negative -> p=0.002)'
print("sign test 10/10 two-sided:",2*0.5**10, " 5/5 per component per dataset:",2*0.5**5)
