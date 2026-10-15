import csv,re,numpy as np
from k09 import holm,bh
rows=[r for r in csv.DictReader(open("/home/user/Fedlearning/docs/paper_assets/numbers.csv")) if r["p_vs_ref"] not in ("",None)]
p=np.array([float(r["p_vs_ref"]) for r in rows])
print("numbers.csv p-values:",len(p),"nominal<0.05:",(p<0.05).sum(),"Holm<0.05:",(holm(p)<0.05).sum(),"BH<0.05:",(bh(p)<0.05).sum())
print("tests by kind:",{k:sum(1 for r in rows if r['test']==k) for k in set(r['test'] for r in rows)})
txt=open("/home/user/Fedlearning/docs/paper_draft.md").read()
q=re.findall(r"p\s*(?:=|≈|≤|\\approx|\\le)\s*\$?\s*([0-9.]+(?:e-?\d+)?)",txt)
q=[float(x.rstrip('.')) for x in q if x.rstrip('.')]
q=np.array(q)
print("p-values quoted in paper_draft.md:",len(q),"nominal<0.05:",(q<0.05).sum(),"Holm<0.05 (family = all quoted):",(holm(q)<0.05).sum(),"BH<0.05:",(bh(q)<0.05).sum())
print(sorted(q))
# which survive Holm
h=holm(q); print("surviving Holm:",sorted(q[h<0.05]))
