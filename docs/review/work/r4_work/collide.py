import json,glob,os
from collections import defaultdict
seen=defaultdict(list)
for f in sorted(glob.glob("/home/user/Fedlearning/experiments/results/*/runs.jsonl")):
    for i,l in enumerate(open(f)):
        r=json.loads(l)
        if "IF" not in r: continue
        k=(os.path.basename(os.path.dirname(f)),r["label"],r["dataset"],float(r["IF"]),int(r["K"]),int(r["seed"]))
        seen[k].append((r.get("alpha"),r.get("quick"),round(r["final_bal"]*100,2)))
n=0
for k,v in seen.items():
    if len(v)>1:
        n+=1
        if n<40: print(k,v)
print("collisions",n)
