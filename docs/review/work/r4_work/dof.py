import json,glob,os
from collections import defaultdict
labels=set(); labels_full=set(); n=0; nq=0; settings=set(); folders=set()
for f in sorted(glob.glob("/home/user/Fedlearning/experiments/results/*/runs.jsonl")):
    fo=os.path.basename(os.path.dirname(f))
    for l in open(f):
        r=json.loads(l)
        if "IF" not in r: continue
        n+=1; folders.add(fo)
        labels.add(r["label"])
        if "_quick" not in fo and not r.get("quick"): labels_full.add(r["label"])
        else: nq+=1
        settings.add((r["dataset"],r["IF"],r.get("alpha"),r["K"]))
print("runs in runs.jsonl:",n,"quick:",nq,"folders:",len(folders))
print("distinct labels (all):",len(labels),"full-schedule:",len(labels_full))
print("distinct settings:",len(settings))
print(sorted(labels_full))
