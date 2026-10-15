import json, glob, os
from collections import defaultdict
RES="/home/user/Fedlearning/experiments/results"
def load():
    runs=defaultdict(dict)
    for f in sorted(glob.glob(RES+"/*/runs.jsonl")):
        folder=os.path.basename(os.path.dirname(f))+("" if f.endswith("/runs.jsonl") else ":"+os.path.basename(f))
        for line in open(f):
            r=json.loads(line)
            if "IF" not in r: continue
            runs[(folder,r["label"],r["dataset"],float(r["IF"]),int(r["K"]))][int(r["seed"])]=r
    return runs
