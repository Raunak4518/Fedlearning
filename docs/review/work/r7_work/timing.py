import json,glob,collections,sys
import numpy as np
pat=sys.argv[1:] or ["*"]
for p in pat:
  for f in sorted(glob.glob(p+"/runs.jsonl")):
    rows=[json.loads(l) for l in open(f) if l.strip()]
    if not rows or rows[0].get("quick"): continue
    g=collections.defaultdict(list)
    for r in rows: g[(r["label"],r["dataset"],r["IF"],r["alpha"],r["K"])].append(r)
    print("==",f)
    for k,v in g.items():
        m=lambda key: np.mean([x.get(key) or np.nan for x in v])
        print("  %-48s %-7s IF=%-5s a=%-4s K=%-3s n=%d t=%6.0f fe=%5.0f gen=%6.0f head=%6.0f gpuMB=%5.0f bal=%.2f ts=%s" % (k[0][:48],k[1],k[2],k[3],k[4],len(v),m("time_s"),m("fe_time_s"),m("gen_time_s"),m("head_time_s"),m("gpu_peak_mb"),100*m("final_bal"), v[0]["method"]["head"].get("ts") if v[0].get("method") and v[0]["method"].get("head") else "-"))
