import json,sys,glob,os,collections,numpy as np
root="/home/user/Fedlearning/experiments/results"
fields=["final_bal","bbc_bal","oracle_bal","er_bal","best_mean_acc","final_tail","final_ens_bal","class_sep_F","bbc_applied"]
for f in sorted(glob.glob(root+"/*/runs.jsonl")):
    folder=os.path.basename(os.path.dirname(f))
    if len(sys.argv)>1 and not any(a in folder for a in sys.argv[1:]): continue
    g=collections.defaultdict(list)
    for line in open(f):
        try: r=json.loads(line)
        except: continue
        g[(r["label"],r["dataset"],r["IF"],r["alpha"],r["K"])].append(r)
    print("==",folder)
    for k,rs in sorted(g.items(),key=lambda x:str(x[0])):
        out=[]
        for fl in fields:
            v=[r.get(fl) for r in rs if r.get(fl) is not None]
            if v:
                if isinstance(v[0],bool): out.append(f"{fl}={sum(v)}/{len(v)}")
                else: out.append(f"{fl}={100*np.mean(v) if fl!='class_sep_F' else np.mean(v):.2f}")
        print(f"  {k[0]:40s} {k[1]:7s} IF{k[2]} a{k[3]} K{k[4]} n={len(rs)} seeds={[r['seed'] for r in rs]} "+" ".join(out))
