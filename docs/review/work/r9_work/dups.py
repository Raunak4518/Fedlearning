import json,os,glob,collections
R='/home/user/Fedlearning/experiments/results'
for jl in sorted(glob.glob(R+'/*/runs*.jsonl')):
    rs=[json.loads(l) for l in open(jl) if l.strip()]
    k1=collections.Counter((r['label'],r['dataset'],float(r['IF']),int(r['K']),int(r['seed'])) for r in rs)
    k2=collections.Counter((r['label'],r['dataset'],float(r['IF']),r.get('alpha'),int(r['K']),int(r['seed'])) for r in rs)
    d1=sum(v-1 for v in k1.values() if v>1); d2=sum(v-1 for v in k2.values() if v>1)
    alphas=set((float(r['IF']),r.get('alpha')) for r in rs)
    quick=set(r.get('quick') for r in rs)
    print(os.path.relpath(jl,R), len(rs), 'dup(no alpha)=',d1,'dup(with alpha)=',d2, 'IF/alpha=',sorted(alphas,key=str), 'quick=',quick)
