import json,sys,statistics as st
R='/home/user/Fedlearning/experiments/results/'
def g(r,k):
  v=r.get(k); 
  return None if v is None else (v*100 if isinstance(v,float) and abs(v)<=1.0001 and k not in('class_sep_F','spread_tail_kh','spread_tail') else v)
keys=sys.argv[2].split(',')
for e in sys.argv[1].split(','):
  rs=[json.loads(l) for l in open(R+e+'/runs.jsonl')]
  groups={}
  for r in rs: groups.setdefault((r['dataset'],r.get('IF'),r.get('K'),r['label']),[]).append(r)
  print('=====',e)
  for k,v in sorted(groups.items(),key=str):
    out=[]
    for kk in keys:
      vals=[g(r,kk) for r in sorted(v,key=lambda r:r['seed']) if g(r,kk) is not None]
      if vals: out.append(f"{kk}={st.mean(vals):.2f}"+(f"±{st.stdev(vals):.2f}" if len(vals)>1 else '')+f"{[round(x,2) for x in vals]}")
    print(k, 'seeds',sorted(r['seed'] for r in v)); print('     ',' | '.join(out))
