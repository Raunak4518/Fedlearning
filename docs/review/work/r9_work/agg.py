import json,os,sys,collections,statistics as st
R='/home/user/Fedlearning/experiments/results'
M=['final_bal','final_tail','best_mean_acc','bbc_bal','oracle_bal','er_bal','final_ens_bal']
def agg(folder,fn='runs.jsonl'):
    g=collections.OrderedDict()
    for l in open(os.path.join(R,folder,fn)):
        if not l.strip(): continue
        r=json.loads(l)
        k=(r['label'],r['dataset'],r['IF'],r.get('alpha'),r['K'])
        g.setdefault(k,{})[r['seed']]=r
    return g
def fmt(v): return f"{v:.2f}"
if __name__=='__main__':
    for folder in sys.argv[1:]:
        print('=====',folder)
        for k,ss in agg(folder).items():
            line=f"{k[0][:38]:38s} {k[1]:7s} IF={k[2]} a={k[3]} K={k[4]} seeds={sorted(ss)}"
            for m in M:
                xs=[100*ss[s][m] for s in sorted(ss) if ss[s].get(m) is not None]
                if xs:
                    line+=f" | {m[:6]}={fmt(st.mean(xs))}" + (f"±{fmt(st.stdev(xs))}" if len(xs)>1 else '') + ("["+",".join(fmt(x) for x in xs)+"]" if m in('final_bal','best_mean_acc') else '')
            print(line)
