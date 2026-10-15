import json,sys,os,collections
R='/home/user/Fedlearning/experiments/results'
def load(folder,fn='runs.jsonl'):
    p=os.path.join(R,folder,fn); out=[]
    for i,l in enumerate(open(p)):
        l=l.strip()
        if not l: continue
        try: out.append(json.loads(l))
        except Exception as e: print('BAD line',folder,i,e)
    return out
if __name__=='__main__':
    for folder in sys.argv[1:]:
        rs=load(folder)
        print('=====',folder,len(rs))
        cnt=collections.Counter((r.get('label'),r.get('dataset'),r.get('IF'),r.get('alpha'),r.get('K'),r.get('seed')) for r in rs)
        d=[k for k,v in cnt.items() if v>1]
        if d: print('DUPLICATE keys:',len(d), d[:5])
        for r in rs:
            f=lambda k: (None if r.get(k) is None else round(100*r[k],2) if isinstance(r.get(k),float) else r.get(k))
            print(f"{str(r.get('label'))[:45]:45s} {r.get('dataset'):7s} IF={r.get('IF')} a={r.get('alpha')} K={r.get('K')} s={r.get('seed')} q={r.get('quick')} bal={f('final_bal')} tail={f('final_tail')} bma={f('best_mean_acc')} bbc={f('bbc_bal')} orc={f('oracle_bal')} er={f('er_bal')}")
