import json,statistics as st,sys
from scipy import stats
R='/home/user/Fedlearning/experiments/results/'
def load(e):
  return [json.loads(l) for l in open(R+e+'/runs.jsonl')]
def get(e,ds,IF,label,metric='final_bal',K=10):
  return {r['seed']:r[metric]*100 for r in load(e) if r['dataset']==ds and r.get('IF')==IF and r['label']==label and r.get('K')==K and r.get(metric) is not None}
def cmp(a,b,metric='final_bal',name=''):
  A=get(*a,metric=metric); B=get(*b,metric=metric)
  s=sorted(set(A)&set(B))
  d=[B[i]-A[i] for i in s]
  if not d: return
  p=stats.ttest_1samp(d,0).pvalue if len(d)>1 else float('nan')
  print(f'{name:45s} {metric:10s} n={len(d)} diff={st.mean(d):+.2f} sd={st.stdev(d) if len(d)>1 else 0:.2f} p={p:.3f} per-seed={[round(x,2) for x in d]}')
for ds in ['mnist','fmnist']:
  print('==',ds)
  E='E20_proto_generator'
  for m in ['final_bal','oracle_bal','bbc_bal']:
    cmp((E,ds,0.01,'Ours (HWA+LA+CSL)'),(E,ds,0.01,'Ours-PC (PC+LA+CSL)'),m,'PC replaces HWA (E20)')
    cmp((E,ds,0.01,'Ours (HWA+LA+CSL)'),(E,ds,0.01,'Ours+MC'),m,'MC on CVAE+HWA')
    cmp((E,ds,0.01,'Ours-PC (PC+LA+CSL)'),(E,ds,0.01,'Ours-PC+MC'),m,'MC on PC')
    cmp((E,ds,0.01,'Ours+MC'),(E,ds,0.01,'Ours-PC+MC'),m,'PC vs HWA, both +MC')
    cmp((E,ds,0.01,'Ours-PC+MC'),('E25_anchored_stack',ds,0.01,'Ours-A (PC+MC+KH+LA+CSL)'),m,'KH (E20->E25 cross-exp)')
    cmp(('E25_anchored_stack',ds,0.01,'Ours-A (PC+MC+KH+LA+CSL)'),('E25_anchored_stack',ds,0.01,'Ours-A Ts=10'),m,'Ts=10 on Ours-A')
    cmp(('E25_anchored_stack',ds,0.01,'Ours-A Ts=10'),('E32_moment_anchor_bayes',ds,0.01,'Ours-A-B Ts=10'),m,'MC-B vs MC (E25->E32)')
    cmp((E,ds,0.01,'GeFL-F'),('E25_anchored_stack',ds,0.01,'Ours-A Ts=10'),m,'Ours-A vs GeFL-F')
    cmp((E,ds,0.01,'Ours (HWA+LA+CSL)'),('E25_anchored_stack',ds,0.01,'Ours-A Ts=10'),m,'Ours-A vs minimal')
    cmp(('E16_csl_synthetic_budget',ds,0.01,'Ours (HWA+LA+CSL) Ts=10'),('E25_anchored_stack',ds,0.01,'Ours-A Ts=10'),m,'Ours-A vs minimal Ts=10 (E16)')
  # GeFL-F oracle vs Ours-A final
  A=get(E,ds,0.01,'GeFL-F','oracle_bal'); B=get('E25_anchored_stack',ds,0.01,'Ours-A Ts=10','final_bal')
  d=[B[s]-A[s] for s in range(3)]; print('Ours-A final - GeFL-F oracle',[round(x,2) for x in d],round(st.mean(d),2),'p',round(stats.ttest_1samp(d,0).pvalue,3))
  A=get(E,ds,0.01,'Ours (HWA+LA+CSL)','oracle_bal')
  d=[B[s]-A[s] for s in range(3)]; print('Ours-A final - minimal oracle',[round(x,2) for x in d],round(st.mean(d),2),'p',round(stats.ttest_1samp(d,0).pvalue,3))
  A=get('E16_csl_synthetic_budget',ds,0.01,'Ours (HWA+LA+CSL) Ts=10','oracle_bal')
  if A: d=[B[s]-A[s] for s in range(3)]; print('Ours-A final - minimal Ts10 oracle',[round(x,2) for x in d],round(st.mean(d),2))
