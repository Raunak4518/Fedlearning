import json,statistics as st,math
from scipy import stats
R='/home/user/Fedlearning/experiments/results/'
for ds in ['mnist','fmnist']:
  rs=[json.loads(l) for l in open(R+f'K09_kaggle_ablation_final__{ds}/runs.jsonl')]
  d={(r['label'],r['seed']):r for r in rs}
  labs=sorted(set(r['label'] for r in rs))
  print('==',ds)
  for m in ['final_bal','bbc_bal','oracle_bal','final_ens_bal','final_tail','final_worst']:
    for l in labs:
      if l.startswith('Ours'):continue
      diffs=[(d[(l,s)][m]-d[('Ours-A Ts=10',s)][m])*100 for s in range(3)]
      mu=st.mean(diffs); sd=st.stdev(diffs)
      t=stats.ttest_1samp(diffs,0).pvalue
      print(f'{m:14s} {l:16s} {mu:+6.2f} sd {sd:4.2f} p {t:.3f}', [round(x,2) for x in diffs])
  # cross: -LA with BBC vs Ours-A no BBC
  x=[(d[('-LA',s)]['bbc_bal']-d[('Ours-A Ts=10',s)]['final_bal'])*100 for s in range(3)]
  print('-LA+BBC minus Ours-A(noBBC)',[round(v,2) for v in x],round(st.mean(x),2))
  x=[(d[('-PC (CVAE+HWA)',s)]['oracle_bal']-d[('-PC (CVAE+HWA)',s)]['final_bal'])*100 for s in range(3)]
  y=[(d[('Ours-A Ts=10',s)]['oracle_bal']-d[('Ours-A Ts=10',s)]['final_bal'])*100 for s in range(3)]
  print('oracle-final gap: -PC',round(st.mean(x),2),' Ours',round(st.mean(y),2))
