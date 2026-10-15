import json,statistics as st
from scipy import stats
exec(open('pair.py').read().split('for ds in')[0])
E='E16_csl_synthetic_budget'
for ds in ['mnist','fmnist']:
  for m in ['best_mean_acc','final_bal']:
    a=get(E,ds,1.0,'GeFL-F',m); b=get(E,ds,1.0,'GeFL-F Ts=5',m); c=get(E,ds,1.0,'+CSL',m); d=get(E,ds,1.0,'+CSL Ts=5',m)
    inter=[(d[s]-c[s])-(b[s]-a[s]) for s in range(3)]
    print(ds,m,'IID Ts1->5 hard',round(st.mean([b[s]-a[s] for s in range(3)]),2),'CSL',round(st.mean([d[s]-c[s] for s in range(3)]),2),'interaction',[round(x,2) for x in inter],'p',round(stats.ttest_1samp(inter,0).pvalue,3))
  cmp((E,ds,0.01,'Ours (HWA+LA+CSL)'),(E,ds,0.01,'Ours (HWA+LA+CSL) Ts=10'),'final_bal','LT minimal Ts1->10')
  cmp((E,ds,0.01,'Ours (HWA+LA+CSL)'),(E,ds,0.01,'Ours (HWA+LA+CSL) Ts=10'),'oracle_bal','LT minimal Ts1->10')
for ds,f in [('cifar10','K08_kaggle_anchored_stack__cifar10_lt'),('svhn','K08_kaggle_anchored_stack__svhn')]:
  cmp((f,ds,0.01,'Ours-A (PC+MC+KH+LA+CSL)'),(f,ds,0.01,'Ours-A Ts=10'),'final_bal','Ts=10 on Ours-A '+ds)
  cmp((f,ds,0.01,'Ours-A (PC+MC+KH+LA+CSL)'),(f,ds,0.01,'Ours-A Ts=10'),'bbc_bal','Ts=10 on Ours-A '+ds)
cmp(('E22_pc_many_clients','mnist',0.01,'Ours-A (PC+MC+KH+LA+CSL)',100),('E22_pc_many_clients','mnist',0.01,'Ours-A Ts=10',100),'final_bal','Ts=10 K=100')
cmp(('E22_pc_many_clients','mnist',0.01,'Ours-PC+MC',100),('E22_pc_many_clients','mnist',0.01,'Ours-A (PC+MC+KH+LA+CSL)',100),'final_bal','KH at K=100 (paired same exp)')
cmp(('E22_pc_many_clients','mnist',0.01,'Ours-PC+MC',100),('E22_pc_many_clients','mnist',0.01,'Ours-A (PC+MC+KH+LA+CSL)',100),'oracle_bal','KH at K=100 (paired same exp)')
