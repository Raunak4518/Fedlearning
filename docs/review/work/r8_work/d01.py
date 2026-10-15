import json,statistics as st,glob
R='/home/user/Fedlearning/experiments/results/'
for src in ['D01_kaggle','D01_component_evidence']:
 for ds in ['mnist','fmnist','cifar10','svhn']:
  fs=sorted(glob.glob(R+f'{src}/{ds}_s*.json'))
  if not fs: continue
  agg={}
  for f in fs:
    d=json.load(open(f))
    for stg,m in d['metrics'].items():
      if 'tstr_recall' not in m and stg!='real (test)': continue
      a=agg.setdefault(stg,{'tstr':[],'tstr_tail':[],'mmd':[],'fid_tail':[],'spread_tail':[],'F':[]})
      if 'tstr_recall' in m:
        a['tstr'].append(st.mean(m['tstr_recall'])*100); a['tstr_tail'].append(st.mean(m['tstr_recall'][6:])*100)
      a['mmd'].append(st.mean(m['mmd'])); a['fid_tail'].append(st.mean(m['fid'][6:])*100); a['spread_tail'].append(st.mean(m['spread'][6:]))
      if 'fisher_F' in m: a['F'].append(m['fisher_F'])
  print('==',src,ds,len(fs), 'stage3 keys:',[k for k in d if k.startswith('stage3')])
  for stg,a in agg.items():
    print(f'  {stg:16s}', ' '.join(f'{k}={st.mean(v):.3f}' + (f'±{st.stdev(v):.2f}' if len(v)>1 else '') for k,v in a.items() if v), '| tstr per-seed',[round(x,1) for x in a['tstr']])
  for f in fs:
    d=json.load(open(f))
    print('   ',f.split('/')[-1],{k:round(v['final_bal']*100,2) for k,v in d.items() if k.startswith('stage3')})
