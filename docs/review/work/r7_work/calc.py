import math
MB=1e6
B4=4
gen={"mnist":(5546659,5937315,3159363),"cifar10":(5613294,6928622,4101450)}
D={"mnist":768,"cifar10":2560}
rounds={"mnist":(20,100,50),"cifar10":(50,200,100)}
headavg={"mnist":364980/10,"cifar10":376894/10}
body={"mnist":342000,"cifar10":341000}
# P per fake epoch (from pool.py) for K=10
P1={("mnist","iid"):576,("mnist","lt"):1173,("cifar10","iid"):2432,("cifar10","lt"):4309}
navg={("mnist","iid"):600,("mnist","lt"):600,("cifar10","iid"):2500,("cifar10","lt"):1848}
C=10
for ds in ["mnist","cifar10"]:
  d=D[ds]; TFE,TKA,TTN=rounds[ds]
  stats=C*(1+d+1+6144)
  print(ds,"stats numbers",stats, "MB",stats*4/MB)
  print("  StageI per client up+down MB", TFE*headavg[ds]*4*2/MB)
  g0,g1,gk=gen[ds]
  print("  StageII GeFL-F up MB/round",g0*4/MB,"total up+down GB",TKA*g0*4*2/1e9)
  print("  StageII PC-VAE up MB/round",g1*4/MB,"total up+down GB",TKA*g1*4*2/1e9, "delta GB", TKA*(g1-g0)*8/1e9)
  print("  KME gen download MB", gk*4/MB, "KME ratio GeFL-F gen up / stats", TKA*g0*4/(stats*4), "up+down", 2*TKA*g0/stats, "vs stats+gen download", 2*TKA*g0/(stats+gk))
  for reg in ["iid","lt"]:
    for ts in [1,10]:
      P=P1[(ds,reg)]*ts
      lab=P*C*4
      pool=P*d*4
      slice_=navg[(ds,reg)]*ts*d*4
      print(f"  CSL {reg} Ts={ts}: P={P} labels/round KB={lab/1e3:.0f} per Stage III MB={lab*TTN/MB:.1f}; full-pool features/round MB={pool/MB:.1f}; avg-client slice/round MB={slice_/MB:.1f} per stage GB={slice_*TTN/1e9:.2f}")
      # KH flops per round: chunks of 4096
      chunks=P/4096
      n=4096/C; cand=4*n
      rff=cand*d*6144*2*C; herd=n*cand*6144*2*C
      print(f"     KH GFLOP/round={(rff+herd)*chunks/1e9:.0f}  per stage TFLOP={(rff+herd)*chunks*TTN/1e12:.1f}  sequential argmax steps/stage={int(n*C*math.ceil(chunks)*TTN)}")
