from load import load
runs=load()
for k in sorted(runs):
    s=runs[k]
    fb={sd:round(100*r.get("final_bal",float('nan')),2) for sd,r in sorted(s.items())}
    bm={sd:round(100*r.get("best_mean_acc",float('nan')),2) for sd,r in sorted(s.items())}
    print(k, "bal",fb,"bma",bm, "quick" if any(r.get("quick") for r in s.values()) else "")
