import numpy as np, gzip, sys
def labels(p):
    with open(p,'rb') as f: d=f.read()
    return np.frombuffer(d,dtype=np.uint8,offset=8).astype(int)
def make_partition(y, K, frac, IF, alpha, seed, min_size=10, C=10):
    rng = np.random.RandomState(seed)
    n_total = int(round(frac * len(y)))
    mu = IF ** (1.0 / (C - 1)); prof = mu ** np.arange(C)
    want = np.floor(n_total * prof / prof.sum()).astype(int)
    by_class = [rng.permutation(np.where(y == c)[0]) for c in range(C)]
    pools = [bc[200:200 + min(want[c], len(bc) - 200)] for c, bc in enumerate(by_class)]
    for _ in range(100):
        parts = [[] for _ in range(K)]
        for c in range(C):
            idx = pools[c]; prop = rng.dirichlet([alpha] * K)
            cuts = (np.cumsum(prop) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)): parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size: break
    while min(len(p) for p in parts) < min_size:
        small = int(np.argmin([len(p) for p in parts])); big = int(np.argmax([len(p) for p in parts]))
        j = rng.randint(len(parts[big])); parts[small].append(parts[big].pop(j))
    return np.array([[int((y[p] == c).sum()) for c in range(C)] for p in parts])
root='/home/user/Fedlearning/experiments/data/'
for ds,p in [('mnist',root+'MNIST/raw/train-labels-idx1-ubyte'),('fmnist',root+'FashionMNIST/raw/train-labels-idx1-ubyte')]:
    y=labels(p)
    for K in (10,100):
      for s in (0,1,2):
        cnt=make_partition(y,K,0.1,0.01,0.5,s)
        tot=cnt.sum(0)
        print(f"{ds} K={K} seed={s} N={tot.sum()} class totals={tot.tolist()}")
        for c in (6,7,8,9):
            h=cnt[:,c]; nz=np.sort(h[h>0])[::-1]
            print(f"   class {c}: total {h.sum()}, holders {len(nz)}, holders with >=1: {nz.tolist()[:12]}, max share {nz[0]/h.sum():.2f}")
