# Kaggle notebooks: one per person

Give each person **one** notebook. Each notebook explains itself in its first cell.

## For each person (5 minutes)
1. Kaggle → **Create** → **New Notebook** → **File** → **Import Notebook** → upload your `.ipynb`.
2. Right panel → **Settings**: **Accelerator = GPU T4 x2**, **Internet = On**.
3. Top right: **Save Version** → **Save & Run All (Commit)** → **Save**. Then you can close the tab.
4. When the version shows a green tick: open it → **Output** → download the `.zip` → send it back.

If the run stops with a red **STOP** message, it says which setting to change. Change it and do step 3 again.

| Notebook | What it runs | Time | Send back |
|---|---|---|---|
| NB36_ablation_seeds_mcb_mnist.ipynb | MNIST long tail, seeds 3-4 only: the K09 ablation arms and MC-B (seeds 0-2 exist in K09 / E32). | ~2 h | `K14_kaggle_ablation_seeds_mcb__mnist.zip` |
| NB37_ablation_seeds_mcb_fmnist.ipynb | FashionMNIST long tail, seeds 3-4 only: the K09 ablation arms and MC-B (seeds 0-2 exist in K09 / E32). | ~2 h | `K14_kaggle_ablation_seeds_mcb__fmnist.zip` |
| NB38_collapse_fixes_mnist.ipynb | MNIST long tail, K = 10/50/100: AdamW (matched 1e-3; 1e-2 at K=10), no decay on class rows, lazy decay, against HWA (references reused). | ~3.5 h | `K15_kaggle_collapse_fixes__mnist.zip` |
| NB39_collapse_fixes_fmnist.ipynb | FashionMNIST long tail, K = 10/50/100: AdamW (matched 1e-3; 1e-2 at K=10), no decay on class rows, lazy decay, against HWA (references reused). | ~3.5 h | `K15_kaggle_collapse_fixes__fmnist.zip` |
| NB40_recalib_mnist_lt.ipynb | MNIST long tail: GeFL-F, +LA, minimal variant, our gated method; every trained head evaluated under 8 readouts (ER, Fed3R, FedNCM, CCVR, ...). | ~1.5 h | `K16_kaggle_recalibration__mnist_lt.zip` |
| NB41_recalib_fmnist_lt.ipynb | FashionMNIST long tail: the same four methods under every readout. | ~1.5 h | `K16_kaggle_recalibration__fmnist_lt.zip` |
| NB42_recalib_svhn_lt.ipynb | SVHN long tail: the same four methods under every readout (the gate picks the class-row generator here). | ~2 h | `K16_kaggle_recalibration__svhn_lt.zip` |
| NB43_recalib_cifar10_lt.ipynb | CIFAR-10 long tail: the same four methods under every readout. | ~4.5 h | `K16_kaggle_recalibration__cifar10_lt.zip` |
| NB44_recalib_cifar100_lt.ipynb | CIFAR-100 long tail (new dataset, 100 classes): the same four methods under every readout. | ~3.5 h | `K16_kaggle_recalibration__cifar100_lt.zip` |
| NB45_recalib_mnist_fmnist_iid.ipynb | MNIST + FashionMNIST, the paper's balanced setting: the same four methods under every readout. | ~3 h | `K16_kaggle_recalibration__mnist_fmnist_iid.zip` |
| NB46_recalib_svhn_iid.ipynb | SVHN, the paper's balanced setting: the same four methods under every readout. | ~2 h | `K16_kaggle_recalibration__svhn_iid.zip` |
| NB47_recalib_cifar10_iid.ipynb | CIFAR-10, the paper's balanced setting: the same four methods under every readout. | ~5.5 h | `K16_kaggle_recalibration__cifar10_iid.zip` |
| NB48_recalib_cifar100_iid.ipynb | CIFAR-100, balanced: the same four methods under every readout. | ~4 h | `K16_kaggle_recalibration__cifar100_iid.zip` |

Most important first: NB1 (CIFAR-10), then NB2, NB3, NB4. NB6–NB8 are long and can run later.
