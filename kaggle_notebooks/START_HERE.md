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

Most important first: NB1 (CIFAR-10), then NB2, NB3, NB4. NB6–NB8 are long and can run later.
