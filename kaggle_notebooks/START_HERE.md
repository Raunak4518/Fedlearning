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
| NB50_fair_mnist.ipynb | MNIST long tail, new seeds: our method (both synthetic budgets) vs GeFL-F and the published FL losses (FedLC, FedRS, balanced softmax). | ~2 h | `K17_kaggle_fair_retest__mnist.zip` |
| NB51_fair_fmnist.ipynb | FashionMNIST long tail, new seeds: the same comparison. | ~2 h | `K17_kaggle_fair_retest__fmnist.zip` |
| NB52_fair_svhn.ipynb | SVHN long tail, new seeds: the same comparison. | ~2.5 h | `K17_kaggle_fair_retest__svhn.zip` |
| NB53_fair_cifar10.ipynb | CIFAR-10 long tail, new seeds: the same comparison. | ~7.5 h | `K17_kaggle_fair_retest__cifar10.zip` |
| NB54_fair_cifar100.ipynb | CIFAR-100 long tail, new seeds: the same comparison. | ~5.5 h | `K17_kaggle_fair_retest__cifar100.zip` |
| NB55_fair_cifar10_rows.ipynb | CIFAR-10 long tail, new seeds: the class-row generator branch forced, to test the automatic switch near its threshold. | ~3 h | `K17_kaggle_fair_retest__cifar10_rows.zip` |

Most important first: NB1 (CIFAR-10), then NB2, NB3, NB4. NB6–NB8 are long and can run later.
