# GeFL / GeFL-F: Alignment with Kang et al. 2025

Comprehensive plan to bring the codebase to paper-exact spec (Part 1) and build GeFL-F from scratch (Part 2), organized by the suggested implementation order (impact ÷ effort).

---

## User Review Required

> [!IMPORTANT]
> **§1.4 Conditioning mechanism — Keep embedding or go paper-exact?**
> The paper uses raw one-hot conditioning (no learned embedding). But Mechanism A's `conditioning_parameter_names()` interface depends on having a learnable embedding matrix whose rows correspond to classes. Switching to paper-exact one-hot means Mechanism A has no natural target to reweight.
>
> **Recommendation**: Keep `nn.Embedding` as a *documented intentional deviation* — it's what makes Mechanism A well-defined. Flag it in the config/README as "differs from paper for Mechanism A compatibility."

> [!IMPORTANT]
> **§1.5 GAN loss — Keep `BCEWithLogits` or match paper's `Sigmoid+BCELoss`?**
> Paper's Table XVII-b ends with explicit sigmoid. Your code uses the numerically stable `BCEWithLogits`. These conflict.
>
> **Recommendation**: Keep `BCEWithLogits` and document as a deliberate stability improvement over the paper.

> [!IMPORTANT]
> **§1.6 Target network pool — Replace with 10-CNN family or keep current 5-family pool?**
> Paper uses 10 homogeneous CNNs (same family, different depth/width). Your pool uses 5 architecturally diverse families (CNN/MobileNet/ResNet/MLP-Mixer), a harder heterogeneity test.
>
> **Recommendation**: Keep your current pool as the primary config. Add the paper's 10-CNN family as an *optional ablation* config (`configs/cifar10_paper_cnns.yaml`). This lets you both reproduce paper numbers AND run your harder heterogeneity test.

> [!IMPORTANT]
> **§1.9 Data fraction — Apply 0.5 subsample?**
> Paper uses `data_fraction=0.5` before any imbalance. Your setup applies imbalance to the full dataset.
>
> **Recommendation**: Add `--data_fraction` arg (default 1.0). For paper-matching configs set to 0.5. Your main experiments can keep 1.0.

## Open Questions

> [!IMPORTANT]
> **DDPM-F architecture**: The paper reduces layers/channels for DDPM-F but doesn't give an exact table. Should I implement a reasonable reduction (e.g., halve `n_feat` and remove one down/up stage from UNet-lite) and document it, or wait until you can verify against the PDF?

> [!NOTE]
> **GeFL-F feature extractor channel count**: The paper says "a single conv → bn → pool" but doesn't specify the output channel count `F` or kernel size. I'll default to `F=32, kernel=3, stride=1, pad=1` with `MaxPool2d(2,2)` — a reasonable minimal choice. If you have a preference, let me know.

---

## Proposed Changes

### Phase 1: Config & Hyperparameter Fixes (Low effort, high impact)

#### [MODIFY] [args.py](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/args.py)

**§1.1 + §1.2 + §1.7 + §1.9**: Add new arguments and update defaults.

New/changed arguments:

| Arg | Default | Purpose |
|---|---|---|
| `--gen_wu_epochs` | `100` ← was `5` | Paper: T_KA/2 = 100 for CIFAR10 |
| `--epochs` | `100` (unchanged) | Paper: T_TN = 100 |
| `--lr` | `0.1` ← was `0.01` | Paper Table XIV: α = 0.1 |
| `--local_bs` | `128` ← was `64` | Paper Table XIV: batch = 128 |
| `--latent_size` | `50` ← was `32` | Paper: CVAE l = 50 |
| `--n_feat` | `128` ← was `64` | Paper: DDPM n_feat = 128 |
| `--n_T` | `400` ← was `200` | Paper: DDPM timesteps = 400 |
| `--guide_w` | `0.0` ← was `0.3` | Paper: guidance w = 0 or 2 |
| `--target_ts` | `1` (NEW) | Synthetic-only local epochs (§1.2) |
| `--target_tr` | `5` (NEW) | Real-only local epochs (§1.2) |
| `--data_fraction` | `1.0` (NEW) | §1.9 subsample before partitioning |
| `--gen_lr_gan` | `2e-4` (NEW) | Paper: DCGAN β = 2e-4 |
| `--gen_lr_ddpm` | `1e-4` (NEW) | Paper: DDPM β = 1e-4 |
| `--weight_decay_ddpm` | `1e-3` (NEW) | Paper: DDPM weight decay |
| `--dcgan_g_channels` | `256` (NEW) | Paper: d_g = 256 |
| `--dcgan_d_channels` | `64` (NEW) | Paper: d_d = 64 |

Remove: `--local_ep`, `--synth_batch` (replaced by `--target_ts`/`--target_tr` sequential design).

---

#### [MODIFY] [cifar10_lt.yaml](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/configs/cifar10_lt.yaml)

Update to paper-matched values:

```diff
-lr: 0.01
-local_bs: 64
-gen_wu_epochs: 10
-epochs: 100
-local_ep: 2
-gen_local_ep: 2
-synth_batch: 64
-latent_size: 64
+lr: 0.1
+local_bs: 128
+gen_wu_epochs: 100
+epochs: 100
+target_ts: 1
+target_tr: 5
+gen_local_ep: 5
+latent_size: 50
+data_fraction: 0.5
```

#### [MODIFY] [cifar10_lt_proposed.yaml](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/configs/cifar10_lt_proposed.yaml)

Mirror the same hyperparameter fixes as `cifar10_lt.yaml`.

---

### Phase 2: Flat Aggregation for Target Nets (§1.3)

#### [MODIFY] [avg.py](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/utils/avg.py)

Change `model_wise_FedAvg` to always use flat `FedAvg` (not `weighted_FedAvg`) for the paper-matching baseline. The function already falls through to `FedAvg` when `sample_counts` is None or mismatched — the change is to **never** use `weighted_FedAvg` for target nets, since Algorithm 2 defines unweighted averaging for both generators and target nets.

```diff
 def model_wise_FedAvg(ws_glob, ws_local, sample_counts=None):
     new_glob = []
     for m, group in enumerate(ws_local):
         if len(group) == 0:
             new_glob.append(ws_glob[m])
-        elif sample_counts is not None and len(sample_counts[m]) == len(group):
-            new_glob.append(weighted_FedAvg(group, sample_counts[m]))
         else:
             new_glob.append(FedAvg(group))
     return new_glob
```

---

### Phase 3: Run Tracking and Isolation

The current codebase writes multiple artifacts (checkpoints, logs, CSVs, JSONL files, results pickles) keyed primarily by `args.name` into shared directories (`args.out_dir` and `args.ckpt_dir`). This causes collisions where sequential runs with the same `--name` overwrite each other's outputs.

## Proposed Changes

We will refactor the system so that every invocation creates a unique run directory, all artifacts for that invocation go inside it, and a global registry tracks all runs.

### 1. Unique Run ID Generation
Create `utils/run_id.py` with `generate_run_id(name)` and Git context retrieval.
- It will return a string formatted as `{name}_{YYYYMMDD_HHMMSS}_{uuid[:6]}`.
- It will safely run `git rev-parse HEAD` and `git status --porcelain`.

### 2. Update Run Entry Points
- `engine.py` (`run_gefl`): Call `generate_run_id`, create `logs/runs/<run_id>`, write to the index, and pass the run directory down to loggers and checkpointers.
- `gefl_f/engine_f.py` (`run_gefl_f`): Same updates as above.
- `Baseline_CReFF.py` (`main`): Same updates as above.
- Update `GeFL_*.py` and `Baseline_CReFF.py` to save `args.json` inside the run directory instead of the top-level directory.

### 3. Update Artifact Consumers/Writers
- `utils/logger.py` (`ExperimentLogger`): Update to take `run_dir` instead of `args.out_dir`. It will place `.log`, `.csv`, `.jsonl`, and `tensorboard/` inside `run_dir`.
- `utils/checkpoint.py` (`save_checkpoint`, `ckpt_path`): Update to save checkpoints under `run_dir/checkpoint/latest.pt`.
- `utils/visualize.py`: Ensure plots are saved to `run_dir/plots/`.
- `scripts/sweep.py`: Update to save sweep plots into the designated directory safely.

### 4. Create Run Tracker (`logs/runs/index.csv`)
- Create `utils/tracker.py` to manage the append-only `logs/runs/index.csv`.
- Append "running" status on start.
- Append "completed" or "failed" status on exit (using `try...finally`).

### 5. Latest Pointer
- Write `logs/runs/latest.txt` on every run start containing the active `run_id`.

### 6. The `--resume` Flag
- Add `--resume` to `args.py` (defaults to empty string).
- In `engine.py` and `engine_f.py`, check if `--resume` is set.
- If set, resolve the run directory and load `checkpoint/latest.pt`.

## Verification Plan
1. Run `python GeFL_CVAE.py --config configs/synthetic_debug.yaml` twice back-to-back.
2. Verify `logs/runs/` contains two unique directories with full sets of artifacts.
3. Verify `logs/runs/index.csv` contains four entries (two starts, two completions).
4. Diff `args.json` / timestamps across the two runs to prove no collision occurred.

```python
class LocalUpdate:
    def train(self, net, gennet=None, label_sampler=None):
        net.train()
        opt = self._make_optimizer(net)
        total_loss, n_batches = 0.0, 0
        fidelity_feedback = {}

        # Phase 1: Synthetic-only (Ts epochs)
        if args.aid_by_gen and gennet is not None:
            for _ in range(args.target_ts):
                for x_real, _ in self.dataloader:  # iterate to match batch count
                    syn_y = label_sampler.sample(args.local_bs).to(device)
                    with torch.no_grad():
                        syn_x = gennet.sample(syn_y)
                    opt.zero_grad()
                    loss = F.cross_entropy(net(syn_x), syn_y)
                    loss.backward()
                    opt.step()
                    # ... collect fidelity feedback here ...

        # Phase 2: Real-only (Tr epochs)
        for _ in range(args.target_tr):
            for x, y in self.dataloader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(net(x), y)
                loss.backward()
                opt.step()
```

---

### Phase 4: MND Privacy Metric Correction (§1.10)

#### [MODIFY] [evaluate.py](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/utils/evaluate.py)

Three corrections to `mnd_ratio`:

1. **Invert query direction**: iterate over **real training samples** and find nearest synthetic/validation, not synthetic→train/val
2. **LPIPS distance** instead of Euclidean (add `lpips` to requirements)
3. **Fix interpretation**: ρ < 1 is good (no memorization), ρ > 1 is concerning

```diff
-def mnd_ratio(synthetic_samples, train_samples, val_samples, max_ref=2000):
+def mnd_ratio(synthetic_samples, train_samples, val_samples,
+              n_query=1000, n_ref=600, use_lpips=True):
     """
-    Query: synthetic → (train, val). Ratio = d(syn,train)/d(syn,val)
-    Near 1 = safe, well below 1 = memorization concern.
+    Paper's MND (Eq. 1): for each real training sample x_i,
+    ρ_i = min_{x∈V} d(x_i,x) / min_{x∈S} d(x_i,x)
+    ρ > 1 = potential memorization, ρ < 1 = good generalization.
+    Distance = LPIPS (perceptual), |S|=|V|=600, averaged over 1000 queries.
     """
```

#### [MODIFY] [engine.py](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/engine.py)

Update the MND call site and interpretation logging:

```diff
-logger.console.info("MND privacy ratio: %.4f (near 1 = no detectable memorization signal)",
-                     results["mnd_ratio"])
+logger.console.info("MND privacy ratio: %.4f (< 1 = good generalization, > 1 = potential memorization)",
+                     results["mnd_ratio"])
```

#### [MODIFY] [requirements.txt](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/requirements.txt)

Add `lpips>=0.1.4`.

---

### Phase 5: Generator-Specific LR and Weight Decay (§1.7)

#### [MODIFY] [localUpdateGen.py](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/utils/localUpdateGen.py)

- `local_update_gan`: use `args.gen_lr_gan` (default 2e-4) instead of `args.gen_lr`
- `local_update_ddpm`: use `args.gen_lr_ddpm` (default 1e-4) and `args.weight_decay_ddpm` (default 1e-3)

---

### Phase 6: DCGAN Separate G/D Channel Widths (§1.7)

#### [MODIFY] [ccgan.py](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/generators/ccgan.py)

Use `args.dcgan_g_channels` (256) for the Generator and `args.dcgan_d_channels` (64) for the Discriminator instead of a single `args.gen_channels`.

---

### Phase 7: Data Fraction Subsample (§1.9)

#### [MODIFY] [partition.py](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/sampling/partition.py)

In `partition_dataset`, apply a random `data_fraction` subsample **before** `apply_long_tail`:

```python
def partition_dataset(labels, num_classes, args):
    # §1.9: subsample to data_fraction FIRST
    if getattr(args, 'data_fraction', 1.0) < 1.0:
        rng = np.random.RandomState(args.partition_seed)
        n = len(labels)
        keep_n = max(1, int(n * args.data_fraction))
        keep_idx = rng.choice(n, keep_n, replace=False)
        labels = labels[keep_idx]
        # ... remap indices ...
    # then apply_long_tail as before
```

---

### Phase 8: GeFL-F Implementation (Part 2 — largest effort)

#### [NEW] [gefl_f/](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/gefl_f/) — New package

```
gefl_f/
├── __init__.py
├── feature_extractor.py    # §2.2 — single conv→bn→pool block
├── feature_generators.py   # §2.4 — DCGAN-F, CVAE-F, DDPM-F architectures
├── engine_f.py             # §2.3 — four-stage training loop (Algorithm 3)
└── headers.py              # §2.1 — heterogeneous headers (existing nets minus shared stem)
```

#### [NEW] [feature_extractor.py](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/gefl_f/feature_extractor.py)

§2.2: One `Conv2d → BatchNorm2d → ReLU → MaxPool2d` block.

```python
class CommonFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel=3, stride=1, pad=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
    def forward(self, x):
        return self.net(x)
```

#### [NEW] [feature_generators.py](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/gefl_f/feature_generators.py)

§2.4: Feature-space generators (DCGAN-F, CVAE-F, DDPM-F). Key differences from full generators:
- One fewer upsampling/downsampling stage
- Final activation is **ReLU** (not tanh) — features aren't bounded to [-1,1]
- Input/output channels match the feature extractor's output channels

#### [NEW] [engine_f.py](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/gefl_f/engine_f.py)

§2.3: Three-stage loop (Algorithm 3):

```
Stage (i):  T_FE=50 rounds — joint FE+header training, aggregate both
Stage (ii): T_KA=200 rounds — freeze FE, train feature-generator on features
Stage (iii): T_TN=100 rounds — freeze FE, sequential Ts/Tr header training
```

Critical: `θ_f` frozen after stage (i). Only headers get gradients in stage (iii).

#### [NEW] [headers.py](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/gefl_f/headers.py)

Heterogeneous header networks — the existing target nets minus the shared stem. Each header takes the feature extractor's output as input and produces class logits.

#### [NEW] [GeFL_F_CVAE.py](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/GeFL_F_CVAE.py) (+ GAN/DDPM variants)

Top-level entry scripts for GeFL-F, analogous to `GeFL_CVAE.py`.

#### [NEW] [configs/cifar10_gefl_f.yaml](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/configs/cifar10_gefl_f.yaml)

GeFL-F specific config with `T_FE=50, T_KA=200, T_TN=100`.

---

### Phase 9: Paper-Matching 10-CNN Ablation (§1.6 — optional)

#### [MODIFY] [nets.py](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/targetNetModels/nets.py)

Add `PaperCNN` — a configurable CNN with the paper's shared stem (`conv(3,3×3)→bn→relu→conv(10,3×3)→bn→relu→maxpool`) and variable depth/width post-stem stages. Register 10 variants (`paper_cnn_1` through `paper_cnn_10`).

#### [NEW] [configs/cifar10_paper_cnns.yaml](file:///d:/federated%20learing/gefl_classbalanced_production/gefl_classbalanced/configs/cifar10_paper_cnns.yaml)

Config using `num_models: 10`, `target_models: paper_cnn_1,...,paper_cnn_10`.

---

## Summary of Kept Deviations (documented, intentional)

| Item | Paper | This codebase | Rationale |
|---|---|---|---|
| Conditioning | One-hot spatial broadcast | `nn.Embedding` | Required for Mechanism A's per-class row reweighting |
| GAN loss | `Sigmoid + BCELoss` | `BCEWithLogits` | Numerical stability; functionally equivalent |
| Target net pool (primary) | 10 homogeneous CNNs | 5 diverse families | Harder heterogeneity test; paper's pool available as ablation |

---

## Verification Plan

### Automated Tests

```bash
# Existing tests should still pass
pytest tests/ -v

# Quick smoke test with synthetic data + new sequential training
python GeFL_CVAE.py --dataset synthetic --num_users 3 --gen_wu_epochs 3 --epochs 5 \
    --target_ts 1 --target_tr 2 --imbalance_factor 0.1 --name smoke_test

# Verify MND metric direction
python -c "from utils.evaluate import mnd_ratio; ..."
```

### Manual Verification

- **§1.1**: Confirm warm-up runs for 100 rounds before target training starts (check logs)
- **§1.2**: Confirm training loops are sequential (synthetic-only then real-only) by inspecting per-round loss logs
- **§1.3**: Confirm all aggregation uses flat averaging (add assertion or log message)
- **§1.7**: Run a 5-round CIFAR10 test to verify LR=0.1 doesn't diverge with batch size 128
- **§1.10**: Verify MND ratio < 1 for FedCVAE on CIFAR10 (matching paper's 0.502)
- **Part 2**: Run GeFL-F 3-stage loop on synthetic data, verify FE freezes after stage (i)
