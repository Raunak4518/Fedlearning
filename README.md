# Class-Balanced GeFL

A research-grade, dataset-agnostic implementation of **GeFL** (Generative
Model-Aided Federated Learning; Kang et al., 2025) extended with two
proposed mechanisms for class-imbalance-aware model-heterogeneous
federated learning, plus a full **CReFF** (Shang et al., IJCAI 2022)
baseline for comparison.

Structured and named in the style of the original GeFL reference
implementation ([honggkang/hetero-model-fl-gen](https://github.com/honggkang/hetero-model-fl-gen))
— a single `args.py`, one top-level script per generator variant
(`GeFL_CVAE.py` / `GeFL_GAN.py` / `GeFL_DDPM.py`), a `utils/` package of
`LocalUpdate*` / `avg.py` / `evaluate.py` helpers — but rebuilt around a
registry pattern (`datasets/`, `generators/`, `targetNetModels/`) so that
**every dataset, generator architecture, and target-network architecture
is selected by name from the command line, with nothing hardcoded
anywhere in the training loop.**

## What's proposed here, on top of GeFL

GeFL's own reference behaviour is class-blind in two places:
1. **Aggregation**: the shared generator is averaged across clients with
   a flat, unweighted mean — a client holding zero samples of a class
   influences that class's conditioning parameters exactly as much as a
   client holding hundreds.
2. **Sampling**: synthetic conditioning labels are drawn `y ~ Uniform`,
   with no correction for true class rarity.

- **Mechanism A** (`--mechanism_a 1`) — frequency-weighted aggregation:
  the generator's class-conditioning pathway is aggregated with a
  per-class effective-number-of-samples weight (Cui et al., 2019,
  repurposed as an aggregation weight rather than a loss weight — see
  the note below), while the unconditional trunk stays on ordinary
  volume-weighted FedAvg.
- **Mechanism B** (`--mechanism_b 1`) — fidelity-gated adaptive sampling:
  `p(y)` drifts from the natural class frequency toward inverse-frequency
  rebalancing only as fast as each class's generated samples earn a
  measured target-network confidence — reusing, for free, a forward pass
  local training already runs.

Both are implemented against a generic interface
(`ConditionalGenerator.conditioning_parameter_names()`) so they work
identically for the VAE, GAN, and DDPM generator variants without any
architecture-specific code in the aggregation or sampling logic itself.

## Install

```bash
pip install -r requirements.txt
```

Everything runs on CPU (slow) or CUDA (`--device cuda`, auto-detected by
default). No GPU is required to run the tests or a `--dataset synthetic`
sanity check.

## Quick start

```bash
# 1. Fast, fully-offline sanity check (~5s, exercises every code path)
python GeFL_CVAE.py --config configs/synthetic_debug.yaml

# 2. GeFL baseline on CIFAR-10-LT (downloads CIFAR-10 via torchvision)
python GeFL_CVAE.py --config configs/cifar10_lt.yaml

# 3. Proposed method (both mechanisms) on the same setup
python GeFL_CVAE.py --config configs/cifar10_lt_proposed.yaml

# 4. Plain FedAvg, no generator at all
python GeFL_CVAE.py --dataset cifar10 --aid_by_gen 0 --name fedavg_baseline

# 5. Classifier-side comparison baseline (CReFF)
python Baseline_CReFF.py --config configs/cifar10_lt.yaml --name creff_cifar10lt

# 6. Same experiment with a GAN or DDPM generator instead of a VAE
python GeFL_GAN.py  --config configs/cifar10_lt.yaml
python GeFL_DDPM.py --config configs/cifar10_lt.yaml --n_T 200

# 7. Any registered dataset, just by changing --dataset -- nothing else
#    in the codebase needs to change
python GeFL_CVAE.py --dataset cifar100 --config configs/cifar10_lt.yaml
python GeFL_CVAE.py --dataset svhn     --img_size 32
python GeFL_CVAE.py --dataset mnist    --num_users 20

# 8. Sweep imbalance_factor x dirichlet_alpha x seed x mechanism
#    (the proposal's Week 6-8 milestone)
python scripts/sweep.py --config configs/cifar10_lt.yaml \
    --imbalance_factors 0.1 0.05 0.01 --dir_params 1.0 0.3 0.1 --seeds 0 1 2

# 9. Run the test suite
pytest tests/ -v
```

A `--config some.yaml` file supplies new argparse *defaults*; any CLI
flag after it still overrides the YAML, so
`python GeFL_CVAE.py --config configs/cifar10_lt.yaml --seed 3` uses
every value in the YAML except seed.

## Everything is dataset-agnostic — nothing is hardcoded

`--dataset {cifar10, cifar100, mnist, fmnist, svhn, stl10, synthetic}`
selects a dataset purely by name from `datasets.vision.DATASET_REGISTRY`
(see `datasets/vision.py`). Every downstream module — the long-tail
partitioner, both mechanisms, all three generators, all five target
networks, and the evaluator — is written against `meta.num_classes`,
`meta.in_channels`, and `meta.native_img_size` (or `--img_size`), never
against a specific dataset's dimensions. Concretely:

- **Partitioning** (`sampling/partition.py`) operates on a plain
  `labels: np.ndarray` + `num_classes: int` — no dataset-specific code.
- **Generators** (`generators/ccvae.py`, `ccgan.py`, `cddpm.py`) build
  their own encoder/decoder/UNet stage list at construction time from
  whatever `img_size` is passed in, by halving resolution until ≤4px
  (verified in `tests/test_generators.py` to work correctly at 16px,
  20px, and 28px in the same test run without any per-dataset branch).
- **Target networks** (`targetNetModels/nets.py`) use adaptive pooling so
  they never need to know resolution in advance.
- Adding a new dataset is exactly one function in `datasets/vision.py`
  with an `@DATASET_REGISTRY.register("name")` decorator — nothing
  elsewhere changes. Same pattern for a new generator
  (`generators/base.py` + `@GEN_REGISTRY.register(...)`) or a new target
  architecture (`@NET_REGISTRY.register(...)` in `targetNetModels/nets.py`).

If a real dataset can't be downloaded (e.g. no network access), `get_dataset()`
automatically falls back to `--dataset synthetic` with a `RuntimeWarning`
so the pipeline still runs — this is how the entire test suite and every
`configs/synthetic_debug.yaml` example above run fully offline.

## Project layout

```
args.py                     single argparse config (+ optional --config YAML), nothing hardcoded elsewhere
registry.py                 generic name -> class registry shared by datasets/generators/target nets
datasets/
  vision.py                   cifar10/100, mnist, fmnist, svhn, stl10 (torchvision-backed), registered
  synthetic.py                 offline procedurally-generated fallback, registered under the same interface
  get_dataset.py                unified entry point + automatic synthetic fallback on download failure
sampling/
  partition.py                 long-tail class-count schedule + Dirichlet / IID client partitioner
generators/
  base.py                       ConditionalGenerator interface (conditioning_parameter_names(), sample())
  ccvae.py                      Conditional VAE -- resolution/channel-agnostic
  ccgan.py                      Conditional DCGAN (G + D) -- resolution/channel-agnostic
  cddpm.py                      Conditional DDPM (real diffusion process + UNet-lite) -- resolution-agnostic
targetNetModels/
  nets.py                       5 architecturally distinct target nets (cnn_small/deep, mobilenet_lite,
                                 resnet_lite, mlp_mixer_lite), all channel/class/resolution-agnostic
utils/
  setup.py                       ties dataset + partition + registries together (setup_experiment())
  user_sampling.py                partial-participation client sampling (--frac)
  localUpdateGen.py                local generator training per gen_model (VAE/GAN/DDPM), LOCAL_GEN_UPDATE_REGISTRY
  localUpdateTarget.py             local target-net training, optional synthetic augmentation
  avg.py                           FedAvg, model_wise_FedAvg, Mechanism A (frequency_weighted_row_average)
  label_sampler.py                 GeFL-baseline uniform sampler + Mechanism B (FidelityGatedSampler)
  evaluate.py                      bucketed accuracy, centralized upper bound, gap analysis, MND privacy metric
  checkpoint.py / logger.py / seed.py
baselines/
  creff.py                       full CReFF classifier-side baseline (shared backbone, dataset-agnostic)
engine.py                    the federated training loop itself (shared by every GeFL_*.py script)
GeFL_CVAE.py / GeFL_GAN.py / GeFL_DDPM.py   thin per-generator entry points (mirrors original repo's naming)
Baseline_CReFF.py            entry point for the CReFF baseline
configs/                     YAML experiment presets (synthetic_debug, cifar10_lt, cifar100_lt, cifar10_lt_proposed)
scripts/sweep.py             grid sweep over imbalance_factor x dirichlet_alpha x seed x mechanism
tests/                       pytest unit tests for partitioning, aggregation, sampling, and generators
```

## The MND privacy metric

`utils/evaluate.py::mnd_ratio` implements GeFL's own privacy metric
(Kang et al. 2025, §IV-C): the mean nearest-neighbor distance ratio
between synthetic samples and the real training set versus a held-out
validation set, in flattened pixel space. A ratio near 1 means synthetic
samples are no more suspiciously close to training data than to unseen
data; a ratio well below 1 signals the generator may be memorizing
individual training samples. It's computed automatically at the end of
every `GeFL_*.py` run when `--aid_by_gen 1`.

## Known bug fixed during development

The first draft of the Mechanism A weight in `utils/avg.py` used Cui et
al. (2019)'s **loss**-reweighting formula, `(1-β)/(1-β^n)` — which
*decreases* as a client's class count `n` grows — directly as an
*aggregation* weight. That's backwards for aggregation: it would hand the
most influence over a class's conditioning row to the client with the
*fewest* samples of that class. The fix uses the **effective number
itself**, `(1-β^n)/(1-β)`, which *increases* with `n` (with diminishing
returns) — so the client that actually holds more of a class dominates
that class's row, matching the direction FedAvg already weights by data
volume. `tests/test_avg.py::test_effective_num_weight_increases_with_count`
is a standing regression test for this.

## Scope / honest limitations

- **GeFL-F** (the feature-level variant that shares a feature extractor
  instead of raw-image synthesis) is not implemented — this project
  targets GeFL's raw generative-augmentation setting, which is what the
  research proposal extends.
- **FedProx / AvgKD** and other baselines present in the original repo's
  broader baseline zoo are out of scope; the comparison set here is
  GeFL-baseline / Mechanism A / Mechanism B / proposed (A+B) / CReFF,
  matching this project's own evaluation plan.
- The DDPM variant is real (a proper linear beta schedule, closed-form
  forward process, UNet-lite epsilon-predictor, classifier-free guidance)
  but intentionally small — the proposal itself flags diffusion sampling
  cost as a stretch goal, not a required deliverable.
- This codebase was developed and tested in a sandbox with no network
  access to torchvision's dataset mirrors, so `configs/synthetic_debug.yaml`
  (not real CIFAR) is what's actually been run end-to-end here. Every
  module was written and unit-tested to be dataset-agnostic by
  construction (see `tests/`), and `--dataset cifar10` etc. will work
  as soon as it's run somewhere with internet access — but that specific
  combination (real CIFAR-10-LT weights over many rounds) has not itself
  been executed by the author of this code.
