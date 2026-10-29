# B: Literature fact-finding (FedECS-LT, manuscript v02, git 248170e)

Agent B (literature researcher). Read-only on the paper. Every statement below says where it was checked. "Read" means
the full text (arXiv HTML or PDF) was opened and the relevant section searched; "metadata only" means only the
bibliographic record (Crossref, Semantic Scholar, arXiv API, publisher page) was opened.

Plain-text copies of the sources are in
`/tmp/claude-0/-home-user-Fedlearning/76f672ce-438e-501b-a061-1360cd7e1c9d/scratchpad/lit/<key>.txt` (41 papers plus two
files of GeFL's code, `kang2025gefl_code_CCVAE.py.txt` and `kang2025gefl_code_setup.py.txt`). RUCR (IEEE TIFS) and FedRS
(KDD) are paywalled and have no arXiv version I could find, so they have no text copy.

Severity: P0 integrity, P1 major, P2 significant, P3 minor.

---

## 1. GeFL (Kang, Cha, Kang; arXiv:2412.18460v2, 17 May 2025; IEEE TMC)

Read: the full arXiv HTML (`lit/kang2025gefl.txt`), plus the authors' code at github.com/honggkang/hetero-model-fl-gen
(branch `master`, MIT licence): `GeFL_CVAE-F.py`, `args.py`, `generators16/CCVAE.py`, `utils/setup.py`, `utils/util.py`,
`utils/localUpdateTarget.py`.

**Problem.** Model-heterogeneous FL: clients cannot share architectures because of hardware (ASIC/FPGA) or proprietary
constraints, and no public data are available.

**Method.**
- *GeFL* (Alg. 1): (i) "generative knowledge aggregation": a conditional generator (FedDCGAN, FedCVAE or FedDDPM) is
  trained on images by FedAvg; (ii) "target network training": each round, each client trains its own target network
  for T_s local epochs on generated samples with labels y ~ p(y), then for T_r epochs on its real data. Target networks
  are averaged only among clients with the same architecture ("model-wise aggregation").
- *GeFL-F* (Alg. 3, §IV): every target network = a common feature extractor (one conv layer + BN + pooling) + a
  heterogeneous header. (i) warm-up of the extractor for T_FE rounds, extractor averaged over all clients, headers within
  an architecture; (ii) a feature generator (FedCVAE-F, FedDCGAN-F, FedDDPM-F) is trained on the extractor's features by
  FedAvg; (iii) headers trained on synthetic features then on real features, averaged within an architecture.
- *Claimed contribution:* a model-agnostic framework that shares knowledge through a federated generator; GeFL-F adds
  scalability to more clients, lower parameter and sampling cost, and lower memorisation (MND ratio).

**Datasets.** GeFL: MNIST, Fashion-MNIST, CIFAR-10 (Table II); GeFL-F: MNIST, Fashion-MNIST, SVHN, CIFAR-10 (Fig. 4).
Appendix D-B: CelebA (gender) for both (Table XXV). CIFAR-100 is used only as FedDF's public set for CIFAR-10, never as
a task.

**Clients and models.** Ten heterogeneous CNNs (different sets for MNIST, FMNIST/SVHN, CIFAR-10; Tables XXI-XXIII),
plus a ResNet/EfficientNet set in Appendix D-A. 10, 50 and 100 clients (Fig. 4: GeFL-F scalability); with 50 clients,
"10 model types are each assigned to 5 clients". Footnote 5: "the total dataset size is fixed and evenly divided across
clients". The code splits IID (`utils/setup.py`: `dict_users = dict_iid(dataset_train, int(1/args.partial_data*args.num_users))`;
the `--noniid` flag is never used there).

**Baselines (§III-B).** FedAvg, FedProx (mu = 1e-2), FedALA, LG-FedAvg, all "applied in grouped settings: clients are
partitioned by model architecture ... In the 10-client case, each client uses a unique model and trains independently
without FL. Throughout the paper, FedAvg, LG-FedAvg, FedProx, and FedALA follow these settings." AvgKD is *not* grouped:
"pseudo labels were aggregated from the outputs of 10 heterogeneous models" (App. B); footnote 3: AvgKD's value is "the
highest performance ... within a few initial rounds (approximately 30 rounds)". FedDF uses *public data*: SVHN for MNIST,
CIFAR-10 for FMNIST, CIFAR-100 for CIFAR-10 (§III-B and App. B).

**Metrics.** "mean classification accuracy (%) across 10 heterogeneous target networks throughout the paper"; Table II
gives 95% confidence intervals; footnote 2 claims significance over FedAvg. The paper does not say which round is
reported. The code reports, for each model, its best test accuracy over the evaluated rounds, then averages these
(`GeFL_CVAE-F.py` lines 25, 54, 122-129; `utils/util.py` `evaluate_models`, lines 105-106: `if acc_test > best_perf[i]: best_perf[i] = acc_test`).

**Settings (Tables XIV, XV).** Target networks: SGD lr 0.1; batch 64 (128 on CIFAR-10); data fraction 0.1 (MNIST,
FMNIST, SVHN) and 0.5 (CIFAR-10); T_w/T_g/T_s/T_r = 5/5/1/5; T_FE = 20 (MNIST, FMNIST) and 50 (SVHN, CIFAR-10).
T_TN in Table XIV: 50 / 50 / 100 (CIFAR-10) / 100 (SVHN), but §IV-B text says T_TN = 50 for MNIST, FMNIST **and SVHN**
(an inconsistency inside GeFL). T_KA = 100 (200 on CIFAR-10). FedCVAE-F: Adam, lr 1e-3, weight decay 1e-3, latent 16
(50 on CIFAR-10). Code: `torch.optim.Adam(gen_glob.parameters(), lr=1e-3, weight_decay=0.001)` (`GeFL_CVAE-F.py:60`), i.e. coupled L2.

**Generator details verified in code** (`generators16/CCVAE.py`): the decoder input is `cat(z, one_hot(y))` followed by
one `nn.Linear(latent+num_classes, ...)` (line 47), so column c of that weight is the only class-specific parameter used
for sampling; the encoder receives the label as a constant image channel (`ones(...)*y`, line 58), i.e. no class-specific
parameter. Sampling draws labels uniformly over **10 hard-coded classes** (lines 99-101: `torch.rand(...)*10`,
`np.zeros((y.shape[0], 10))`). GeFL's code therefore cannot run CIFAR-100 unchanged.

**Results on CIFAR-10.** Table II gives GeFL only (FedCVAE 57.70 +- 1.09; FedDCGAN 58.45; FedAvg 55.65). There is no
table of GeFL-F accuracies on MNIST/FMNIST/SVHN/CIFAR-10: those are only plotted in Fig. 4. The paper's "published"
GeFL-F CIFAR-10 value is read from that plot (`docs/papers/fill.py:121`, "GeFL paper, Fig. 4, 10 clients").

**Limitations stated.** §III-C: privacy leakage through generator memorisation, degradation with more clients, resource
cost of generators; DP "orthogonal ... future direction". §IV-B: "for the CIFAR10 dataset, GeFL-F does not perform as
effectively ... attributed to the lower performance of the feature extractor and target networks on CIFAR10". App. D-C:
non-image domains. No long tail, no label skew is evaluated.

### 1.1 Statements of the FedECS-LT paper about GeFL

| # | Location | Problem | Evidence | Why it matters | Sev. | Conf. | Action |
|---|---|---|---|---|---|---|---|
| G1 | sec6_setup.tex:171-172 | "FedDF ... As in GeFL, the distillation runs on generated features instead of public data." False: GeFL ran FedDF on public data. | GeFL §III-B: "We used the public dataset SVHN for MNIST, CIFAR10 for FMNIST, and CIFAR100 for the CIFAR10 dataset"; App. B repeats it. | Misattributes a design choice to the source paper; a reviewer who knows GeFL will spot it. | P1 | High | "Unlike GeFL, which distils on public data, we run FedDF's distillation on generated features, so that no method uses extra data." |
| G2 | sec6_setup.tex:177-179; table row sec6_setup.tex:225 "AvgKD (grouped)" | AvgKD said to "run within each architecture group, as GeFL compares them". GeFL groups only FedAvg, LG-FedAvg, FedProx, FedALA; AvgKD averages the predictions of all 10 heterogeneous models. | GeFL §III-B (quoted above) and App. B. The study's own code is cross-architecture: `fedecs_lt/src/grouped_client.py:116` "softmax(one-hot(y) + sum over the other architectures' models of their softmax)". | Text contradicts both GeFL and the implementation. | P2 | High | Move AvgKD out of "grouped": "FedAvg, FedProx and FedALA within each architecture group, and AvgKD across all architectures, as in GeFL". Also note GeFL reports AvgKD at its best early round (footnote 3). |
| G3 | sec6_setup.tex:101-102 | "made long-tailed as in GeFL-F's code". GeFL's code has no long-tail construction; it splits a fraction of the data IID. | `utils/setup.py` (`dict_iid`), `args.py` (only `--noniid`, `--dir_param`, `--partial_data`). | Credits GeFL for a protocol it does not contain. | P2 | High | "We take GeFL-F's data fraction (Table XIV) and divide it over the classes by the profile of Sec. 3 (our construction)." |
| G4 | sec6_setup.tex:219 (Venue "IEEE TMC 2025"); refs.bib `kang2025gefl` | Final publication: IEEE TMC vol. 25, no. 4, pp. 4493-4505, April 2026, DOI 10.1109/TMC.2025.3592483 (early access 2025). | Crossref record of DOI 10.1109/TMC.2025.3592483; Semantic Scholar (journal 25, 4493-4505). | Incomplete reference. | P3 | High | Add volume, number, pages, DOI; year 2026 (or "early access 2025"). |
| G5 | sec8_discussion.tex:33-35; appendix.tex:82 ("GeFL reports X%") | The published GeFL-F CIFAR-10 number is read off a plot (Fig. 4), not a table. | GeFL has no GeFL-F accuracy table; fill.py:121. | A plot reading carries reading error; readers will look for a table. | P3 | High | "GeFL's Fig. 4 shows about X%". |
| G6 | sec6_setup.tex:273-274 | "GeFL reports the mean over the ten models in the same way." GeFL reports mean top-1 accuracy with each model at its *best* round (code), not mean class recall at the last round. Same quantity only on balanced test sets. Line 293 discloses the best-round point. | GeFL §III-B; utils/util.py:105-106. | Minor; the paper already gives GeFL's measure in tab:app-gefl. | P3 | High | "GeFL also averages over the ten models (top-1 accuracy, each model's best round)". |
| G7 | appendix.tex:69-70 (SVHN 50/100/100) | Matches GeFL Table XIV (T_TN = 100 on SVHN) but contradicts GeFL §IV-B text (T_TN = 50 on SVHN). | GeFL Table XIV vs §IV-B. | Reproducibility; state the source. | P3 | High | "(GeFL's Table XIV; its text gives 50 for SVHN)". |
| G8 | appendix.tex:72-75 vs sec6_setup.tex:319-321 | Appendix: checked "against the original implementation ... within seed noise"; Sec. 6: checked "against our earlier, single-process implementation, which follows GeFL's code ... within the larger of two points and three seed standard deviations". "Original implementation" reads as GeFL's code. | Both passages. | Two different claims about the reproduction check; one suggests GeFL's code was run. | P2 | High | Use the Sec. 6 wording in the appendix. |
| G9 | sec1_intro.tex:16-17 "GeFL-F was evaluated on balanced clients" | Supported, but GeFL never writes "IID"; the evidence is footnote 5 ("evenly divided") and the code (`dict_iid`). | As above. | None if cited precisely. | P3 | High | Keep; cite footnote 5 / code. |
| G10 | sec3_setting.tex:11-16 (stages; Adam with coupled decay; conditioning row the only class-specific parameter; uniform synthetic labels) | Supported. | GeFL Alg. 3, Table XV; code lines quoted above. | — | OK | High | Optionally cite the code for "coupled" (Table XV only says "weight decay"). |
| G11 | sec8_discussion.tex:32-33 (GeFL-F less effective on CIFAR-10 because of extractor/target networks) | Supported (GeFL §IV-B, quoted above). | — | — | OK | High | — |
| G12 | sec6_setup.tex:143-144 (K = 50, five per architecture) | Supported (GeFL §III-B). GeFL-F's scalability figure also has 100 clients. | — | — | OK | High | — |

---

## 2. Closest work: per-paper notes

All rows: Read = full text opened unless marked. "Eval" = how results are reported.

| Paper (key) | Problem | Method | Protocol / baselines / eval | Limitations stated or evident | How FedECS-LT differs |
|---|---|---|---|---|---|
| FedLTLib (lin2026fedltlib), arXiv 2609.15625v1, 14 Sep 2026 | Benchmark for federated long-tail learning ("double heterogeneity") | Library (LongTailLib, github.com/gzhu-hcai/LongTailLib, README: MIT, "free for academic and non-commercial use"); 13 algorithms: FedAvg, FedProx, FedLC, MOON, CReFF, CLIP2FL, CCVR, RUCR, FedETF, FedLoGe, FedIC, FedGraB, FedYoYo | CIFAR-10/100-LT, Tiny-ImageNet; IF in {10, 50, 100}; Dirichlet alpha = 0.5; 20 clients; Table VI reports "the best global Top-1 accuracy"; Many/Medium/Few metrics. The paper text does not state backbone, participation, local epochs, batch size or optimiser. README default backbones: ResNet8 (CReFF, CLIP2FL, CCVR, RUCR, FedIC), ResNet20 (FedETF), ResNet18 (FedLoGe, FedGraB, FedYoYo); `-gr 200` | Best-round reporting; one architecture | FedECS-LT uses its splits and code, but forces ResNet-8 for all (`fedecs_lt/src/track2.py:273`, `"-m", "ResNet8"`) |
| HtFLlib (zhang2025htfllib), KDD 2025, pp. 5900-5911 | Benchmark for model-heterogeneous FL | 10 methods in 3 families: partial sharing (LG-FedAvg, FedGen, FedGH), mutual distillation (FML, FedKD, FedMRL), prototypes (FD, FedProto, FedTGP, FedKTL); repo lists also FedL2G; Apache-2.0 | 12 datasets; label skew (pathological, Dirichlet), feature shift, real-world; 20 clients; 1 local epoch, batch 10, lr 0.01, 1000 rounds, K = 512; each client's data split 3:1 into train/test and "average test accuracy across all clients' test sets"; "We report the best results" | Personalised objective; no long-tail global distribution | FedECS-LT scores HtFLlib methods on the global balanced test set after the last round, with 5 local epochs and a frozen extractor (a protocol change; see C5) |
| CCVR (luo2021ccvr), NeurIPS 2021 | Classifier bias under non-IID data | After FedAvg, clients upload per-class means and covariances of features; server forms the pooled Gaussian per class, samples virtual features, retrains the classifier | CIFAR-10/100, CINIC-10; Dir(0.5); with FedAvg/FedProx/FedAvgM/MOON | Gaussian assumption; per-client statistics revealed to the server | FedECS-LT's moment correction and readout use the same pooled first/second moments, via SecAgg, per architecture |
| CReFF (shang2022creff), IJCAI 2022, pp. 2218-2224 | Heterogeneous + long-tailed FL | Server learns "federated features" by matching clients' classifier gradients; retrains classifier on them | **CIFAR-10/100-LT (IF 100/50/10), ResNet-8, 20 clients, 40% active, batch 32, 200 rounds, SGD lr 0.1**; ImageNet-LT ResNet-50; Top-1 | Single model | This, not FedLTLib, is the origin of the one-shared-model protocol (see C4) |
| FedETF (li2023fedetf), ICCV 2023, pp. 5296-5306 | Classifier bias under non-IID | Fixed simplex-ETF classifier, projection layer, balanced feature loss; local fine-tuning for personalisation | CIFAR-10/100, Tiny-ImageNet; Dirichlet; SGD momentum 0.9, decay 0.99/round; seeds 7-10 | Not long-tail specific | — |
| FedLoGe (xiao2024fedloge), ICLR 2024 | Joint generic + personalised FL under long tail | Static sparse ETF classifier (SSE-C) for the backbone; global and local adaptive feature realignment heads | CIFAR-10-LT ResNet-18 / 40 clients; CIFAR-100-LT ResNet-34 / 20 clients; ImageNet-LT, iNat-160k; GM on balanced test, PM on local test | — | Run here with ResNet-8 and 20 clients |
| Fed-GraB (xiao2023fedgrab), NeurIPS 2023, pp. 77745-77757 | Federated long tail | Direct Prior Analyzer (global prior from classifier gradients) + self-adjusting gradient balancer per client | Same backbones/clients as FedLoGe | Privacy of DPA discussed | Run here with ResNet-8 |
| RUCR (huang2024rucr), IEEE TIFS 19, 5738-5750 | Federated long tail | Prototype-based representation unification + classifier rectification with balanced virtual features (from FedLTLib's and a search summary; **paper not read**, paywalled) | not read | not read | — |
| FedYoYo (yan2025fedyoyo), ICCV 2025, pp. 2750-2759 | Heterogeneous + long-tailed FL | Augmented self-bootstrap distillation + distribution-aware logit adjustment | Follows CReFF's setup; ResNet-8 on CIFAR-LT/SVHN-LT; **300 rounds**; Table 2, CIFAR-10-LT IF=100, alpha=0.5: FedAvg 58.22, CCVR 68.13, FedLC 69.02, FedETF 68.25, CReFF 70.51, Fed-GraB 70.63, FedLoGe 70.54, RUCR 55.32, FedYoYo 81.45 | Uses global distribution info; DP suggested | — |
| FedProto (tan2022fedproto), AAAI 2022 | Model + statistical heterogeneity | Exchange class prototypes (mean embeddings); regularise local embeddings to global prototypes; inference by nearest prototype | Personalised local test sets | Needs equal embedding dimension | FedECS-LT's class means are exact (SecAgg) and used to anchor a generator, not as a regulariser |
| FedTGP (zhang2024fedtgp), AAAI 2024, 38(15):16768-16776 | Same | Server-trainable global prototypes with adaptive-margin contrastive loss | 75/25 local split; "choose the best averaged result among iterations"; lr 0.01, batch 10, 1000 rounds | — | — |
| FedGen (zhu2021fedgen), ICML 2021, PMLR 139:12878-12889 | Heterogeneous FL, data-free KD | Server trains a label-conditional generator in latent space so that the *ensemble of user predictors* classifies its outputs; label counts p(y) from users; App.: per-class weights lambda_k^c proportional to n_k^c in the generator loss | — | Generator trained against predictors, not data | Precedent for class-count weighting (relevant to CCWA) |
| FedDF (lin2020feddf), NeurIPS 2020, pp. 2351-2363 | Robust model fusion | Server ensemble distillation on unlabeled or generated data | — | Needs proxy data | FedECS-LT's consensus soft label is ensemble distillation on generated features |
| AvgKD (afonin2022avgkd), ICLR 2022 | Model-agnostic FL via KD | Clients distil from the average of other clients' predictions on their own data, averaged with the true labels | Theory for kernel regression; experiments | Degenerates under heterogeneity (GeFL footnote 3) | Consensus soft label t = (1-omega) e_y + omega * mean softmax is the AvgKD target form, applied on generated features |
| FedALA (zhang2023fedala), AAAI 2023 | Personalised FL | Adaptive local aggregation: element-wise learned mix of global and local model | Personalised | Homogeneous models | Grouped baseline only |
| FedGH (yi2023fedgh), ACM MM 2023, pp. 8686-8696 | Model-heterogeneous FL | Server trains a global header on uploaded prototypes; replaces local headers | Local 8:1:1 splits | — | — |
| LG-FedAvg (liang2020lgfedavg), arXiv 2020 / NeurIPS-W 2019 | Local representations, global head | Local feature extractors, global top layers | Local and new-device test | — | — |
| FedMRL (yi2024fedmrl), NeurIPS 2024, pp. 66431-66454 | Model-heterogeneous FL | Shared small homogeneous model + Matryoshka representation fusion | Personalised | — | — |
| FedFTG (zhang2022fedftg), CVPR 2022, pp. 10164-10173 | Non-IID FL with **one global model** | Server generator in input space mines hard samples from local models; class-level ensemble; fine-tunes the global model | — | Homogeneous | Listed in sec2 as model-heterogeneous (C6) |
| FedVTC (niu2026fedvtc), ICLR 2026 (arXiv 2508.01669, v1 titled "Boosting Generalization Performance in Model-Heterogeneous FL Using Variational Transposed Convolution") | Generalisation of heterogeneous clients | Each client trains a VTC decoder (VAE-like) on its own data with a KL term pulling latents to the **global class prototype c^y = flat mean of holders' local prototypes**, shared sigma; generates *images* to fine-tune local models; clients exchange only means and sigma | MNIST, CIFAR-10/100, Tiny-ImageNet | Not long-tail; flat (not count-weighted) prototype mean; no SecAgg | Closest precedent for (b) mean-anchored generation |
| Fed3R (fani2024fed3r), ICML 2024, PMLR 235:13029-13048 | Fast HtFL (statistical heterogeneity) with pre-trained features | Closed-form ridge classifier from summed G = sum zz^T and B = sum z y^T; "mathematically equivalent to the corresponding centralized RR"; class-norm normalisation for imbalance; App. B: "a practical implementation might incorporate a Secure Aggregation step" | Landmarks, iNat-120k etc. | Pre-trained features; linear | (a): FedECS-LT's readout is the same kind of estimator, per architecture, on trained bodies |
| FedNCM (legate2023fedncm), NeurIPS 2023, pp. 69832-69848 | Last layer with pre-trained models | Nearest class means computed exactly in FL; init for fine-tuning | — | — | — |
| FedCOF (goswami2025fedcof, **not cited**), NeurIPS 2025, arXiv 2412.14326 | Training-free FL with pre-trained models | Unbiased estimate of class covariances from client class means; classifier W = G_hat^{-1} B with only **within-class** covariances, which it links explicitly to **LDA**; shrinkage gamma*I; "samples belonging to the same class in different clients are sampled from the same distribution. We exploit this fact"; discusses SecAgg (supported, at Fed3R's cost) | iNat-Users-120K, CUB, Cars, ImageNet-R; Dirichlet | iid-within-class assumption | Strongest prior for (a) and for the label-shift argument of Sec. 3 |
| AFL (he2025afl, **not cited**), CVPR 2025, arXiv 2405.16240 | Single-round FL with pre-trained models | Closed-form least-squares head; "Absolute Aggregation"; "invariance to data partitioning ... the aggregated result remains identical" | — | Linear head only | Prior statement of Prop. partition invariance |
| DFL-FS (chen2024dflfs, **not cited**), arXiv 2403.08364 | Long-tailed non-IID FL | Stage 1: client selection by clustering masked feature statistics; stage 2: server forms global feature statistics and "regenerate[s] samples from Gaussian distribution by resampling or weighted covariance to retrain the classifier" | read (PDF) | arXiv only | Prior for (c) |
| FedCG (wu2022fedcg, not cited), IJCAI 2022 | Privacy in FL | Private extractor, public classifier, clients share conditional GANs that generate features | — | Homogeneous classifier | Already discussed in GeFL §II-C; optional |
| BSM (ren2020bsm), LA (menon2021la), FedLC (zhang2022fedlc), FedRS (li2021fedrs, not read), FedRoD (chen2022fedrod) | Label prior correction | Additive log-prior (BSM, LA), pairwise margin tau*n^{-1/4} (FedLC; its printed Eq. sums the denominator over i != y), restricted softmax (FedRS), balanced-softmax generic head with client counts (FedRoD) | — | — | FedECS-LT's Prop. 2 applies to these per-client constants |
| FedProx (li2020fedprox), MLSys 2020, 2:429-450 | Heterogeneous networks | Proximal term | — | — | Grouped baseline |

---

## 3. Novelty threats (ranked)

| Rank | Threat | Overlaps with | Evidence | Sev. | Conf. | How to position |
|---|---|---|---|---|---|---|
| N1 | **FedCOF** (Goswami et al., NeurIPS 2025) and **Fed3R** (cited) | (a) exact readout: a federated LDA-type linear classifier from summed class statistics, with shrinkage; Sec. 3's "pure label shift" argument; Prop. partition invariance | FedCOF §4.3, Eq. 8 and the LDA remark; its "same class ... same distribution" sentence; Fed3R §3 "mathematically equivalent to the centralized RR solution" and App. B (SecAgg) | P1 | High | Cite FedCOF next to Fed3R/FedNCM (sec2:20-22, sec4:86-88, intro:45-47). State that the readout is the shared-covariance LDA plug-in, as in FedCOF's within-class variant, computed per architecture on trained (not pre-trained) bodies with class-balanced (equal-prior) biases, and that the novelty is pooling it *across architectures* (every client runs every body). Present Prop. partition invariance as a known property (Fed3R, AFL), not a contribution. |
| N2 | **AFL** (He et al., CVPR 2025) | Prop. partition invariance; closed-form head | AFL abstract and §3 ("invariance to data partitioning") | P2 | High | Cite with Fed3R in sec4 around Prop. invariance. |
| N3 | **CCVR** (cited), **DFL-FS** (not cited) | (c) moment correction: generating class features with exactly the pooled first and second moments | CCVR §4 (Gaussian with pooled mu_c, Sigma_c); DFL-FS stage 2 | P2 | High | Say that a Gaussian with the exact moments is CCVR's sampler; the moment correction is an affine map of a learned generator's samples onto the same moments (least displacement, App. Prop. mc). The ablation arm "GeFL-F with a Gaussian class-conditional generator from the exact statistics" (sec6_setup.tex:335) is a CCVR-style sampler; name and cite it as such. |
| N4 | **FedVTC** (cited) | (b) mean-anchored generator | FedVTC §3.2: latent prior centred on the global class prototype, shared sigma, decoder per client | P2 | High | Already cited; describe it correctly (C7): clients train the decoder locally with a KL to the server-averaged (flat) prototype, generating images. Difference: FedECS-LT anchors the *output* (h = ReLU(mu_y + dec(z, A mu_y))) at exact SecAgg means, with no class-specific parameter, trained by FedAvg. |
| N5 | **FedGen** class-count weights; **FedCPD** (He, Li, Seo, Cai, Information Fusion 110:102481, 2024, DOI 10.1016/j.inffus.2024.102481; *bibliographic record checked, content only from a search summary*) | CCWA: class rows averaged over holders, weighted by counts | FedGen App. "Weighting user models": lambda_k^c / lambda_j^c = n_k^c / n_j^c; FedCPD reportedly aggregates observed and missing class proxies separately | P2 | Med | Cite FedGen's count-weighted ensemble as precedent for count weighting; read FedCPD before citing. CCWA's specific choice (effective-number weights, holders only, on generator conditioning rows) appears new. |
| N6 | Consensus soft labels vs **AvgKD**/**FedDF** | Soft target = mix of one-hot label and averaged predictions of all architectures | AvgKD §3 ("we average the predictions with the original labels"); FedDF | P2 | High | In sec5:76-80, cite AvgKD and FedDF for the target form; the new part is computing it on generated features through a secure sum. menon2021distill alone is an indirect support. |
| N7 | Gate between generators | F = mean between-class scatter / mean within-class spread | Fisher's criterion (Fisher, Annals of Eugenics 7(2):179-188, 1936, DOI 10.1111/j.1469-1809.1936.tb02137.x, Crossref checked); FedCOF discusses between vs within scatter | P3 | Med | Call F a trace form of Fisher's discriminant ratio and cite Fisher; no FL precedent for a gate between generator types was found (two searches). |
| N8 | Diagnosis: coupled L2 under Adam shrinks rare classes' conditioning rows on non-holders | — | No FL paper found (searched "federated conditional generator label embedding collapse weight decay Adam"); Loshchilov & Hutter give the coupled/decoupled difference | — | Low (absence of evidence) | Keep as the contribution; say a search found no prior FL report. |

The dossier's earlier precedents (DP-MERF, FedRoD, FedVTC) are now cited (sec2:24, :37-38, :10-11). Nothing in v02
reintroduces the dossier's false claims about novelty.

---

## 4. refs.bib audit (75 entries)

Sources: arXiv API (export.arxiv.org/api/query?id_list=...), Crossref (api.crossref.org/works), Semantic Scholar
(api.semanticscholar.org, title match), NeurIPS/PMLR/MLSys/JMLR proceedings pages. No nonexistent paper was found.
Nine entries are never cited (fill.py included): collins2021fedrep, shuai2022balancefl, cao2019ldam, papyan2020nc,
demsar2006, kornblith2019cka, gretton2012mmd, burda2016iwae, mcinnes2018umap (BibTeX drops them; harmless).

| Key | Status | Correction / note | Source checked |
|---|---|---|---|
| mcmahan2017fedavg | OK | — | S2 record (DBLP conf/aistats/McMahanMRHA17, pp. 1273-1282) |
| li2020fedprox | OK | — | proceedings.mlsys.org 2020 page (vol 2, 429-450) |
| li2021moon | OK, pages differ | IEEE Xplore pages 10708-10717 (DOI 10.1109/CVPR46437.2021.01057); bib has 10713-10722 (probably CVF Open Access pagination, not checked) | Crossref |
| kairouz2021advances | OK | add DOI 10.1561/2200000083 | Crossref |
| beutel2020flower | OK | — | arXiv 2007.14390 |
| bonawitz2017secagg | OK | DOI 10.1145/3133956.3133982 | Crossref |
| bell2020secaggplus | OK | DOI 10.1145/3372297.3417885 | Crossref |
| hsu2019dirichlet | OK | — | arXiv 1909.06335 |
| zhang2025htfllib | OK | add pages 5900-5911, DOI 10.1145/3711896.3737379 | Crossref |
| lin2026fedltlib | OK | arXiv 2609.15625v1 (14 Sep 2026), Changkun Lin, Junxiao Wang | arXiv API |
| kang2025gefl | ERROR (incomplete) | IEEE TMC 25(4):4493-4505, 2026, DOI 10.1109/TMC.2025.3592483 | Crossref; S2 |
| tan2022fedproto | OK | 36(8), DOI 10.1609/aaai.v36i8.20819 | Crossref |
| zhang2024fedtgp | OK | add 38(15):16768-16776, DOI 10.1609/aaai.v38i15.29617 | Crossref |
| zhu2021fedgen | OK | add pages 12878-12889 | proceedings.mlr.press/v139/zhu21b.html |
| yi2023fedgh | OK | pages 8686-8696, DOI 10.1145/3581783.3611781 | Crossref |
| liang2020lgfedavg | OK | 8 authors as on arXiv v3 | arXiv 2001.01523 |
| yi2024fedmrl | OK | pages 66431-66454 | Crossref 10.52202/079017-2125 |
| lin2020feddf | OK | — | papers.nips.cc 2020 hash 18df51b9... (2351-2363) |
| li2019fedmd | OK | — | arXiv 1910.03581 |
| zhang2022fedftg | OK | pages 10164-10173, DOI 10.1109/CVPR52688.2022.00993 | Crossref |
| niu2026fedvtc | OK | ICLR 2026 poster iclr.cc/virtual/2026/poster/10006626; arXiv 2508.01669 (v1 under another title) | web search + arXiv |
| collins2021fedrep | OK (uncited) | pages 2089-2099 | PMLR v139/collins21a |
| luo2021ccvr | OK | — | papers.nips.cc 2021 (5972-5984) |
| shang2022creff | OK | pages 2218-2224, DOI 10.24963/ijcai.2022/308 | Crossref |
| li2023fedetf | OK | pages 5296-5306, DOI 10.1109/ICCV51070.2023.00490 | Crossref |
| xiao2023fedgrab | OK | pages 77745-77757 | Crossref 10.52202/075280-3396 |
| xiao2024fedloge | OK | — | arXiv 2401.08977 ("Accepted by ICLR 2024") |
| yan2025fedyoyo | OK | DOI 10.1109/ICCV51701.2025.00264 | Crossref |
| huang2024rucr | OK | add pages 5738-5750, DOI 10.1109/TIFS.2024.3402361 | Crossref |
| zhang2022fedlc | OK | pages 26311-26329 | proceedings.mlr.press/v162/zhang22p.html |
| li2021fedrs | OK | pages 995-1005, DOI 10.1145/3447548.3467254 | Crossref |
| chen2022fedrod | OK | — | arXiv 2107.00778 (ICLR 2022) |
| shuai2022balancefl | OK (uncited) | pages 271-284 | S2 (DOI 10.1109/IPSN54338.2022.00029) |
| fani2024fed3r | OK | pages 13029-13048 | S2; arXiv 2406.01116 |
| legate2023fedncm | OK | pages 69832-69848 | Crossref 10.52202/075280-3061 |
| yu2020fedaws | OK | pages 10946-10956 | PMLR v119/yu20f |
| ren2020bsm | OK | — | papers.nips.cc 2020 (4175-4186) |
| menon2021la | OK | — | arXiv 2007.07314 (ICLR 2021) |
| menon2021distill | OK | pages 7632-7642 | PMLR v139/menon21a |
| cui2019cb | OK, pages differ | IEEE 9260-9269 (DOI 10.1109/CVPR.2019.00949); bib 9268-9277 (CVF pagination, not checked) | Crossref |
| cao2019ldam | OK (uncited) | — | S2 |
| kang2020decoupling | OK | — | S2 (ICLR) |
| kingma2015adam | OK | — | S2/arXiv 1412.6980 |
| loshchilov2019adamw | OK | — | arXiv 1711.05101 comment "Published as a conference paper at ICLR 2019" |
| sohn2015cvae | OK | — | papers.nips.cc 2015 page |
| shahbazi2022collapse | OK | — | S2 (ICLR 2022) |
| ravuri2019cas | OK | — | S2 (NeurIPS 2019) |
| harder2021dpmerf | OK | pages 1819-1827 | S2 (conf/aistats/HarderAP21) |
| he2016resnet | OK | — | Crossref 10.1109/CVPR.2016.90 |
| fang2021minority | OK | — | Crossref 10.1073/pnas.2103091118 |
| papyan2020nc | OK (uncited) | — | Crossref |
| soudry2018implicit | OK | — | jmlr.org/papers/v19/18-188.html (5 authors incl. Nacson) |
| gelbrich1990 | OK | DOI 10.1002/mana.19901470121 | Crossref |
| guo2017calibration | OK | — | S2 (PMLR 70, 1321-1330) |
| kull2019classwise | OK | pages 12295-12305 | S2; papers.nips.cc 2019 |
| gorodkin2004mcc | OK | — | Crossref |
| holm1979 | OK | — | S2 (DOI 10.2307/4615733) |
| demsar2006 | OK (uncited) | — | S2 |
| kornblith2019cka | OK (uncited) | — | S2 |
| gretton2012mmd | OK (uncited) | — | S2 |
| burda2016iwae | OK (uncited) | — | S2 |
| vandermaaten2008tsne | OK | — | S2 |
| kobak2019tsne | OK | DOI 10.1038/s41467-019-13056-x | Crossref |
| wattenberg2016tsne | OK | volume 1, issue 10 | Crossref |
| mcinnes2018umap | OK (uncited) | — | arXiv 1802.03426 |
| lecun1998mnist | OK | — | Crossref 10.1109/5.726791 |
| xiao2017fmnist | OK | — | arXiv 1708.07747 |
| netzer2011svhn | OK | — | S2 |
| krizhevsky2009cifar | OK | — | S2 |
| yang2023medmnist | OK | DOI 10.1038/s41597-022-01721-8 | Crossref |
| stallkamp2012gtsrb | OK | — | Crossref 10.1016/j.neunet.2012.02.016 |
| cohen2017emnist | OK | — | Crossref 10.1109/IJCNN.2017.7966217 |
| goodfellow2013fer | OK | DOI 10.1007/978-3-642-42051-1_16 | Crossref |
| zhang2023fedala (new) | OK | matches 37(9):11237-11244, DOI 10.1609/aaai.v37i9.26330 | Crossref |
| afonin2022avgkd (new) | OK | arXiv 2110.15210 "Published at ICLR 2022" | arXiv API |

---

## 5. Citation-support mismatches (v02)

G1-G3, G8 above are the most important. Others:

| # | Location | Sentence | Problem | Evidence | Sev. | Conf. | Fix |
|---|---|---|---|---|---|---|---|
| C4 | sec6_setup.tex:146-154 | "We follow the FedLTLib protocol~\cite{lin2026fedltlib}: ... ResNet-8, 200 rounds of 10 local epochs with batch size 32 and SGD (learning rate 0.1, momentum 0.9, decay 0.99 per round). The baselines run the authors' LongTailLib training code unchanged" | The FedLTLib paper gives datasets, IF, alpha = 0.5 and 20 clients, but not the backbone, participation, local epochs, batch size or optimiser. ResNet-8 / 20 clients / 40% / batch 32 / 200 rounds / lr 0.1 are CReFF's setup (and FedYoYo follows CReFF). LongTailLib's README defaults to ResNet18 for FedLoGe, FedGraB, FedYoYo and ResNet20 for FedETF; the study forces ResNet8 (`fedecs_lt/src/track2.py:273`). | FedLTLib text; CReFF §5 "Experimental Setup"; LongTailLib README | P2 | High | Cite CReFF (and the LongTailLib code) for the training settings; say "unchanged training code, with the backbone set to ResNet-8 for every method (the library defaults to ResNet-18/20 for FedLoGe, Fed-GraB, FedYoYo, FedETF)". This also explains part of the Fed-GraB/FedLoGe gap to their own papers (ResNet-18/34, 40 clients). |
| C5 | sec2_related.tex:13-14 | "HtFLlib benchmarks these methods under label skew but not under a long-tailed global distribution." | HtFLlib includes LG-FedAvg, FedGen, FedGH, FedProto, FedTGP, FedMRL (and FML, FedKD, FD, FedKTL) but not FedMD, FedDF, FedFTG, FedVTC or GeFL. It evaluates on clients' local test splits and reports the best round. | HtFLlib §2.2, App. A.1 | P3 | High | "HtFLlib benchmarks six of these methods (...), with personalised evaluation on each client's test split ..." and state in Sec. 6 that they are scored here on the global balanced test set at the last round. |
| C6 | sec2_related.tex:8-10 | FedFTG listed as a model-heterogeneous method whose generator is trained "against the ensemble of client heads" | FedFTG fine-tunes one global model (homogeneous FL); its generator works in input space against the full local models. | FedFTG abstract/method | P3 | High | Move FedFTG to "generators for one shared model" or drop "heads" for it. |
| C7 | sec2_related.tex:10-11 | FedVTC "train[s] generators from feature-distribution statistics that the clients send to the server" | In FedVTC each client trains its own VTC on its own data; the server only averages prototypes (flat mean over holders) and sigma; the generator outputs images. | FedVTC §3.2, Alg. 1 | P3 | High | "each client trains a decoder whose latent prior is centred on server-averaged class prototypes". |
| C8 | sec2_related.tex:23-24 | "Loss corrections use each client's label frequencies (FedLC, FedRS, FedRoD)" | FedRS uses which classes are present (scales the logits of absent classes by alpha), not frequencies. (FedRS not read in full; its mechanism as described in this paper's own Prop. 2.) | sec4_analysis.tex:70 | P3 | Med | "label frequencies (FedLC, FedRoD) or the set of present classes (FedRS)". |
| C9 | sec2_related.tex:27-28 | "FedLTLib benchmarks them in one protocol" | FedLTLib contains FedLC, CCVR, CReFF, RUCR, FedETF, FedLoGe, Fed-GraB, FedYoYo (plus FedIC, CLIP2FL) but not FedRS, FedRoD or FedAwS. | FedLTLib §IV | P3 | High | "benchmarks most of them (...)". |
| C10 | sec2_related.tex:35-36 | "Class-conditional generators can collapse on classes with little data even centrally~\cite{shahbazi2022collapse}" | Shahbazi et al. study limited data overall (few images per class for all classes), not rare classes in an imbalanced set. | Shahbazi §3, Fig. 2 | P3 | High | "collapse when trained on little data, even centrally". |
| C11 | sec2_related.tex:36-37 | "synthetic training data help only as far as they carry their labels~\cite{ravuri2019cas}" | CAS shows classifiers trained only on generated data fall short and that CAS does not track FID/IS; "only as far as they carry their labels" is an interpretation. CAS also reports that low-diversity samples can help augmentation more than CAS predicts. | CAS §4-5 | P3 | Med | "classifiers trained on generated data reveal how much class information the samples carry". |
| C12 | sec4_analysis.tex:86-88 | CCVR listed with Fed3R/FedNCM as a function of a sum "when computed over all clients" | CCVR's server retrains the classifier by SGD on random virtual features: a *randomised* function of the pooled moments (deterministic only with a fixed seed); clients upload per-client statistics, not a secure sum. | CCVR §4 | P3 | High | "CCVR's virtual features, for a fixed sampling seed". |
| C13 | sec4_analysis.tex:81-88 (Prop. partition invariance) | Stated as the paper's own proposition without credit | Fed3R ("mathematically equivalent to the corresponding centralized RR solution") and AFL ("invariance to data partitioning") state it. | see N1, N2 | P2 | High | Add "as noted for Fed3R and AFL". |
| C14 | sec1_intro.tex:45-48; sec2_related.tex:20-22 | "closed-form federated classifiers~\cite{fani2024fed3r,legate2023fedncm}"; "our readout is a class-balanced linear discriminant of the same kind" | Misses the closest prior, FedCOF (LDA-type, within-class covariance, shrinkage). | FedCOF §4.3 | P1 | High | Cite FedCOF (Sec. 6 below). |
| C15 | sec5_method.tex:76-80 | consensus soft label "the federation's averaged prediction on that feature~\cite{menon2021distill}" | Menon et al. support soft labels as Bayes-probability estimates; the direct precedents for averaging heterogeneous models' predictions with the label are AvgKD and FedDF (both in refs.bib). | AvgKD §3; FedDF §3 | P3 | High | Add \cite{afonin2022avgkd,lin2020feddf}. |
| C16 | sec2_related.tex:42-67 (tab:positioning) | Lists the compared methods | v02 compares FedALA, AvgKD and grouped CCVR/FedETF (sec6_setup.tex:177-180) but the table omits FedALA and AvgKD; FedDF is marked "Synthetic --" although this study runs it on generated features. | sec6 vs sec2 table | P3 | High | Add the rows; footnote FedDF. |
| C17 | sec7_results.tex:180-186 | published-number comparison | Checked against FedYoYo Table 2 (CIFAR-10-LT, IF=100): FedAvg 58.22, CCVR 68.13, FedETF 68.25, FedYoYo 81.45, CReFF 70.51, Fed-GraB 70.63, FedLC 69.02: all quoted values correct. "FedLTLib reports the best round" is supported ("we extract the best global Top-1 accuracy"). "Higher values for every method": true against the v00 values of tab:shared (e.g. FedLTLib FedAvg 64.34 vs 61.0, FedYoYo 80.54 vs 78.1, Fed-GraB 68.82 vs 49.4); note FedLTLib is *lower* than FedYoYo's own paper for FedYoYo, CReFF, Fed-GraB and FedLC, so the sentence must say "than ours". | FedYoYo Table 2; FedLTLib Table VI | P3 | High | "higher values than ours for every method". Re-check against the filled v02 table. |
| C18 | sec6_setup.tex:194-195 | HtFLlib settings "SGD, learning rate 0.01, batch 10, feature dimension 512" | Supported (HtFLlib App. A.1: lr 0.01, batch 10, 1 local epoch, 1000 rounds; §3: K = 512). | — | OK | High | Optionally add "(HtFLlib: 1 local epoch, 1000 rounds)". |
| C19 | sec6_setup.tex:209-212 | licences: HtFLlib Apache-2.0; GeFL MIT | Supported (GitHub sidebar; LICENSE file "MIT License, Copyright (c) 2023 hongkang"). LongTailLib README: "MIT License ... Free for academic and non-commercial use" (mixed wording). | GitHub pages | OK / P3 | High | Optionally note LongTailLib's licence wording. |
| C20 | sec4_analysis.tex:65-71, sec1_intro.tex:29-31 | FedLC as an additive per-client constant | FedLC's calibrated logits f_i - tau*n_i^{-1/4} are a per-client constant shift, so the proposition applies; note that FedLC's printed loss (its Eq. for L_Cal) sums the denominator over i != y. | FedLC §4 | P3 | Med | State the form used ("cross-entropy on f - tau n^{-1/4}, as in the authors' code") — code not checked by me. |
| C21 | Other cites | sec2:5-8 (FedMD, FedDF, FedProto, FedTGP, LG-FedAvg, FedGH, FedMRL); sec2:29 (FedAwS); sec2:32-35, sec4:24 (LA, Cui, Kang, AdamW/Adam); sec4:48 (effective number); sec4:93 (minority collapse); sec4:109 (Soudry: unbounded logit growth on separable data); sec3:6 (Dirichlet, Hsu); sec3:30, intro:38 (SecAgg, SecAgg+); appendix:59 (Gelbrich); sec6:282-286 (MCC, ECE, class-wise ECE); sec7:276-286 (t-SNE refs); sec2:37-38 (DP-MERF) | Supported by the cited works as far as checked (abstracts and method sections; t-SNE/metrics refs metadata only). | — | OK | Med-High | — |

---

## 6. Recommended additions to related work (all verified as existing; read unless noted)

1. Goswami, Magistri, Wang, Twardowski, Bagdanov, van de Weijer. "Covariances for Free: Exploiting Mean Distributions
   for Training-free Federated Learning." NeurIPS 2025. arXiv:2412.14326. **Must cite** (N1, C14).
2. He, Tong, Fang, Sun, Zeng, Li, Chen, Zhuang. "AFL: A Single-Round Analytic Approach for Federated Learning with
   Pre-trained Models." CVPR 2025. arXiv:2405.16240. (N2, C13)
3. Chen, Wu, Ji. "Decoupled Federated Learning on Long-Tailed and Non-IID data with Feature Statistics." arXiv:2403.08364,
   2024 (no venue found). (N3)
4. Afonin, Karimireddy (already in refs.bib): cite at the consensus soft label (N6, C15).
5. Fisher. "The Use of Multiple Measurements in Taxonomic Problems." Annals of Eugenics 7(2):179-188, 1936,
   DOI 10.1111/j.1469-1809.1936.tb02137.x (Crossref) for the gate statistic and the LDA readout (N7).
6. Optional: Wu, Kang, Luo, He, Yang. "FedCG: Leverage Conditional GAN for Protecting Privacy and Maintaining Competitive
   Performance in Federated Learning." IJCAI 2022. arXiv:2111.08211 (conditional feature generators with a shared
   classifier; GeFL itself contrasts with it).
7. Optional, read before citing: He, Li, Seo, Cai. "FedCPD: Addressing label distribution skew in federated learning with
   class proxy decoupling and proxy regularization." Information Fusion 110:102481, 2024 (metadata only) for CCWA (N5).
8. Optional context (read, not threats): CLIP2FL (Shi et al., AAAI 2024, arXiv:2312.08648), which is in FedLTLib;
   FedPAC (Xu, Tong, Huang, ICLR 2023, arXiv:2306.11867), classifier collaboration with feature alignment;
   FedFed (Yang et al., NeurIPS 2023, arXiv:2310.05077); FedZKT (Zhang, Wu, Yuan, ICDCS 2022, arXiv:2109.03775), a
   server generator for heterogeneous on-device models.
