# Night plan, 9 Oct: closing the last gaps to the best GeFL numbers

## Where we stand (paper's metric `best_mean_acc`, IID, K = 10)

| Dataset | Best of any GeFL variant in the paper | Ours | Gap |
|---|---|---|---|
| MNIST | 96.44 (GeFL + image DDPM) | 96.77 | **+0.33** |
| FMNIST | 84.28 (GeFL-F + feature DDPM) | 83.21 | −1.07 |
| SVHN | 76.26 (GeFL-F CVAE-F) | 76.30 (+CSL) | +0.04 |
| CIFAR-10 | 59.36 (GeFL + DDPM); 62.67 with MixUp | ? (GeFL-F CVAE-F: 55.86) | ≈ −3.5 to −6.8 |

## The inductive bias we have not yet used

The metric averages *single* heads, but the federation holds G = 10 heads with different architectures, trained on
different clients. Their ensemble is much stronger than the average head in every run we have:

| Run | Mean head | Ensemble of heads | Gap |
|---|---|---|---|
| FMNIST IID, ours | 83.17 | 86.40 | 3.23 |
| MNIST LT, ours | 89.98 | 94.44 | 4.46 |
| SVHN IID, GeFL-F | 74.54 | 82.45 | 7.91 |
| SVHN LT, ours | 62.45 | 74.96 | 12.51 |

The ensemble gap is knowledge the federation already *has* but does not put into the deployed heads. It is the
classic bias–variance picture: the heads have different architectures and different data, so their errors are partly
independent, and averaging removes the independent part. The question is how to move that knowledge into each
single head without breaking GeFL-F's privacy model, in which no real data leaves a client and architectures are not
shared with other clients.

CSL already does this, but weakly. Its consensus-labelled features are seen for one synthetic epoch, then five
real-data epochs pull the head back toward its own client's optimum. The headroom recovered is only about
0.5 of the 3.2 points on FMNIST.

## H1 — Server-side ensemble distillation on generated features (SED)

The GeFL-F server already holds every architecture's aggregated head and the feature generator $G_F$. After
aggregation, for each architecture $g$:

$$\min_{\theta_g}\; \mathbb{E}_{y\sim U,\ \tilde h\sim G_F(\cdot\mid y)}\Big[-\sum_c t_c(\tilde h)\log p_{\theta_g}(c\mid\tilde h)\Big],\qquad
t = (1-\beta)\,e_y + \beta\,\tfrac1G\textstyle\sum_{g'} p_{\theta_{g'}}(\cdot\mid\tilde h)$$

This runs for $S$ SGD steps, with $\beta = 0.5$ (CSL's value, so no new weight) and batch size 128.

*Why it should transfer to real data.* For any head $h$ and loss bounded by $\ell_{\max}$:

$$R_{\text{real}}(h) \le R_{\text{syn}}(h) + \ell_{\max}\,\mathrm{TV}(p_{\text{syn}}, p_{\text{real}}).$$

Distillation lowers $R_{\text{syn}}$ toward the teacher's risk. The penalty is the generator's infidelity, which HWA
reduces under the long tail. So SED should
1. help more where the ensemble gap is larger: SVHN and CIFAR before MNIST;
2. help more with HWA than without it under the long tail;
3. hurt only if $S$ is so large that heads over-fit generator artefacts. That is testable with $S \in \{20, 60\}$.

*Cost and privacy.* The client side is unchanged, with no extra messages. The server pays $G\cdot S$ small-head
SGD steps per round, under 1 s here. This is FedDF's ensemble distillation, but on generated *features* rather than a
public dataset. That matters because FedDF on a mismatched public dataset fails badly: the paper's own Table II
reports 58.81 for FedDF on FMNIST.

*Falsification.* If SED ≤ CSL in the quick pass on FMNIST IID, H1 is wrong, and we do not run it at scale.

## H2 — The best generator depends on per-client data size (from F04)

At K = 100 (≈ 60 images per client) FSG + LA beats the repaired CVAE-F, 86.1 vs 82.7. At K = 10 the CVAE-F wins.
FSG's estimate is exact at any K, while the CVAE's rows are fit locally on shrinking data. This needs no new
mechanism: K03 on Kaggle measures it on FMNIST. It is reported as a finding, not tuned.

## H3 — The CIFAR-10 gap is mostly the feature extractor, not the generator

The paper says so: GeFL-F on CIFAR-10 is held back by the FE quality. Our oracle (heads re-fit on pooled real
features) on CIFAR-10 LT is only about 50, an upper bound for anything that keeps that FE. So no head-side method can
reach 59–62 on CIFAR-10 *with this FE*. The honest report is therefore:
(i) our gain over GeFL-F on CIFAR-10, with the same FE;
(ii) the fact that the paper's 59.36 / 62.67 come from **image-space** GeFL with DDPM / DCGAN + MixUp, a different
and far costlier pipeline with no shared FE.
We do not change the FE, because that would no longer be GeFL-F.

## Order of work tonight

1. E15 quick (FMNIST, IID + LT): does SED add to CSL and to Ours?
2. If yes: F11 full (MNIST + FMNIST, IID + LT, 3 seeds) with GeFL-F, Ours, Ours + SED.
3. K04 Kaggle file: SVHN + CIFAR-10 with SED, for the teammates.
4. Update the report, the draft and the comparison table against *every* GeFL variant, including the augmented ones.
