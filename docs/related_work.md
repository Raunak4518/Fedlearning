# Related Work (draft for §6)

*Every reference below was located by web search on 2026-10-09; the URL given is the page used to check title,
authors, venue and year. One entry (FedRS [27]) is marked: its metadata is consistent across several citing papers,
but its ACM page could not be loaded. Numbered citations [n] refer to the list at the end.*

---

## 6. Related work

**Model-heterogeneous FL by distillation and partial sharing.** FedAvg [2] assumes that every client trains the
same architecture, so that parameters can be averaged. A first family of methods replaces parameter averaging with
an exchange of predictions. FedMD [3] and DS-FL [6] share client outputs on a public or unlabelled dataset and
distil the consensus back into each model. FedDF [4] fuses heterogeneous client models at the server by ensemble
distillation on unlabelled or generated inputs. FedGKT [5] moves a large model to the server and trains it by
distilling from small edge models. AvgKD [7] analyses these protocols in a kernel-regression model and shows that
distillation-based fusion degrades under data heterogeneity. A second family shares only part of the network.
LG-FedAvg [8] keeps local representations and averages a global head. FedRep [9] does the reverse: it learns a
shared representation and personal heads. FedProto [10] exchanges class prototypes instead of parameters. GeFL-F
[1] combines both ideas: it shares a frozen feature extractor and a federated feature generator, and it leaves
the heads architecture-specific. Our work keeps GeFL-F's communication pattern and architecture freedom
unchanged. It studies what that pattern does to rare classes, a question the distillation and partial-sharing
literature has not asked of a shared *generator*.

**Generator-based and data-free knowledge transfer.** FedGen [11] trains a lightweight generator in feature space at
the server. The generator is fitted so that the ensemble of client predictors classifies its samples as the
conditioning label, and clients then regularise local training with these samples under that hard label. Its
supplementary variant weights each client's predictor per label by that client's label count. FedFTG [12] uses a
server-side generator to fine-tune the global model by data-free distillation. It samples labels in proportion to
global class counts and weights each client's teacher loss per class by the client's share of that class. DENSE
[13] trains a generator against the client ensemble and distils the ensemble into a global model in one round.
FedCG [14] has clients share conditional GANs in place of their private extractors. FedCVAE [15] fits client
conditional VAEs for one-shot FL under extreme heterogeneity. Felo/Velo [16] trains a server-side conditional VAE
on clients' mid-level features and sends synthetic features and per-class averaged logits back to heterogeneous
clients. FedVTC [17] shares feature mean and covariance statistics, from which clients generate data to fine-tune
heterogeneous models. GeFL [1] trains the generator itself by FedAvg across clients, with CVAE, DCGAN or DDPM
backbones, in image space (GeFL) or feature space (GeFL-F). FedGen and FedFTG apply label-count weights to the
*teacher predictions or losses* that train a server-side generator. HWA instead weights the *parameters* of a
client-trained generator, namely its per-class conditioning rows, during aggregation. It is derived from a failure
that is specific to that setting: under FedAvg with Adam's coupled weight decay, rows of classes held by fewer
than half the clients shrink every round.

**FL under class imbalance and long tails.** Much of this work locates the bias in the classifier. CCVR [18]
re-calibrates the classifier after training on virtual features sampled from a Gaussian mixture built from
federated feature statistics. Our federated Gaussian sufficient-statistics generator is close to this idea. CReFF
[19] re-trains the classifier on "federated features" optimised so that their gradients match those of real data.
FEDIC [20] uses ensemble distillation with logit adjustment and a calibration gating network. A second group
changes the local loss. Wang et al. [22] infer each round's class composition with a monitor and counter it with
a Ratio Loss. FedLC [21]
calibrates logits by each client's label frequencies through pairwise margins. FedRS [27] restricts softmax updates
for classes a client does not hold, and MAP [28] develops this restricted-softmax view for incomplete classes.
BalanceFL [24] changes local training so that the uploaded model behaves as if trained on balanced data. CLIMB
[23] casts class imbalance as a constrained learning problem. Fed-GraB [25] re-weights gradients using an
estimate of the global long-tailed prior, and FedLoGe [26] combines a shared backbone with a static ETF
classifier and per-client classifiers. Class-aware aggregation also appears for classifier layers: cwFedAvg [29]
averages per class with weights given by client class distributions, and FedLA [30] weights client updates using
both data size and label distribution. Our LA component is a direct application of logit adjustment by the client's
own prior. It is closely related to FedLC and to the logit adjustment in FEDIC, and we do not claim it as new on its
own. What is new is the setting, a generator and heads trained in separate phases, and the finding that LA helps
only after the generator is repaired. None of these methods trains a class-conditional generator by FedAvg, so none
encounters the row collapse we analyse.

**Long-tailed recognition.** The class-balanced loss [31] re-weights classes by the effective number of samples
$E(n)=(1-\beta^n)/(1-\beta)$, a saturating model of how much new information extra samples add. LDAM [33] enforces
label-dependent margins and defers re-weighting. Decoupling [34] shows that instance-balanced representations with
a re-balanced classifier are hard to beat. Logit adjustment [32] adds $\tau\log\pi$ to the logits during training
(or subtracts it post hoc), which is Fisher-consistent for the balanced error. We borrow the effective number as an
inverse-variance weight for *holders* of a class, not as a loss weight. We use logit adjustment with a per-client
rather than a global prior, which needs no communication. Our diagnosis also depends on the difference between
coupled and decoupled weight decay in Adam [43].

**Synthetic data, label quality and soft labels.** Knowledge distillation [35] trains a student on a teacher's soft
predictions, and label smoothing [36] mixes the one-hot target with a uniform distribution, which can remove
inter-class information useful to a student [37]. Menon et al. [38] show that a teacher that approximates Bayes
class-probabilities lowers the variance of the student's objective, with a bias–variance trade-off when the
approximation is poor. Work on training classifiers with generated data finds that label fidelity of samples, not
only visual quality, limits accuracy. The Classification Accuracy Score [39] trains on generated samples and shows
large drops relative to real data, with class-specific failures. Later studies of diffusion-generated training data
report gains from synthetic data together with clear limits [40, 41, 42]. CSL applies this view to generated
*features* in heterogeneous FL. Its target mixes the one-hot label with the mean softmax of all architectures' heads,
so it is label smoothing toward a learned ensemble posterior rather than toward a uniform distribution. The
bias–variance argument of [38] explains why the mixture ($\beta=0.5$) beats the pure ensemble ($\beta=1$), and why
lower-bias labels make a larger synthetic budget useful. Unlike FedDF, DENSE or our negative server-side variant,
CSL does not distil at the server. It only relabels the clients' synthetic phase.

**Privacy.** Secure aggregation [44] lets a server learn only the sum of client vectors. HWA needs only the sums
$\sum_k E(n_{k,c})\,w_c^{(k)}$ and $\sum_k E(n_{k,c})$, so it runs inside such a protocol and reveals neither
individual rows nor individual histograms. Differential privacy [45] and its use in deep learning [46] and in
federated training at the user or client level [47, 48] give formal guarantees. Our DP experiment adds Laplace
noise to class counts only. A formal accounting for the generator itself is left to future work.

---

## Positioning relative to the closest works

GeFL/GeFL-F [1] was evaluated only with IID clients and aggregates every generator parameter by flat FedAvg. We
show that, combined with Adam's coupled weight decay, this shrinks the conditioning rows of every class held by
fewer than half the clients, and HWA removes exactly this failure while reducing to FedAvg when class counts are
equal. FedGen [11] and FedFTG [12] also use label-count weights, but on the teacher predictions or losses that train
a *server-side* generator; we weight the *parameters* of a *client-trained* generator. Both label generated samples
with the hard conditioning label, whereas CSL relabels them with a one-hot/cross-architecture-consensus mixture.
CReFF [19] and CCVR [18] use synthetic or virtual features to re-train one homogeneous classifier after training,
while we keep per-architecture heads and change how the generator is aggregated and its samples labelled throughout
training. FedLC [21] already debiases local training with client label frequencies; our LA belongs to the same
family, and our claim for it is only empirical: it is needed, but it pays off fully only after HWA. FedDF-style
server distillation [4] of the ensemble on generated features lowers IID accuracy in our hands. CSL uses the same
ensemble only as a low-variance label inside the clients' synthetic phase, and this is what makes a larger
synthetic budget pay off. The closest labelling scheme is Felo/Velo [16]. Its server averages clients' logits
*per class* (one vector per class, computed on real data) and uses them as KL targets on clients' *real* samples; its
generated features serve only as MSE targets for the feature extractor. CSL instead labels *each generated feature*
with the consensus of all architectures' heads on that very feature, and trains the heads on it. The per-sample
consensus is what tracks the generator's per-sample label error, the bias $b$ of Proposition 3.

---

## References

[1] H. Kang, S. Cha, J. Kang. "GeFL: Model-Agnostic Federated Learning with Generative Models." IEEE Transactions on
Mobile Computing, 2025 (arXiv:2412.18460). https://arxiv.org/abs/2412.18460

[2] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas. "Communication-Efficient Learning of Deep
Networks from Decentralized Data." AISTATS, PMLR 54:1273–1282, 2017. https://proceedings.mlr.press/v54/mcmahan17a.html

[3] D. Li, J. Wang. "FedMD: Heterogenous Federated Learning via Model Distillation." NeurIPS Workshop on Federated
Learning for Data Privacy and Confidentiality, 2019 (arXiv:1910.03581). https://arxiv.org/abs/1910.03581

[4] T. Lin, L. Kong, S. U. Stich, M. Jaggi. "Ensemble Distillation for Robust Model Fusion in Federated Learning."
NeurIPS, 2020 (arXiv:2006.07242). https://proceedings.neurips.cc/paper_files/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Paper.pdf

[5] C. He, M. Annavaram, S. Avestimehr. "Group Knowledge Transfer: Federated Learning of Large CNNs at the Edge."
NeurIPS, 2020 (arXiv:2007.14513). https://arxiv.org/abs/2007.14513

[6] S. Itahara, T. Nishio, Y. Koda, M. Morikura, K. Yamamoto. "Distillation-Based Semi-Supervised Federated Learning
for Communication-Efficient Collaborative Training with Non-IID Private Data." IEEE Transactions on Mobile
Computing (DOI 10.1109/TMC.2021.3070013) (arXiv:2008.06180). https://arxiv.org/abs/2008.06180

[7] A. Afonin, S. P. Karimireddy. "Towards Model Agnostic Federated Learning Using Knowledge Distillation." ICLR,
2022 (arXiv:2110.15210). https://iclr.cc/virtual/2022/poster/6644

[8] P. P. Liang, T. Liu, L. Ziyin, N. B. Allen, R. P. Auerbach, D. Brent, R. Salakhutdinov, L.-P. Morency. "Think
Locally, Act Globally: Federated Learning with Local and Global Representations." NeurIPS Workshop on Federated
Learning, 2019 (arXiv:2001.01523). https://arxiv.org/abs/2001.01523

[9] L. Collins, H. Hassani, A. Mokhtari, S. Shakkottai. "Exploiting Shared Representations for Personalized
Federated Learning." ICML, 2021 (arXiv:2102.07078). https://icml.cc/virtual/2021/poster/10309

[10] Y. Tan, G. Long, L. Liu, T. Zhou, Q. Lu, J. Jiang, C. Zhang. "FedProto: Federated Prototype Learning across
Heterogeneous Clients." AAAI, 36(8):8432–8440, 2022. https://ojs.aaai.org/index.php/AAAI/article/view/20819

[11] Z. Zhu, J. Hong, J. Zhou. "Data-Free Knowledge Distillation for Heterogeneous Federated Learning." ICML, PMLR
139:12878–12889, 2021. https://proceedings.mlr.press/v139/zhu21b.html

[12] L. Zhang, L. Shen, L. Ding, D. Tao, L.-Y. Duan. "Fine-tuning Global Model via Data-Free Knowledge Distillation
for Non-IID Federated Learning." CVPR, 2022 (arXiv:2203.09249). https://arxiv.org/abs/2203.09249

[13] J. Zhang, C. Chen, B. Li, L. Lyu, S. Wu, S. Ding, C. Shen, C. Wu. "DENSE: Data-Free One-Shot Federated
Learning." NeurIPS, 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/868f2266086530b2c71006ea1908b14a-Abstract.html

[14] Y. Wu, Y. Kang, J. Luo, Y. He, L. Fan, R. Pan, Q. Yang. "FedCG: Leverage Conditional GAN for Protecting
Privacy and Maintaining Competitive Performance in Federated Learning." IJCAI, 2334–2340, 2022.
https://www.ijcai.org/proceedings/2022/324

[15] C. Heinbaugh, E. Luz-Ricca, H. Shao. "Data-Free One-Shot Federated Learning Under Very High Statistical
Heterogeneity." ICLR, 2023. https://iclr.cc/virtual/2023/poster/11962

[16] Y.-H. Chan, E. C.-H. Ngai. "Exploiting Features and Logits in Heterogeneous Federated Learning." Computer
Networks, 2025 (DOI 10.1016/j.comnet.2025.111271) (arXiv:2210.15527). https://arxiv.org/abs/2210.15527

[17] Z. Niu, H. Dong, A. K. Qin. "Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative
Models." ICLR, 2026 (arXiv:2508.01669). https://arxiv.org/abs/2508.01669

[18] M. Luo, F. Chen, D. Hu, Y. Zhang, J. Liang, J. Feng. "No Fear of Heterogeneity: Classifier Calibration for
Federated Learning with Non-IID Data." NeurIPS, 2021 (arXiv:2106.05001). https://neurips.cc/virtual/2021/poster/27280

[19] X. Shang, Y. Lu, G. Huang, H. Wang. "Federated Learning on Heterogeneous and Long-Tailed Data via Classifier
Re-Training with Federated Features." IJCAI, 2218–2224, 2022. https://www.ijcai.org/proceedings/2022/308

[20] X. Shang, Y. Lu, Y.-M. Cheung, H. Wang. "FEDIC: Federated Learning on Non-IID and Long-Tailed Data via
Calibrated Distillation." IEEE ICME, 2022 (arXiv:2205.00172). https://arxiv.org/abs/2205.00172

[21] J. Zhang, Z. Li, B. Li, J. Xu, S. Wu, S. Ding, C. Wu. "Federated Learning with Label Distribution Skew via
Logits Calibration." ICML, PMLR 162:26311–26329, 2022. https://proceedings.mlr.press/v162/zhang22p.html

[22] L. Wang, S. Xu, X. Wang, Q. Zhu. "Addressing Class Imbalance in Federated Learning." AAAI, 35(11), 2021
(arXiv:2008.06217). https://ojs.aaai.org/index.php/AAAI/article/view/17219

[23] Z. Shen, J. Cervino, H. Hassani, A. Ribeiro. "An Agnostic Approach to Federated Learning with Class Imbalance"
(CLIMB). ICLR, 2022. https://openreview.net/forum?id=Xo0lbDt975 (listing: https://iclr.cc/virtual/2022/poster/6104)

[24] X. Shuai, Y. Shen, S. Jiang, Z. Zhao, Z. Yan, G. Xing. "BalanceFL: Addressing Class Imbalance in Long-Tail
Federated Learning." ACM/IEEE IPSN, 2022.
https://conferences.computer.org/cpsiot/pdfs/IPSN2022-6R1M30NXCSXmbVKUqzz1Of/962400a259/962400a259.pdf

[25] Z. Xiao, Z. Chen, S. Liu, H. Wang, Y. Feng, J. Hao, J. T. Zhou, J. Wu, H. H. Yang, Z. Liu. "Fed-GraB:
Federated Long-tailed Learning with Self-Adjusting Gradient Balancer." NeurIPS, 2023 (arXiv:2310.07587).
https://proceedings.neurips.cc/paper_files/paper/2023/hash/f4b8ddb9b1aa3cb11462d64a70b84db2-Abstract-Conference.html

[26] Z. Xiao, Z. Chen, et al. (corresponding author Z. Liu). "FedLoGe: Joint Local and Generic Federated Learning
under Long-tailed Data." ICLR, 2024 (arXiv:2401.08977).
https://proceedings.iclr.cc/paper_files/paper/2024/hash/db174d373133dcc6bf83bc98e4b681f8-Abstract-Conference.html

[27] X.-C. Li, D.-C. Zhan. "FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data."
ACM SIGKDD, 995–1005, 2021. https://dl.acm.org/doi/10.1145/3447548.3467254 *(metadata consistent across citing
papers, e.g. [21]; the ACM page itself could not be loaded)*

[28] X.-C. Li, S. Song, Y. Li, B. Li, Y. Shao, Y. Yang, D.-C. Zhan. "MAP: Model Aggregation and Personalization in
Federated Learning with Incomplete Classes." IEEE Transactions on Knowledge and Data Engineering, 2024
(arXiv:2404.09232). https://arxiv.org/abs/2404.09232

[29] G. Lee, D. Choi. "Class-Wise Federated Averaging for Efficient Personalization." ICCV, 1773–1782, 2025
(arXiv:2406.07800).
https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Class-Wise_Federated_Averaging_for_Efficient_Personalization_ICCV_2025_paper.html

[30] A. Khalil, A. Wainakh, E. Zimmer, J. Parra-Arnau, A. Fernández Anta, T. Meuser, R. Steinmetz. "Label-Aware
Aggregation for Improved Federated Learning." IEEE FMEC, 2023. https://dspace.networks.imdea.org/handle/20.500.12761/1742

[31] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, S. Belongie. "Class-Balanced Loss Based on Effective Number of Samples."
CVPR, 9268–9277, 2019.
https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html

[32] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, S. Kumar. "Long-tail learning via logit
adjustment." ICLR, 2021 (arXiv:2007.07314). https://iclr.cc/virtual/2021/poster/2675

[33] K. Cao, C. Wei, A. Gaidon, N. Arechiga, T. Ma. "Learning Imbalanced Datasets with Label-Distribution-Aware
Margin Loss." NeurIPS, 2019 (arXiv:1906.07413).
https://proceedings.neurips.cc/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html

[34] B. Kang, S. Xie, M. Rohrbach, Z. Yan, A. Gordo, J. Feng, Y. Kalantidis. "Decoupling Representation and
Classifier for Long-Tailed Recognition." ICLR, 2020 (arXiv:1910.09217). https://arxiv.org/abs/1910.09217

[35] G. Hinton, O. Vinyals, J. Dean. "Distilling the Knowledge in a Neural Network." NIPS Deep Learning and
Representation Learning Workshop, 2015 (arXiv:1503.02531). https://arxiv.org/abs/1503.02531

[36] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna. "Rethinking the Inception Architecture for Computer
Vision." CVPR, 2818–2826, 2016 (arXiv:1512.00567). https://arxiv.org/abs/1512.00567

[37] R. Müller, S. Kornblith, G. Hinton. "When Does Label Smoothing Help?" NeurIPS, 2019 (arXiv:1906.02629).
https://papers.nips.cc/paper/8717-when-does-label-smoothing-help

[38] A. K. Menon, A. S. Rawat, S. Reddi, S. Kim, S. Kumar. "A Statistical Perspective on Distillation." ICML, PMLR
139:7632–7642, 2021. https://proceedings.mlr.press/v139/menon21a.html

[39] S. Ravuri, O. Vinyals. "Classification Accuracy Score for Conditional Generative Models." NeurIPS, 2019
(arXiv:1905.10887). https://proceedings.neurips.cc/paper/2019/hash/fcf55a303b71b84d326fb1d06e332a26-Abstract.html

[40] R. He et al. "Is Synthetic Data from Generative Models Ready for Image Recognition?" ICLR, 2023
(arXiv:2210.07574). https://arxiv.org/abs/2210.07574

[41] S. Azizi, S. Kornblith, C. Saharia, M. Norouzi, D. J. Fleet. "Synthetic Data from Diffusion Models Improves
ImageNet Classification." Transactions on Machine Learning Research, 2023 (arXiv:2304.08466).
https://arxiv.org/abs/2304.08466

[42] M. B. Sarıyıldız, K. Alahari, D. Larlus, Y. Kalantidis. "Fake It Till You Make It: Learning Transferable
Representations from Synthetic ImageNet Clones." CVPR, 8011–8021, 2023.
https://openaccess.thecvf.com/content/CVPR2023/html/Sariyildiz_Fake_It_Till_You_Make_It_Learning_Transferable_Representations_From_CVPR_2023_paper.html

[43] I. Loshchilov, F. Hutter. "Decoupled Weight Decay Regularization." ICLR, 2019 (arXiv:1711.05101).
https://arxiv.org/abs/1711.05101

[44] K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Segal, K. Seth.
"Practical Secure Aggregation for Privacy-Preserving Machine Learning." ACM CCS, 2017. https://eprint.iacr.org/2017/281
(also https://research.google/pubs/pub47246/)

[45] C. Dwork, F. McSherry, K. Nissim, A. Smith. "Calibrating Noise to Sensitivity in Private Data Analysis." TCC,
LNCS 3876:265–284, 2006. https://www.microsoft.com/en-us/research/publication/calibrating-noise-to-sensitivity-in-private-data-analysis/

[46] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, L. Zhang. "Deep Learning with
Differential Privacy." ACM CCS, 308–318, 2016 (arXiv:1607.00133). https://arxiv.org/abs/1607.00133

[47] H. B. McMahan, D. Ramage, K. Talwar, L. Zhang. "Learning Differentially Private Recurrent Language Models."
ICLR, 2018 (arXiv:1710.06963). https://iclr.cc/virtual/2018/poster/187

[48] R. C. Geyer, T. Klein, M. Nabi. "Differentially Private Federated Learning: A Client Level Perspective." arXiv
preprint arXiv:1712.07557, 2017. https://arxiv.org/abs/1712.07557

---

### Notes for the authors (not for the paper)

* **Check before submission.** [6] DS-FL: the TMC volume and year were not confirmed; only the DOI was. [26]
  FedLoGe: only the first two authors and the corresponding author were confirmed. [40]: only the first author was
  confirmed. [27] FedRS: the ACM page could not be loaded.
* **Possible additions, not cited because metadata was incomplete:** FedCPD (Information Fusion, 2024; "class
  proxy decoupling" aggregation for observed vs. missing classes),
  https://www.sciencedirect.com/science/article/abs/pii/S1566253524002598 . SAAFL (IFIP SEC 2025; secure
  aggregation for label-aware FL, relevant to HWA's privacy argument), https://www.eurecom.edu/en/node/4899949 .
