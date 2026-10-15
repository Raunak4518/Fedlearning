# GPT review package vs our adversarial dossier

*Compared 15 October 2026.*

- **GPT package:** `NeurIPS_ESA_Review_Package_All_Branches.zip`, copied unchanged to `docs/review/external_gpt/`. It is a simulated 100-reviewer panel with 43 issues.
- **Our dossier:** `docs/review/adversarial_review_dossier.md`, ten independent reviewers plus cross-examination.
- Every claim below that marks GPT right or wrong was checked against the repository or a primary source.

## 1. Bottom line

| | GPT package | Our dossier |
|---|---|---|
| Verdict | 2/6, reject as submitted, confidence 4/5 (84 of 100 simulated reviewers give 2, 10 give 3, 6 give 1) | Reject at top tier, confidence 4/5 |
| Main reasons | K15 (optimiser-fix baseline) not run; recalibration baseline missing; gate described but not in code; K14/K15 pending | Recalibration baseline missing; novelty/theory depth; final method selected per dataset on the test set and never run end to end |

**The two reviews agree on the verdict and on the two biggest issues:** the missing recalibration baseline, and the untested optimiser fix (K15). They are independent, so that agreement is real evidence.

## 2. Where GPT is wrong or weaker (checked)

| GPT item | Problem | Evidence |
|---|---|---|
| THEORY-05, GATE-02: "F ≈ 0.01 for SVHN, ≈ 0.14–0.60 in other datasets … well separated, no data near the 0.05 cutoff" | **Wrong.** GPT copied the manuscript's number. The logged gate statistic for CIFAR-10 is 0.055–0.078, right next to the cutoff. This is one of our demonstrated false statements (F6). | `class_sep_F` in every `runs.jsonl`: CIFAR-10 LT 0.066–0.078, balanced 0.055–0.063 |
| STAT-03: K = 50 entries 86.8 / 87.2 / 87.9 "with no source" | **Incomplete.** They do have a source: runs the authors set aside after the consensus-label bug fix. That is worse than "no source" (F10). | `experiments/results/F04_clients/runs_before_cslfix.jsonl` gives exactly 86.78 / 87.17 / 87.86 and 82.71 / 85.45 / 85.22 |
| EXT-01: "numbers are promising: … CIFAR-10 60.93 vs 59.36" | **Misses that the gain disappears** at the final round: 56.82 vs 56.68 (F8) | K08 cifar10_iid `final_bal` |
| TH-01, TH-02: theorem "correct but conditional" | **Misses the false sentence** "a class held by every client cannot be contracted" (counterexample; our own T01 log shows classes 3–4 contracting), the ε-limit of λ-invariance, and the wrong variance formula in Prop 3 | `R2_proofs.md` (s2, s1, s5 outputs) |
| BR-01 marked CRITICAL (latest work is on an unmerged branch) | **Real but over-graded.** NeurIPS reviewers see a submitted PDF and an anonymised supplement, not branches. It is housekeeping, not a reason to reject. | – |
| PRIV-02: "the manuscript correctly states the generator's FedAvg is not DP" | **Too generous.** The abstract still says "formal privacy is cheap", and other passages say "every statistic made differentially private" (F12) | `paper_draft.md:51-52`; `R6_privacy.md` |
| COMM-01: byte accounting only | **Misses compute.** Ours-A is 2.2–2.8× slower than GeFL-F end to end, kernel herding is 83–88% of Stage III, and the draft's "+1.9%" measures a different variant (F13) | `R7_systems.md` timing table |
| "100 reviewers" | The package itself says it is "a simulated panel, not 100 independent model instances". Treat it as one review organised into topics, not 100 votes. | package header |

## 3. Issues only our dossier found (all demonstrated)

**Evaluation and statistics**
- The final method was never run as one pipeline. K12, the gated method, is NOT RUN. The SVHN headline (65.6, 77.07) and the CIFAR-10 LT headline (46.1) use 1 synthetic epoch, while the stated method uses 10 (F5).
- There is no validation split, and git history shows design choices were made after seeing test results (F5).
- Training is non-deterministic. For Ours-A on MNIST, same-seed replicate variation is larger than seed-to-seed variation (F9, F15).
- At 10 clients every architecture has exactly one client, so heads are never federated (F14).

**Errors in the text**
- The abstract's "+12 to +20 from HWA on all four datasets" is false for CIFAR-10, where the gain is +7.0 (F10).

**Novelty**
- DP-MERF (= KME-Gen), FedRoD (= logit adjustment with client counts) and FedVTC (a generator conditioned on class statistics for heterogeneous FL) are close precedents and are not cited (F2).

**Mechanism claims**
- PC-VAE replacing HWA adds nothing on its own (+0.34 / −0.65, not significant).
- Kernel herding's MMD rationale is contradicted by our own fake-only test (F11).
- The kernel-herding risk bound cannot apply to ReLU heads (F4).

**Data and privacy**
- CIFAR-10's real imbalance is 48:1, not 100:1. The tail classes are the same in every seed (F14).
- A membership attack on the exact class kernel mean reaches AUC 0.78–0.93 for a 24-sample class (F12; medium confidence).

## 4. Issues only GPT found (valid; added to the dossier as an addendum)

| GPT ID | Issue | Check | Our grade |
|---|---|---|---|
| NOV-04 | **FedNPC** (CVPR 2026 Findings): post-hoc classifier calibration for federated long-tailed learning using a virtual balanced dataset. It is close to our BBC and to ER, and it is not cited. | Exists: the supplement is on CVF open access ([link](https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Gao_FedNPC_Stochastic_Noise-driven_CVPRF_2026_supplemental.pdf)); it relies on a neural-collapse argument. We have not yet read the main paper. | 3 (novelty / related work; strengthens F1 and F2) |
| METRIC-01 | Worst-class accuracy is low and not reported in headlines (e.g. MC-B FashionMNIST LT: balanced 81.57, worst class 33.19 ± 4.02) | Confirmed in `E32_moment_anchor_bayes/summary.md` | 3 (reporting) |
| FMT-01 | No anonymised NeurIPS LaTeX manuscript or checklist exists. Repository metadata would de-anonymise the supplement. | True | 3 (submission readiness, not a scientific flaw) |
| BR-01 | The latest work exists only on `claude/moved-session-c4mfhn`. `main` and `revision/adam-collapse-drift` are an older snapshot, so anyone reading `main` sees the old K/2 theorem. | True (branch tips as GPT lists) | 4 (housekeeping): tag one canonical commit before submission |
| HWA-01 | The effective-number weighting is called inverse-variance-optimal without proof | True; our K02 note shows the weighting rule barely matters | 4 |
| SYS-01 | `ncm_acc` (nearest-mean accuracy) is computed on test features; fine as a diagnostic, but it must not be described as server-computable | True; the gate (if implemented) must use only `class_sep_F` | 4 |

## 5. What to take from GPT's package

1. Both reviews independently put the same two experiments first: **K15** (optimiser fixes vs HWA) and **a recalibration baseline** (GeFL-F + exact readout / CCVR-style, full schedule, 3 seeds). That should settle the order of the next compute.
2. Add **FedNPC** to related work and, if it fits per-architecture heads, to the baselines alongside CCVR / Fed3R.
3. Report **worst-class accuracy** next to balanced accuracy in the main tables.
4. Before submission: merge or tag one canonical commit, and build an anonymised LaTeX version with the NeurIPS checklist.
5. Do **not** copy GPT's statement that the gate statistic is ≥ 0.14 outside SVHN. It repeats an error in our own draft.

Sources: [FedNPC supplement (CVPR 2026 Findings)](https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Gao_FedNPC_Stochastic_Noise-driven_CVPRF_2026_supplemental.pdf); [CCVR](https://arxiv.org/abs/2106.05001v2).

## 6. Full read of every file in the package (added after a second pass)

| File | What it contains | Anything new beyond §§2–4? |
|---|---|---|
| `NeurIPS_100_Reviewer_Package_All_Branches.md` (1,994 lines) | Verdict; branch inventory; research state; contradictions; 43 issue write-ups; maths summary; evidence ledger; novelty map; cross-examination; dashboard; 100 reviews | The narrative sections restate the issue database. Its maths summary and novelty map match ours, except that it repeats the wrong gate values (§2). |
| `NeurIPS_ESA_Review_Data_All_Branches.json` | The same data as the CSVs (issues, reviewers, cross-review, statistics, verdict) | No |
| `NeurIPS_Master_Issue_Database_All_Branches.csv` | 43 issues | Covered in §§2–4 |
| `NeurIPS_100_Reviewer_Scores_All_Branches.csv`, `NeurIPS_Review_Dashboard_All_Branches.csv` | Scores (overall 2/6: 84×, 3/6: 10×, 1/6: 6×) and per-panel averages | No |
| `NeurIPS_Issue_CrossReview_All_Branches.csv` (96 rows) | Challenge and disposition per issue: 63 "survives with narrower claim", 20 "survives, needs fix", 7 "partly resolved by limitation", 6 "narrower form" | None withdrawn |
| `NeurIPS_Branch_Inventory…csv`, `NeurIPS_Branch_Delta…csv` | Branch tips and the 15 most important files that differ between branches | Branch tips confirmed with `git ls-remote`: `main` = `revision/adam-collapse-drift` = `f9f4b44` |

**How the 100 reviews are built.** They are generated from one template. Every review cites only issue IDs from the database, and none adds a new issue. They share 10 distinct strength sentences, and each review's weakness paragraph is assembled from its issue entries. Eight database issues are never cited by any reviewer: THEORY-05, SYS-02, METRIC-01, CONS-01, EXP-04, EXP-07, EXP-09, KME-01.

**Reviewer questions that add something not in our dossier:**
- **R033: is per-client optimiser state handled as in the reference GeFL-F?** Checked; consistent. Our port of the reference generator update (`utils/localUpdateGen.py:17`) says each client keeps its optimizer momentum between rounds, "as in the original repo". `experiments/core.py` `GenClient` does the same. (The authors' original repository itself was not opened.)
- **R054: test moment correction on a multimodal class distribution.** R055: report sensitivity to the herding pool multiplier and the random-feature dimension. Both are valid, cheap robustness checks on the head stage only. Added to the dossier's test list as optional.
- **R041 / R045 / R057: concept shift, a trainable feature extractor, different domains.** These fall under our scope issue F14.
- **R026: uncertainty of the exact statistics when the rarest class has 24 samples.** This connects to the privacy finding (membership attack) and to the MC-B instability on MNIST.
