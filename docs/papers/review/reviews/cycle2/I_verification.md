# Agent I: verification of the accepted P0/P1 items against v05

Read-only check. Version checked: git 2025d2a ("Paper v05: editorial pass of reviewer F"). The PDFs are
`elsevier/main_final.pdf` (54 pp., submission build) and `elsevier/main.pdf` (57 pp., draft), read with `pdftotext -layout`;
`final:NNN` below is a line of that text. The sources are `content/*.tex` at 2025d2a, `generated/numbers.tex` at
2025d2a, and `fedecs_lt/src`. HEAD then moved to 325fe62, which changed notebooks only. While I worked, the working tree
gained uncommitted edits to `abstract.tex`, `fill.py`, `numbers.tex` and the tables. They touch some of the problems below
(abstract wording, the sign of "+0.0", the p-value digits). They are **not** verified here.

Recomputation: `scratchpad/I/recompute.py` reads every `fedecs_lt/results/*/runs.jsonl` except the NB00 and `_` folders
(350 runs). It pairs FedECS-LT's `er_bal` with each method's `final_bal` (published) or `er_bal` (same readout) over
common seeds, then applies a one-sample t-test and Holm's correction (12 methods as published, 5 with the readout).
**Every headline value matches numbers.tex:**

| Quantity | Recomputed | Paper |
|---|---|---|
| FedECS-LT er_bal, CIFAR-10-LT, seeds 3/4/5 | 51.11 / 53.20 / 53.47, mean 52.59 ± 1.29 | 52.6 ± 1.3 |
| FedECS-LT er_bal, CIFAR-100-LT | 18.79 / 18.42 / 18.87, mean 18.69 ± 0.24 | 18.7 ± 0.2 |
| Strongest method as published | FedDF on both | FedDF |
| Margin over it | +15.06 (Holm p 0.0052); +8.65 (0.0042) | +15.1 / +8.6, p 0.0052 / 0.0042 |
| Heads only | +5.22 (0.0663); +4.26 (0.0208) | +5.2 / +4.3, p 0.066 / 0.021 |
| Same readout, vs Balanced Softmax (C10) and FedRS (C100) | +1.43 ± 2.13 (Holm 0.375); +1.47 ± 0.89 (0.0584) | +1.4 [-0.7, +3.6], p 0.37; +1.5 [+0.6, +2.4], p 0.058 |
| Same readout, vs GeFL-F | +2.29 (0.338); +3.83 (0.0095) | match |
| Same readout, vs FedDF (C100) | 0.0143 | match |
| GeFL-F as published | +18.93 ± 2.68; +11.01 ± 0.79 | match |
| Own-mix accuracy, ECE, weakest architecture | 59.2 / 28.3; 21.2 (rank 3) and 43.8 (rank 5); 48.5 | match |
| Natural imbalance, strongest-method Holm p | Derma 0.034, Organ 0.062, Blood 0.053, Path 0.015, FER 0.024 | "3 of 5" |
| One shared model: FedECS-LT with readout | 57.2 / 59.5 | 58.4 |
| One shared model: FedYoYo lead | +19.8 (seeds 3-4); +9.0 (seed 4) | match |

The abstract, the introduction and §7.1 of the final PDF contain no number that differs from numbers.tex.

## 1. Verdict per accepted P0/P1 item

PASS = fixed in the text (and code or data where it is a fact); PARTIAL = fixed in part; FAIL = not fixed or regressed;
BLOCKED = needs runs that have not returned, and the final build no longer claims them.

| ID | Source | Sev | Verdict | Evidence | Still needed |
|---|---|---|---|---|---|
| C-1 | C (c0/c1) | P0 | PARTIAL | Body correct. The abstract (abstract.tex:14-17) says "every method of the GeFL-F pipeline". The Table 8 caption says the HtFLlib methods and Local were not evaluated (final:1391-1395). §6.9 states the construct caveat (final:1215). **The highlight still says "given to all methods"** (highlights.tex:11), but the readout was given to 5 of 11 methods. | Change H4; see NEW-3. |
| C-2 | C | P0 | PASS | §6.6 gives worst = min_c (1/G) Σ_g r_gc (final:1093-1096) = metrics.py:129-133. G-mean is computed per model (App. C). | - |
| C-3 | C | P0 | PASS | The settings list in §6.5 (final:988-1010) = htfl_server.py:24,88 and runner.py:22-33. | - |
| C-4 | C | P0 | PASS | §6.7 "our earlier implementation" (final:1136). App. B gives the tolerance max(2, 3 SD), seed 3, and the older reference (final:2170-2177) = acceptance.py. | - |
| N-1 | C (c1) | P0 | PASS | §6.5: seeds 0-5, including 3-5 with the same splits, same test sets, about 36 / 17 configurations (final:1043-1050). The counts match LEDGER.md (40 / 20 rows, of which 4 / 3 are GeFL-F or FedDF). The threats section agrees (final:1171-1175). | - |
| E-1 | E | P0 | PARTIAL | The final PDF has no slot name, no placeholder, no ablation or K = 50 text and no RQ5. **But it still describes experiments that have no results:** GeFL's grouped baselines, ρ = 50/10, and the pooled reference. | NEW-1 |
| C-5 | C | P1 | PASS | Table 4 says "loss on each head, no generator" (final:1024-1026). §6.5 says "applied to GeFL-F without its generator". This matches runner.py:27-29. | - |
| C-6 | C | P1 | PASS | GeFL-F T_s = 1 appears in §5.3, §6.2 item 5, the Table 2 caption and Table 5. This matches runner.py:22. | - |
| C-7 | C | P1 | PASS (superseded) | User decision: the readout is part of FedECS-LT. The text agrees throughout: Alg. 1 line 10; §5 "adds a closing step"; RQ2; Table 6 has a FedECS-LT row and a heads-only row. fill.py `with_readout()` maps er_* to final_* for FedECS-LT only; Table C.16 and Fig. C.5 use the heads only and say so. | - |
| C-8 | C | P1 | PASS | Lemma 3 says "smaller learning rate α²η" (final:404-405). Intro: "up to a step size, FedRS". The logit-shift sentence names LA, FedLC and BSM only (final:437-441). | - |
| N-2 | C (c1) | P1 | PASS | Test-set balance is stated per dataset (final:1054-1057). The MNIST/SVHN mentions are conditional. | - |
| D-21 | D (c1) | P1 | PASS | "No baseline was tuned" (final:988). Threats: "There are no client splits that the development never saw" (final:1175). | - |
| D-1 | D | P1 | BLOCKED | MNIST, F-MNIST, SVHN and GTSRB do not appear in the results or the tables of the final PDF (they appear only as development data, final:1048, and in the reproduction check). STATUS.md: folder 1 and GTSRB not returned. | Runs |
| D-2 | D | P1 | BLOCKED | Missing runs are disclosed as not random (final:1188-1192), and the Table 11 caption says a value without ± is from one seed. Residual P3: "Table 11 gives the seeds that completed" is not true. Six methods on CIFAR-10-LT have 2 seeds (CCVR, Fed-GraB, FedECS-LT, FedLC, FedYoYo, RUCR) and the table does not show which. | Runs; per-method seeds in the caption |
| D-3 | D | P1 | PASS | §7.2 gives head/medium/tail: +36.4 vs +2.7 (C10), +12.6/+15.4/+6.5 (C100), and the readout tail gain over BSM of +0.0 / -0.1 (numbers.tex:303,413,711,821). | - |
| D-4 | D | P1 | PASS | FedLC 69.0 is named and 300 vs 200 rounds are stated (final:1419-1430). | (P3: no CIFAR-100 comparison) |
| D-5 | D | P1 | **FAIL (regressed)** | With the readout, FedECS-LT on CIFAR-10-LT (58.4) falls below FedAvg (61.0) and MOON (59.9). §7.4 now says "several **long-tail** methods are stronger: FedYoYo, FedLoGe, CCVR, FedETF, **FedAvg and MOON**" (sec7_results.tex:161; final:1382-1385), which is false. The abstract, the intro (sec1_intro.tex:76) and the conclusion (sec8_discussion.tex:41-42) say only "the strongest long-tail methods are better". | NEW-2 |
| D-6/D-26 | D | P1 | BLOCKED | The compute difference is disclosed: Table 5, and threats "not measured ... not part of the results reported here" (final:1178-1185). The `matched_synthetic_epochs` and `components` notebooks are built. | Runs |
| D-7 | D | P1 | PARTIAL | Disclosed in the threats ("Ports", final:1195-1199). No sanity-check run is planned: folder 8 has no port check. | A port sanity check, or keep it as a stated limitation (decide) |
| D-8 | D | P1 | BLOCKED | The development history is disclosed. The other branch is in `components_of_fedecs_lt` (runner.py:50-51). The text never says that CIFAR-10-LT sits at F = 0.066-0.080 against F0 = 0.05. | Runs; state the CIFAR-10 F values |
| B-G1 | B | P1 | PASS | "GeFL distils on public data; we run FedDF's distillation on GeFL-F's generated features instead" (final:954-956). The Table 1 note and App. B FedDF say the same. | - |
| B-N1/C14 | B | P1 | PASS | FedCOF [25] is cited in the intro, §2 and Lemma 4. The readout is described as "the shared-covariance discriminant of the same family ... on trained rather than pre-trained bodies" (final:195-198). Lemma 4 is credited as known (final:454). | - |
| E-2 | E | P1 | BLOCKED | Abstract: "+1.4 and +1.5, unresolved with three seeds". Conclusion: "Which of these components carry the effect beyond the readout is not yet measured" (final:1612-1613). The Gaussian-sampler arm is built (runner.py:66). | Runs |
| E-3 | E | P1 | PASS | Option (a) is taken: FedECS-LT is defined with the readout, the headline is over the published methods, the same-readout comparison sits next to it, and the heads-only row is kept. | - |
| E-4 | E | P1 | PARTIAL | The abstract now separates the components that use statistics (anchor, moment correction, readout) from the head training (LA, consensus). But T_s = 10 is not in the abstract. Contribution 2 says the statistics are used "in ... the heads" (sec1_intro.tex:66). §5's opening says "for every quantity specific to one class" a secure statistic is used (sec5_method.tex:4), while LA uses the client's local prior π̂_k. | Reword contribution 2 and sec5:4 |
| E-5 | E | P1 | PARTIAL (run BLOCKED) | §4.1 now says the shrinkage is "not a demonstrated cause" (sec4_analysis.tex:47; final:376-379), and H1 says "shrinks". **But the abstract still says "The loss has three sources: Adam with coupled weight decay shrinks ..."** (abstract.tex:5-6). The intro says "We locate three such losses". The conclusion says the method "loses rare classes in at least three places: coupled weight decay ..." (sec8_discussion.tex:34). The AdamW-only arm is built. | Abstract, intro and conclusion: "three suspected sources, the first measured but not tested causally" |
| E-6 | E | P1 | PASS | Table C.15 gives fidelity by head/medium/tail. §4.1 cites the GeFL-F values (12.9 / 9.4 / 16.8; 1.1) = numbers.tex:182-185. The unlogged real-feature accuracy is stated (final:2320-2321). | - |
| E-7/E-26 | E | P1 | BLOCKED | The development count and the untouched natural datasets are stated (final:1046-1049). LEDGER.md has no natural-imbalance section, which confirms it. Fresh seeds 6-8 are built. | Runs |
| E-8 | E | P1 | BLOCKED | Threats: not measured (final:1183-1185). | Runs |
| E-9 | E | P1 | PASS | §3 paragraph "What every client must run" (final:309-320). §5.3: "every client thus receives every architecture's current head, which FedAvg within an architecture does not reveal" (final:669-671). | - |
| E-10 | E | P1 | PASS | Claims restricted: §3 "K = G = 10, so heads are never averaged", and the §8 bullet. The final PDF contains no K = 50 text. | (K = 50 runs remain BLOCKED, folder 7) |
| E-11 | E | P1 | PARTIAL | highlights.tex: H1 "shrinks" and H2/H5 are fine. **H4 "given to all methods"** is false. **The committed highlights.pdf is stale.** It was built 05:10 from the v03 text and still says "erases rare classes", "fix generator, heads and readout", "recovers most of the long-tail loss" and "Compared in Flower". | NEW-3 |
| E-12 | E | P1 | PARTIAL | The abstract now gives the same-readout result ("by at most +0.9 points"). Missing: EMNIST is one seed, and the margin as published is significant on only 3 of the 5 datasets with three seeds. "Best on all six" is unqualified in the abstract and in contribution 3. | Add "(one with one seed; significant on three of five)" |
| E-13 | E | P1 | PASS | Conclusion: "Each is addressed by a component ...; which of these components carry the effect beyond the readout is not yet measured". (Its first sentence is covered under E-5.) | - |
| F P1-1 | F | P1 | PASS | No "Title.." in the final text (grep). | - |
| F P1-2 | F | P1 | PASS | §6.9 holds the design threats and §8 the limits of scope, with a pointer between them. No item is repeated. | - |
| F P1-3 | F | P1 | PASS | RQ5 and the ablation sentences are conditional or removed. Unconditional grouped-baseline text remains; that is a different passage (NEW-1). | - |
| F P1-4 | F | P1 | PASS | The intro routes (optimiser, prior, synthetic data, sample size) are the same as §3 (i)-(iv) (final:72-75, 293-299). | - |
| F P1-5 | F | P1 | PASS | §6.2 no longer copies §3. The novelty sentence appears once. | - |
| F P1-6 | F | P1 | PASS | Extractor layers, software, reproduction check and HtFLlib settings are in App. B. | - |
| F P1-7 | F | P1 | PASS | tab:rq is dropped and the Venue column is gone. fig:forest, hmt, realgen, shared-rounds and recall-size are in App. C. (The final PDF is still 54 pages because the appendix grew.) | - |
| F P1-8 | F | P1 | PASS | The appendix figures are cited with one sentence each, and Table C.18 is cited. | - |
| F P1-9 | F | P1 | PASS | §7.1 is split into three paragraphs. | - |
| F P1-10 | F | P1 | PASS | Dataset names come from slots or `\whennum`. Table headers are CIFAR only. | - |

**Counts (44 items):** PASS 29, PARTIAL 7 (C-1, E-1, D-7, E-4, E-5, E-11, E-12), FAIL 1 (D-5), BLOCKED 7 (D-1, D-2,
D-6/D-26, D-8, E-2, E-7/E-26, E-8).

## 2. New problems

| ID | Sev | Location | Evidence | Proposed fix |
|---|---|---|---|---|
| NEW-1 | P1 | Unconditional text: sec6_setup.tex:150-154 (group "GeFL's baselines"; "six groups"), :171-172 (their settings), :189 (tab:baselines caption sentence), :242-243 (two tab:fairness rows); sec2_related.tex:68-69 (Table 1 rows FedALA, AvgKD); sec8_discussion.tex:9-10; sec6_setup.tex:120 (ρ ∈ {100,50,10}); sec6_setup.tex:328 (pooled reference) | The final PDF presents GeFL's grouped baselines (FedAvg/FedProx/FedALA grouped, AvgKD, grouped CCVR/FedETF) as compared methods, but folder 7 has not returned and no table has them. The §8 bullet "CCVR and FedETF enter only GeFL's grouped protocol" is false twice: the grouped runs do not exist, and CCVR and FedETF are in Table 11. The one-shared-model protocol is described at ρ ∈ {100, 50, 10}, but only ρ = 100 exists. Table 1 lists 23 methods while the text says 21. | Wrap all of these in `\whennum{main/cifar10/FedAvg/bal}`, as the tab:baselines rows already are. Generate the ρ list from a slot. Drop the pooled-reference mention or make it conditional. Rewrite the §8 bullet. |
| NEW-2 | P1 | sec7_results.tex:161; abstract.tex:19; sec1_intro.tex:76; sec8_discussion.tex:41-42 | With one shared model, FedECS-LT as now defined (with its readout) gets 58.4 on CIFAR-10-LT, below FedAvg (61.0) and MOON (59.9). §7.4 calls them "long-tail methods". The abstract, intro and conclusion hide that plain FedAvg is better. fill.py's `above_list` mixes LT and non-LT methods (this is D-5). | "With one shared model, FedECS-LT is below FedAvg and the strongest long-tail methods (six methods on CIFAR-10-LT), and its readout lowers it by 3.0 points." Split `above_list` into LT and other methods. (The working tree's abstract edit, "n_above other methods are better", goes this way; not verified.) |
| NEW-3 | P1 | elsevier/highlights.tex:11; elsevier/highlights.pdf | H4 "given to all methods" repeats C-1 in the file that editors read first. highlights.pdf has not been rebuilt since v03 and states four claims the paper has withdrawn. | "With the same readout for GeFL-F-pipeline methods, the lead is 1.4-1.5 points" (at most 85 characters). Run `build.sh highlights` and `fill.py --check-highlights`, and commit the PDF. |
| NEW-4 | P2 | review/claims.md | The audit is stale after v05. 33 claims, of which 17 have empty evidence and status (C1-3, C9, C11-13, C16, C18, C20-21, C23-26, C28, C31, C33). C10's evidence cites the heads-as-trained per-seed differences (+4.1/+4.5/+7.1); with the readout they are +14.6/+14.4/+16.2. C22 is marked Supported, but its wording is wrong (NEW-2). | Re-run `claims_audit.py` and fill every row before submission. |
| NEW-5 | P3 | numbers.tex `nat/er_margin_lo`, `main/*/ertail_diff/*` (fill.py signed formatting) | BloodMNIST same-readout margin printed "+0.0", but FedDF is ahead (paired -0.03). The CIFAR-10 tail gain over BSM is "+0.0" for -0.03. | Print the sign (the working tree's `signed()` appears to do this; not verified). Say "level with FedDF on BloodMNIST". |
| NEW-6 | P3 | abstract.tex:16; numbers.tex `er_p/*` vs `er_strongest_p` | Same family, different rounding: "p = 0.338" in the abstract vs "0.37" in §7.1. | One p-format rule (working tree changes it; not verified). |
| NEW-7 | P3 | sec1_intro.tex:78 with elsevier/main.tex:82 | "(Section 9)" points to the Conclusion: a `\label` after `\section*` takes the last numbered section. | `\phantomsection\label{...}` and `\nameref`, or name the statement. |
| NEW-8 | P3 | abstract | About 255 words with the values filled in (pdftotext count without ±, %, =), against the 250-word Elsevier/NN limit. | Count with the journal's tool and trim. |
| NEW-9 | P3 | sec7_results.tex own-mix paragraph | "below 7 of 11 other federated methods ...; GeFL-F, ..., Local, ... are above it on both" lists 8 methods, because Local is not counted among the federated methods. | Name Local separately. |
| NEW-10 | P3 | front matter, declarations | Placeholders beyond names and URL: CRediT roles, Funding, AI-tool name, Acknowledgments (final:1654-1694). | Author input (with E-24). |
| NEW-11 | P3 | §7.3, §7.6 | The readout lowers FedECS-LT on OrganAMNIST (83.3 vs 83.7 heads only; heads-only is bold in Table 10) and worsens its ECE on CIFAR-100-LT (43.8 vs 41.4). The text does not say so. | One clause each. |

## 3. Claims audit spot check (5 claims)

- **C10** (§7.1 margins and p-values): values recomputed exactly. Status Supported is right, but the evidence text is stale (NEW-4).
- **C17** (same-readout comparison): all eight numbers recomputed exactly. Supported.
- **C20** (natural imbalance; status empty): "all six", +6.4 (Organ) to +18.4 (Path), and 3 of 5 significant are all correct (Holm p
  0.034 / 0.062 / 0.053 / 0.015 / 0.024). "+0.0 (BloodMNIST)" is actually -0.03, so the claim is partially supported.
- **C22** (one shared model; marked Supported): the numbers are right (58.4; FedYoYo +19.8 and +9.0 on seeds 3-4 and 4). The
  wording "long-tail methods ... FedAvg and MOON" is wrong. Should be Partially supported.
- **C32** (cost; Partially supported): 11,278/8,659 = 1.30; 7,910/7,025 = 1.13; 11,278/60.8 = 185; 7,910/542 = 14.6, printed "15".
  The status is right.

Abstract, highlights and conclusion against the tables: the abstract overstates the decay diagnosis (E-5), "best on all
six" (E-12) and, by omission, the one-shared-model result (NEW-2). The highlights overstate the readout scope (NEW-3). The
conclusion overstates the decay diagnosis (E-5). Everything else in them is supported by Tables 6-11.
