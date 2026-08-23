# TrustFake WP1 README â correction list, verified against artifacts on disk

Method: every number recomputed from `runs/predictions/*.npz` with `src/metrics.py` and `src/moderation.py` in float64, CPU only. GPU untouched. Where the five audits disagreed I recomputed and say which reading I believe. Scratch scripts: `/tmp/claude-1000/-home-samuel-renteria-Desktop/29f64a6d-3439-4bcc-8e45-e43e6291bcc6/scratchpad/{d3,d4,d4b,tables,temp,misc,short,cache,e2e,bp,coll,man,last}.py`

---

## 1. CLAIMS THAT DO NOT HOLD â ranked by blast radius

### 1.1 The temperature-invariance bound is contradicted by a table sitting in `runs/` (WORST)

**README l.67â69:** *"Bound: it is exact down to T = 0.5; below T â 0.2 float64 softmax saturates MSP to exactly 1.0 and manufactures tie blocks, at which point AURC moves by ~4e-3."*

This is the headline theoretical claim of WP1 and it is falsified by his own artifact. `/home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/wp1/runs/d3_standard_eps2_255.md` prints the T=1 control row directly beneath the fitted-T row:

```
overconf_eps0.0157_s20        0.8435  ... 0.4135  194.24   89.68 ... 19.25%  12674
overconf_eps0.0157_s20 (T=1)  0.8435  ... 0.5000  156.50  156.50 ... 15.65%      1
```

I reproduced this from `runs/predictions/predictions_effb0_standard_eps2_255_test_overconf_eps0.0157_s20.npz`:

| T | AURC Ã10â»Â³ | AUROC(fail) | n_op | n saturated to 1.0 |
|---|---|---|---|---|
| 100 | 194.296 | 0.4135 | â | 0 |
| 3.3378 (reported) | 194.243 | 0.4135 | 12,674 | 0 |
| 2.75 | 191.774 | 0.4139 | 682 | 1,175 |
| **2.0** | **156.537** | **0.4999** | 30 | 15,268 |
| 1.0 | 156.503 | 0.5000 | 1 | 15,316 |

AURC moves **37.8e-3** â 9Ã the quoted ~4e-3 â and AUROC(failure) moves 0.0865 on a quantity the README says cannot move. The break is at Tâ2.5, an order of magnitude above the stated Tâ0.2, and *inside* the range declared exact. Two further rows in the same file also move under T (`pgd_linf` AUGRC 502.20 vs 973.47, n_op 12,205 vs 259; `ace_uint8` AURC 329.51 vs 333.22). Three of six "control" rows disagree, and the table presents them as agreeing.

Where the auditors split: auditor 5 read the sentence as describing the unit-test fixture (`tests/test_metrics.py`, `rng.normal(0, 2.5)`, max|margin|â15) and marked it IMPRECISE. On that fixture the bound is fine. **I side with auditors 1 and 4.** The sentence sits in the Results section, immediately after a results table, with no fixture qualifier; a referee will apply it to the artifacts, and on the artifacts it fails. Note the mechanism is *not* a `[0, margin]` contract issue (auditor 1 conflated these) â the margin `zââzâ` is shift-invariant. The cause is simply that trained checkpoint heads produce margins up to 174, against 12â13 for the linear probes the bound was derived on.

**Replacement:**
> Bound: MSP = Ï(|margin|/T) is strictly monotone, so the *ordering* is exactly temperature-invariant. The *representation* is not: float64 Ï returns exactly 1.0 once |margin|/T â¥ 36.74, and neighbouring margins start colliding into tie blocks well before that. On the linear probes (max|margin| â 12) the invariance is bit-exact for every T â¥ 0.5 â AURC 67.054405e-3 at T â {0.5, 1, 2, 5, 10}. On the WP3 checkpoints it is not: `effb0_standard` under the over-confidence attack reaches max|margin| = 118.6, and its AURC runs 194.24e-3 at the fitted T = 3.338 down to 156.50e-3 at T = 1. The guard is `n_op`: that cell already reports 12,674 operating points out of 15,316 at its own reported temperature. Any row with n_op < n is in the degraded regime and its AURC is temperature-dependent.

### 1.2 The img_id collision count is low by 2.2Ã, and it was never computed

**README l.74â75:** *"8,634 img_ids appear in BOTH the train and validation splits â 4,496 synthetic and 4,138 tampered"*

**True: 19,107 â 10,000 synthetic and 9,107 tampered.** Full scan of all 283 shards under `/home/samuel-renteria/Desktop/FILES/DATASETS/sid-set/data` (249 train = 210,000 unique img_ids, 34 validation = 30,000). `8634` is a **hard-coded integer literal** at `src/manifest.py:171` (`"img_id_collisions_in_full_dataset": 8634`), copied unchanged into `runs/manifest_v1.json`, `runs/manifest_v2.json` and `runs/smoke/manifest_smoke.json`. Nothing in the repo computes it; the 4,496/4,138 split appears in no file at all. Most likely counted when about half the train shards had been fetched.

This is the flagship "dataset finding that fails silently" â and the number fails silently. The correct number makes the point stronger.

**Replacement:**
> **1. `img_id` is not a key.** 19,107 img_ids appear in BOTH the train and validation splits â 10,000 synthetic and 9,107 tampered â because those classes are numbered sequentially and the counter restarts per split. Every one of the 10,000 fully-synthetic validation images shares its img_id with a train image.

Fix `src/manifest.py:171` to compute it (an img_id-column-only scan runs in a couple of minutes) rather than assert it.

### 1.3 "The detector was never riding the artifact" â the control is a no-op on 100% of the positives

**README l.130:** *"Unchanged. The detector was never riding the artifact."* (mirrored in the `src/baselines.py` docstring: *"a model evaluated under this condition cannot be riding the artifact"*)

The numbers are right (Â§5). The inference is not, and this is the one an adversarial-robustness PI will take apart in the room. Three measurements, none requiring a GPU:

1. **The treatment is bit-identical to the baseline on every fake.** Cached sizes for all 15,316 test rows (`/home/samuel-renteria/Desktop/FILES/DATASETS/sid-set/cache256`): label 1 â 3,829/3,829 square, label 2 â 3,829/3,829 square, label 0 â 334/7,658. `squarecrop` = centre-crop to min(w,h), so it crops nothing on 100% of the positive class.
2. **The uncontrolled arm never exposed geometry either.** The CLIP feature sidecar records `crop_pct 1.0`, `input_size [3,224,224]` â timm's eval transform is Resize(short sideâ224) + CenterCrop(224). Both arms deliver the same centre-square field of view at the same output size; they differ only in resampling order, on the 95.6% of reals that are non-square.
3. **The shortcut survives into the pixels in both arms.** 100% of test fakes have original short side exactly 1024 â decoded at scale 0.25; only 4.01% of reals do. A single threshold on that decode scale factor, **fitted on calib and frozen**, scores **0.9798 on test** (calib 0.9808) â the same shortcut re-expressed as a property of the model's actual input rather than of discarded metadata.

**Replacement for l.130â131:**
> Unchanged â but that is what the control was always going to say. Every fake in the split is already square in the 256px cache, so `squarecrop` is a bit-exact no-op on 100% of the positive class, and the eval transform (crop_pct 1.0: resize short side to 224, centre-crop 224) already discards aspect ratio in both arms. What this establishes is that the models cannot read aspect ratio; it does not establish that they are not riding the shortcut, because the shortcut is laundered into resampling and compression history that centre-cropping cannot touch. A frozen threshold on the decode scale factor â 1024 â 256 is a 0.25 downscale for 100% of fakes and 4.0% of reals â still scores 0.9798 on test. Whether the probe rides that residue is open. The experiment that would settle it: an evaluation subset on which format and compression provenance carry no label information â either re-encode every image through one identical JPEG pass, or restrict to realâ©JPEG vs tamperedâ©JPEG. `src/baselines.py`'s own docstring already names this as the correct fix.

Also weaken l.131: `tests/test_attacks.py:201` asserts only that `apply_condition(Image.new('RGB',(w,h)),'squarecrop').size` is square, on three blank images. It locks a tautology about a min-side crop, not "the property that the control actually removes geometry."

### 1.4 The two-numeric-paths comparison describes a run that no longer exists

**README l.318â322:**
```
cached fp16 features : acc 0.9289   AURC 11.83e-3
fp32 pixel forward   : acc 0.9292   AURC 11.82e-3
margin |delta| mean 0.0135, max 0.2573; 8 of 15,316 predictions differ
```
*"Three in ten thousand, of purely numerical origin."*

`runs/predictions/predictions_vit_l14_e2e_test_clean.{npz,json}` holds **n = 4,000**, `sample_n: 4000`, `sample_seed: 0`, `sample_uid_sha: 115f77e24f7412a3` â a class-balanced seeded subsample (commit `fc7874a`, "subsample the expensive ones"). Recomputed on the 4,000 shared uids at the frozen T = 0.9099:

| | acc | AURC Ã10â»Â³ |
|---|---|---|
| cached fp16 | 0.9313 | 12.09 |
| fp32 pixel forward | 0.9323 | 12.09 |

margin \|Î\| mean **0.0135** (matches), max **0.1703** (not 0.2573), **4 of 4,000** predictions differ (not 8 of 15,316 â and 1.0e-3 is a different rate from 5.2e-4). `src.verify` cannot catch this: its `sample_uid_sha` guard only compares rows *within* one `model_id`, so a full-split row and a subsampled row under two different model_ids are never compared.

**Replacement:**
> Same weights, same folded head, same images. Measured on the 4,000-image class-balanced seeded subsample the differentiable path runs on (`sample_uid_sha 115f77e2`):
> ```
> cached fp16 features : acc 0.9313   AURC 12.09e-3
> fp32 pixel forward   : acc 0.9323   AURC 12.09e-3
> margin |delta| mean 0.0135, max 0.1703; 4 of 4,000 predictions differ
> ```
> One in a thousand, of purely numerical origin.

### 1.5 Two D4 cells are computed against superseded predictions

**README l.293â294:** PGD column, PGD-AT **13.01%**, TRADES **13.53%**.

Recomputed with `src/moderation.py` (policy refit on each model's own clean calib at the 5% SLA, frozen): **PGD-AT 12.97%** (coverage 41.57%), **TRADES 13.51%** (coverage 36.59%). All other 13 cells reproduce exactly.

Cause is timestamp, not method: `runs/d4_{standard,at_pgd,trades}_eps2_255.json` were written at 16:19:38â45; `runs/rerun_pgd.sh` regenerated the three PGD npz at 16:25:06 / 16:28:19 / 16:31:46. D3 was regenerated after (`d3_comparative.json`, 16:31:59) and matches; D4 never was. The README's D3 and D4 tables are scored against two different generations of the same predictions.

**Fix:** re-run `src.moderation` for the three models; change 13.01 â 12.97 and 13.53 â 13.51. The narrative sentence "100% to 13%" survives unchanged. Also regenerate or delete `runs/d3_{standard,at_pgd,trades}_eps2_255.{json,md}` (16:16:53â55, also pre-rerun) â their PGD rows disagree with the npz: `d3_standard` prints n_op 12,205 where the file gives 10,102; `d3_at_pgd` prints AURC 147.05 vs 146.82; `d3_trades` 170.51 vs 170.33. A referee who opens the per-model table and the comparative table will find two different AURCs for the same row.

### 1.6 "the only one that drives coverage UP" is refuted by a row printed two paragraphs above

**README l.302â303:** *"The residual threat is the label-free over-confidence attack: it is the only one that drives coverage UP (to 100% undefended, 80.8% with AT)."*

The parenthetical is exact (100.00%, 80.83%). The claim around it is not. On the standard model, all three attacks raise coverage from its clean 72.90%: PGD â **100.00%**, ACE uint8 â 80.69%, over-confidence â 100.00%. PGD also stops the system abstaining â and takes it to 100% residual risk, which the README's own D4 table prints.

**Replacement:**
> The residual threat is the label-free over-confidence attack: it is the only one that drives coverage UP on the *defended* models (PGD-AT 49.9% â 80.8%, TRADES 44.6% â 74.0%), where PGD and ACE both push coverage down.

### 1.7 "rounding a sub-step perturbation back to uint8 erases it" is true for one of the three epsilons

**README l.135â138.** The means are right â `eps_effective_mean Ã 255` = **0.102346 / 0.260736 / 0.447469** grey levels, exactly the printed 0.102/0.261/0.447. But the discriminator for survival under round-to-nearest is the **per-sample max**, and `eps_effective_max Ã 255` = **0.128 / 0.510 / 1.275**. Only eps=0.0005 sits entirely below the 0.5 rounding radius.

The README's own next table contradicts the mean-based argument: a blanket erasure claim predicts all three uint8 rows to be no-ops, and only eps=0.0005 is. Measured survival at eps=0.002/0.005: **3,889 and 3,890 of 15,316 rows (25.4%)** carry a full 1-grey-level change (`|Îmargin|` above the 8.31e-3 batch-shape noise floor; independently confirmed by `eps_effective_mean/eps_effective_max` = 0.25393/0.25399).

**Replacement:**
> Unquantised ACE produces mean per-sample Lâ perturbations of 0.102 / 0.261 / 0.447 grey levels at eps = 0.0005 / 0.002 / 0.005 â but the quantity that decides survival is the per-sample maximum, which is 0.128 / 0.510 / 1.275. Round-to-nearest annihilates a step only below half a grey level (0.5/255 = 0.00196; the step itself is 1/255 = 0.00392). So eps=0.0005 vanishes exactly â that is the control â while at eps=0.002/0.005 a quarter of the images (3,889 of 15,316) retain a full one-grey-level change and the attack is realisable. Unquantised numbers still must not be quoted as file-upload numbers, because a saved file is a different input â not because the perturbation is erased.

Also fix the sub-clause: 0.5/255 is **half** a quantisation step, not one. The same mislabel is repeated at `src/attack_suite.py:139`, so it reads as a definition rather than a typo â exactly the slip that makes two Lâ-budget specialists re-check everything else.

### 1.8 Permutation spread: 5.6e-3 â 1.6e-2

**README l.381 / `src/metrics.py:23` / `tests/test_metrics.py:135`.** Re-running the repo's own fixture (`default_rng(5)`, n=600, r=0.25, conf rounded to 10 levels, 8 permutations from seeds 0â7): naive spread = **1.62e-2**. Over 100 permutations 2.33e-2; over 1,000, 2.99e-2. Block-weighted spread is exactly 0.0 throughout. The test only asserts `> 1e-3`, so it never checked the stated figure. The error is in the safe direction â the defect is 3Ã worse than advertised â but it is a hard number stated in three places that does not survive re-running the fixture. Say **1.6e-2 over 8 permutations, growing with the number sampled**, and tighten the assert to bracket it.

### 1.9 Smaller, but each one is a free hit

| README | true | source |
|---|---|---|
| l.7â8 *"Everything below â¦ reproducible with `python -m src.run_all`"* | `run_all` invokes manifest, decode, features, probe, attacks, score, figures. It never invokes `baselines`, `run_attacks`, `train`, `compare`, `moderation` or `verify` â i.e. it produces none of the shortcut table, threat-model table, D3 or D4. Those came from `runs/{clip_attacks,d3_eval,rerun_pgd}.sh`. | `src/run_all.py:48â74` |
| l.177 / l.345 *"seven attacks"* | **eight**: fgsm, pgd_linf, pgd_l2, autoattack, ace, **ace_uint8**, overconf, underconf. The D2 table omits `ace_uint8` â the attack the whole threat-model section rests on. | `src/attack_suite.py:254` |
| l.114 *"only ~5% of real images are square"* | **4.30%** on the test split (329/7,658); 4.22% over all 30,000 validation rows. | full parquet scan; matches `runs/trivial_baselines.json` square_rate label0 = 0.042961608775 |
| l.118 *"every table prints it"* (the 0.9785 baseline) | 6 of 14 generated tables print it. Missing from `condition_table_clip.md`, `condition_table_realistic.md`, **`d3_comparative.md`**, all three `d4_*.md`, both `moderation_*.md`. | grep over `runs/*.md` |
| l.69â70 *"`n_op` â¦ is printed beside AURC in every table"* | absent from `d3_comparative.md`, all `d4_*.md`, both `moderation_*.md` â and from **all eight README tables**. It is the one column that would have caught Â§1.1. | `src/compare.py` cell format; `src/moderation.py` COLS |
| l.372â373 *"CLIP L/14 features 232 img/s, EfficientNet-B0 features 784 img/s"* | CLIP full-split: **189.0, 189.0, 190.6, 198.0, 202.2, 214.4**. EffB0 full-split: **1142.6, 1298.7, 1476.6, 1517.4, 1548.9**. 232 is above every CLIP measurement (closest is the *smoke* run, 236.4/237.1); 784 matches nothing anywhere, including smoke (538.3). Decode 214 img/s and ACE 41â62 s are correct. | `/home/samuel-renteria/Desktop/FILES/DATASETS/sid-set/features/*.json`, `runs/smoke/features/*.json`, `runs/decode_calib_test.log` |
| l.204â205 *"loss pinned at ln 2"* | 0.7647 â 0.7525 â 0.7411 â 0.7254 â 0.7171 â 0.7102, descending toward ln 2 = 0.6931, never pinned. Say *"loss decaying toward ln 2 (0.7102 by epoch 6, from 0.7647)"*. The collapse itself (clean and robust both 0.4943 at epochs 4â6, killed at 6) is real. | `runs/train_wp3_eps8_255_collapsed.log` |
| l.361 *"<2 s"* | 2.48 s reported by pytest, 3.15 s wall over three runs. 19 tests (9 + 10) is exact; no GPU/data/network is exact. | `pytest tests/ -q` |
| l.18 interface spec *"logits float32[n,2] `[0, margin]`"* | true for the linear probes; **22 of 44 npz** (every `effb0_standard/at_pgd/trades` file) have `logits[:,0] â  0` â e.g. `[5.907, â5.919]`. Metrics are unaffected (softmax is shift-invariant), but this is the "publish this on Day 1" contract and half the artifacts break it. | direct read of the npz |

Also: the epsilon-finding table (l.211â214) row `PGD-AT eps=2/255, 4-epoch warm-up 0.8251 / 0.7359` comes from `runs/train_wp3_eps2.log` / `runs/checkpoints/at_pgd.json` â a **different run** from the D3 arm `at_pgd_eps2_255` (0.8269 / 0.7305, `runs/train_wp3_matched.log`). Two near-identically-named checkpoints at the same epsilon with different selected epochs is a trap in live discussion.

---

## 2. CLAIMS THAT HOLD, STATED TOO STRONGLY

**"â¦while ECE moves 0.33 â 0.47" (l.62â63).** The invariance half is exactly right: at eps 0.0005/0.002/0.005, AURC (353.83/466.86/469.14), AUGRC (136.47/154.23/154.63), AUROC(fail) (0.1295/0.0030/0.0002) and risk@cov.5 (32.76/33.77/33.78%) are bit-identical between T=1.2785 and T=1. But the ECE move under temperature is 0.4876 â 0.4715 on the attacked row and 0.0080 â 0.0318 on clean. **0.33 â 0.47 is the epsilon axis** (0.3257 at eps=0.0005 â 0.4732 at eps=0.002), not the temperature axis. Both auditors who checked agree; I confirm. As written it invites a referee to check the pair against the table, find it is the wrong axis, and doubt the invariance result that is actually correct.
> *"â¦while ECE on the attacked row moves 0.488 â 0.472 and on the clean row 0.008 â 0.032. The calibration metric responds to temperature; the selective metrics cannot."*

**"The sub-step row vanishes exactly as predicted" (l.150).** Metrically true â AUROC(fail) 0.7703, AURC 67.05, residual 4.65%, argmax preserved on all 15,316 rows. But only **3 of 15,316** logit rows are bit-identical to clean (max |Îmargin| 8.31e-3), and `eps_effective_max` = 5.96e-08 (one fp32 ulp), not zero. Cause is the accept-check batch shape, which the README documents at l.85â88 and l.328â337 without connecting the two. Say **"vanishes to the reported precision"**, and link it to the cuDNN section â that pre-empts the exact question a careful reviewer asks.

**"ACE eps=0.002-0.005, uint8" as one row (l.148).** True at the printed precision, false literally: exactly **one** of 15,316 rows differs (index 10846), where the margin is 3.7503 at eps=0.002 and 0.00067 at eps=0.005 â p(fake) 0.949 â 0.500. Add *"(identical to the precision shown; the two runs differ on 1 of 15,316 samples)"* or print them separately.

**The coincidence itself is unexplained and that is the likeliest challenge.** Nothing in the README says why eps=0.002 and eps=0.005 give the same row. The mechanism is fully derivable from `src/attack_suite.py:163â178` â `round(cand*255)/255` inside a loop that halves eps per rejection. In grey levels: eps=0.002 â 0.51, 0.26, â¦ and eps=0.005 â 1.28, 0.64, 0.32, â¦; round-to-nearest maps both ladders onto {1 grey level, nothing}. Both attacks explore the same two-point set. It also predicts the eps=0.0157 row on disk, which has `eps_effective_max` = 4.000 grey levels and does move (AUROC(fail) 0.0378, AURC 442.37e-3). Two sentences here turn a coincidence that reads like a copy-paste error into a mechanism.

**"generalized == coverage Â· selective elementwise (asserted)" (l.377â378).** Max absolute error 5.551e-17 over all 44 npz, well inside the suite's own 1e-15 assert. If pressed on "elementwise", say *"to 3e-16 relative"* â float division is not associative.

**"E-AURC â¦ 9.65e-05 off at n=1000" (l.383â384).** The policy (empirical oracle) is right and worth keeping; the number is quoted as a constant when it is a function of r. The gap is exactly r/(2n): 2.50e-05 at r=0.05, 8.45e-05 at r=0.1689 (the EffB0 clean rate), 1.00e-04 at r=0.20. 9.65e-05 corresponds to r = 0.193, an error rate that appears nowhere in the repo.
> *"â¦never the asymptotic closed form r + (1âr)ln(1âr), which is low by r/(2n) â 8.4e-05 at our clean error rate and n=1000, the size of the effects being reported."*

**"Every prediction sidecar records `batch`" (l.333).** The key exists in all 44, but is `null` in the four CLIP cached-feature sidecars â the artifacts behind the headline Results table and the geometry control â and `src.verify` prints a warning for exactly those four in the same session. Say *"every sidecar produced by `run_attacks` records `batch`; the four cached-feature CLIP rows pre-date the field and `src.verify` warns about them."*

**"All three arms share one initialisation and one 12-epoch schedule â¦ a difference between rows is a difference of METHOD, not of budget" (l.246â248).** The *run* budget is genuinely matched â `runs/train_wp3_matched.log` shows all three completing ep12/12 at 473 steps/epoch = 5,676 optimizer steps, and the checkpoint mtimes reconstruct the timeline exactly. Auditor 5 is right that the claim is defensible; auditor 4 is right that the sidecars contradict it on a plain reading and that the *scored* weights did not get equal training. `runs/checkpoints/*.json` record `optimizer_steps_spent` 1,419 / 4,257 / 2,838, `completed_full_schedule: false`, and `selected_on` = `fit_val_clean` for standard vs `fit_val_robust_pgd` for both defenses. The standard arm's epoch-3 pick is at the OneCycleLR peak (lr 0.0200) against 0.0050 and 0.0150 for the others.
> *"All three arms share one initialisation and one matched 12-epoch / 5,676-step run budget on an identical data protocol and evaluation. Model selection then picked epoch 3 / 9 / 6, so the scored checkpoints carry 1,419 / 4,257 / 2,838 updates and were selected on clean vs robust fit_val respectively."*

Three artifact fixes that go with it, because a WP3 referee will open these files:
- `runs/checkpoints/phase1_init.pt` is **byte-identical to `standard.pt`** (both sha256 `d015ea80858c70bdâ¦`) â the "shared initialisation" is the earlier standard run's epoch-7 checkpoint renamed with a `role` field. No phase-1 training log exists.
- `phase1_init.json` claims `optimizer_steps: 5676` and `epochs: 12` for a run whose history stops at epoch 7 (actual 3,311 steps). This is verbatim the defect `src/verify.py`'s docstring says was already caught â still present, on the file the parity claim rests on. It also carries `eps: 0.03137` (8/255) left over from the collapsed configuration, which matters because 8/255 is the headline epsilon finding.
- The committed sidecars use the *old* schema (`optimizer_steps_spent`, `completed_full_schedule`); the current `src/train.py:238` writes `optimizer_steps_to_selected_epoch`, `selected_epoch`, `epochs_planned`, `is_last_epoch`. The code in the repo cannot regenerate the artifacts in the repo field-for-field. There is also no `init_from` field written anywhere, so the shared-initialisation claim rests on three stdout lines in a log.

**The validation caveat is stated twice in a 24 KB document, and the split is named "test."** Line 27 (heading) and lines 161â163 (limitation bullet). Six results tables carry no provenance line, while **17 generated artifacts in `runs/` carry** `provenance: SID-Set validation split, held-out slice (official test split withheld)` â the code is more honest than the README. And "the exact test split" appears bare at l.104, in the most quotable table in the document, 57 lines from the caveat; also at l.242 and l.316. Add the provenance line under every table (and to `src/compare.py`'s output â `d3_comparative.md` is the one artifact that omits it), and never write "the test split" unqualified: say *"the held-out validation slice (n = 15,316)."* The README says "state these on every slide"; right now it does not follow its own instruction, and this is the caveat two trustworthy-AI researchers are least likely to forgive.

---

## 3. NUMBERS WITH NO ARTIFACT

Inherited or remembered rather than computed here. Each is a "can I see the code for that?" waiting to happen.

1. **8,634 / 4,496 / 4,138 img_id collisions** â hard-coded literal, `src/manifest.py:171`; the breakdown exists in no file. True value 19,107 / 10,000 / 9,107 (Â§1.2).
2. **232 img/s CLIP, 784 img/s EffB0** â no sidecar anywhere matches either. Measured ranges 189â214 and 1,143â1,549.
3. **e2e block: acc 0.9292, AURC 11.82e-3, max |Î| 0.2573, 8 of 15,316** â the full-split e2e run was replaced by a 4,000-row subsample.
4. **"Regenerating the clean baseline at batch 16 instead of 32 moved accuracy by one sample in 15,316" (l.330â331)** â batch values across the 44 sidecars are 32 (Ã36), null (Ã4), 16 (Ã3), and all three batch-16 files are `vit_l14_e2e`, not an EfficientNet clean baseline. The run was overwritten, which is the point of the story but leaves it unreproducible.
5. **"preservation reads 0.9980" (l.87)** â appears only in two source docstrings (`src/attacks.py:108`, `src/attack_suite.py:151`), never in a run artifact.
6. **"5.6e-3" permutation spread** â contradicted by the repo's own fixture (1.62e-2).
7. **"9.65e-05 at n=1000"** â r never stated; corresponds to r = 0.193, which appears in no table here.
8. **`src/metrics.py:198â199`: "at a fixed AUROC(failure) of ~0.921, E-AURC still moves from 0.0077 to 0.0227"** â not in the README, but it is the docstring for `eaurc()` and will be read if anyone opens the module. Neither value appears anywhere in `runs/`. The only 0.921 on disk is `condition_table_clip.json` "real vs synthetic", auroc_failure 0.92145, whose eaurc is 0.00428. There is no second row near 0.921 to pair it with. The *direction* is real (E-AURC is not error-rate-free), but the pair is inherited. The harness has real pairs that make the point â e.g. `effb0_at_pgd` clean (AUROC(fail) 0.8006, E-AURC 44.6e-3) vs `effb0_trades` clean (0.7761, 58.2e-3).
9. **AutoAttack gotcha (l.392â398)** â corroborated only by the `autoattack_safe` docstring, which records it as "MEASURED on this box." No autoattack run exists in `runs/`. **UNVERIFIABLE without the GPU** â flag it to the PIs as documented-but-not-artifacted. One short CPU run against a 2-logit toy model would produce a committable artifact.
10. **"~8 min on a 3090" for `src.run_all`** â the 3090 is confirmed; summing committed per-stage times for the stages run_all actually invokes gives ~7â8 min, so it is plausible for the WP1 core. **UNVERIFIABLE without the GPU.** Qualify: *"~8 min for the WP1 core; WP3 training is a further ~70 min and the D3/D4 evaluation ~30 min."*
11. `runs/trivial_baselines.json` carries **no `manifest_sha`** â it is the only artifact in `runs/` with no provenance stamp, and it is the number every accuracy is read against. (I re-derived it independently from the raw shards; it is correct. But stamp it.)

---

## 4. THE INFERENCES WORTH ATTACKING

**(a) The geometry control (Â§1.3).** Confound: `squarecrop` is bit-identical to `clean` on 100% of fakes, and both arms feed the backbone a 224Ã224 centre crop, so the treatment and the baseline are the same experiment on the class that matters. Settling experiment: a format/compression-matched subset â re-encode every image through one identical JPEG pass, or evaluate on realâ©JPEG vs tamperedâ©JPEG where original format carries no label information. Named in `src/baselines.py`'s own docstring as "the correct fix" and never run. Meanwhile add the decode-scale row (0.9798) to the shortcut table: converting a null result from a no-op control into a stated open problem with a named fix is a much stronger thing to hand a PI.

**(b) "It also protects CONFIDENCE" (result 3) rests on one epsilon of an attack with three warning signs.** The arithmetic is right (0.8006â0.7770, 0.7761â0.7789). The reading is not earned:
- **TRADES' AUROC(failure) goes UP under the attack.** An attack that improves the defender's failure prediction has not been shown to be at its strongest against that model.
- **The internal control cuts the other way.** On the same defended models, ACE uint8 at 0.96 grey levels mean drops AUROC(fail) by 0.135 (PGD-AT) and 0.122 (TRADES); over-confidence at 4 grey levels â 4Ã the budget â drops it by 0.024 and â0.003. A 4Ã larger budget producing a 5Ã smaller effect with a sign flip is the signature of an under-optimised attack.
- **The attack does not saturate its own ball on PGD-AT.** `eps_effective_mean` for `overconf_eps0.0157`: standard 4.0035 grey levels, TRADES 4.0035, **PGD-AT 2.565** (64% of budget) with `frac_perturbed: 1.0`. With alpha = 0.5 grey levels Ã 20 steps the ball should saturate; falling short means `g.sign()` was exactly zero on many steps â textbook vanishing-gradient masking, on the one model the conclusion depends on.

`src/attack_suite.py:182` runs 20 steps, single restart, no random init, no step-size search, no epsilon sweep. Settling experiment: random restarts + an epsilon ladder (1, 2, 4, 8/255) + a gradient-norm trace on PGD-AT to test masking directly. Carlini et al., *On Evaluating Adversarial Robustness*, is the standard the room will apply.

**(c) A budget confound the table hides.** D3's three adversarial rows use three different budgets and only one is labelled. PGD: eps = 0.00784 (2/255, exactly the AT training radius â the defenses' best case). Over-confidence: eps = 0.0157 (4/255, 2Ã the training radius). ACE uint8: `eps_effective_mean` of 0.2253 / 0.9563 / 0.9616 grey levels for standard / PGD-AT / TRADES. Result 3 reads "AT protects confidence" across rows whose budgets differ by up to 4Ã â and, in the ACE case, differ *by model*. Label the epsilon on every adversarial row, as the README's own threat-model section demands. (One point in his favour worth making explicitly at the camp: the standard model received a **4Ã smaller** ACE perturbation than the defended ones and took **6Ã more damage** â 0.109 vs 0.666/0.654. That comparison is real and it is stronger than the over-confidence one.)

**(d) The Ledda claim (l.279â282).** Auditor 4 called it untestable from one cell; I disagree in part â the hypothesis is "AT protects against under-confidence but *not* over-confidence," and showing that AT *does* blunt over-confidence is a legitimate contradiction of the second clause without needing the first. But the refutation is only as strong as the attack, which is (b), and the under-confidence arm was **never run** â zero `underconf` npz among 44, absent from `runs/d3_eval.sh`, though `underconfidence()` is implemented at `src/attack_suite.py:216` and listed in the README's own D2 table. Presenting a differential as contradicted with half the 2Ã2 missing and the run half showing under-optimisation signs is the highest-risk sentence in the document.
> *"AT and TRADES resist this over-confidence attack at 4/255 (AUROC(failure) 0.801 â 0.777 and 0.776 â 0.779). We have not run the under-confidence arm, and this attack is a fixed 20-step single-restart descent whose effective epsilon on PGD-AT is only 64% of budget â so we treat this as a question for the PIs, not a refutation."*

**(e) The undefended contrast in result 3 is measured in the eleventh decimal of float64.** The standard/over-confidence cell (0.4135) sits in the saturation regime: mean |margin| goes 9.67 (clean) â 89.36 (attacked), a 9.2Ã inflation; median (1âconf) = 4.6e-12; mean confidence is 1.0000 for correct *and* wrong predictions (0.99999999995 vs 0.99999999987 â that tiny gap, wrong-above-correct, is the entire AUROC(fail) < 0.5 signal); n_op collapses 15,270 â 12,674. And the cell has no temperature-stable value at all (194.296 at T=100, 194.243 at T=3.338, 156.5 below Tâ2.5). Report it as *"confidence destroyed â all MSP = 1 â O(10â»Â¹Â²); AUROC(failure) indistinguishable from chance,"* with n_op and a saturation flag. Reporting 0.4135 to four decimals implies a measured slight inversion that is not a robust effect.

**(f) Budget parity vs selection asymmetry (Â§2).** Confound: the scored checkpoints differ in optimizer steps (1,419 / 4,257 / 2,838) and in selection objective (clean vs robust fit_val). Settling experiment: score the epoch-12 checkpoint of each arm alongside the selected one â same steps, same objective, no selection asymmetry. There is currently no matched-budget undefended control in `runs/predictions/` to fall back on.

---

## 5. WHAT IS GENUINELY SOLID â lean on this

**The D3 comparative table is exact. All 18 cells, recomputed from the npz through `src/metrics.py` with each model's temperature refitted on its own clean calib:**

```
clean          0.8435/0.7945/52.52  0.8177/0.8006/62.28  0.7877/0.7761/82.55
jpeg_q50       0.7198/0.7188/140.41 0.8169/0.7997/62.84  0.7876/0.7730/83.41
downscale_0.5  0.7987/0.7793/75.66  0.7857/0.7853/80.64  0.7649/0.7586/100.87
pgd_linf       0.0000/nan/1000.00   0.7192/0.7054/146.82 0.7062/0.6799/170.33
ace_uint8      0.8435/0.1093/329.51 0.8177/0.6659/94.49  0.7877/0.6541/117.95
overconf       0.8435/0.4135/194.24 0.8177/0.7770/68.78  0.7877/0.7789/81.31
```
Temperatures 3.337815 / 0.903458 / 0.757392 â 3.338 / 0.903 / 0.757, all fitted on own-calib (structurally enforced in `src/compare.py`, not merely asserted). Derived narrative all checks: â2.58 pp / â5.58 pp clean-accuracy cost; standard has the best clean AURC; JPEG q50 costs standard 12.4 pp and PGD-AT 0.08 pp.

**PGD accuracy is literally 0.0000** â 0 correct of 15,316, predicted histogram exactly [7658, 7658] against a y of exactly [7658, 7658]: every prediction inverted. Cross-checks against the sidecar's `label_preservation` = 0.15654 = 1 â 0.8435. The `nan` AUROC(failure) is the empty-case guard firing (`c.min() == c.max()`), and the companion AURC = exactly 1.0 is the correct degenerate value. This is the one place the NaN-over-0.0 discipline visibly pays.

**The whole ACE confidence-attack section is exact.** Every cell of the l.43â48 table reproduces at T = 1.2784557913 (accuracy 0.83109166884304 identically across all four rows), `label_preservation` exactly 1.0000 on all 17 ACE sidecars, `eff. Îµ` 0.00040/0.00102/0.00175 and the 41/54/62 s timings straight from `runs/ace_sweep.log`.

**The headline is exactly right and is the strongest sentence in the document.** risk@cov.5 = 5.77% clean â 33.78% at ACE eps=0.005, against a full-coverage error rate of 16.89% â ratio exactly 2.00. AUROC(failure) 0.7703 â 0.0002456.

**The threat-model table is exact and is the best-supported scientific claim here.** Policy refit from scratch (T = 1.278456, t_low = 0.092901, t_high = 0.888072, calib residual 4.9451% â "4.95%"): clean residual 4.652% at coverage 43.23%; ace_uint8 eps=0.002/0.005 residual **33.057%/33.061%** at coverage **50.37%/50.36%** â coverage does rise; `eps_effective_max` = 1.000015 grey levels, so "one grey level" is the correct Lâ description; predicted-label vectors elementwise identical to clean (12,729/15,316 = 0.83109166884304 in all three). One sentence would make it *sharper*, not softer: only 3,889 of 15,316 images (25.4%) actually carry the change â **a quarter of the traffic perturbed by one grey level moves residual risk 4.65% â 33.06% and pushes coverage UP.**

**The WP4 table is exact:** 0.8311/43.2%/56.8%/4.65%/2.34%; 0.7875/33.6%/66.4%/7.45%/3.98%; 0.6762/58.2%/41.8%/22.46%/0.18% (25.95% false flags); 0.8311/17.1%/82.9%/98.13%/17.16%. And "4.95% on calib, 4.65% on test â it generalizes" is real.

**The D4 narrative survives its own broken cells:** 100.00% â 12.97% (states as 13%); 19.18% â 8.05%; review 27.10% â 50.07%; JPEG q50 residual 11.166% â 4.743%, under the SLA. The framing â the price is a moderation-staffing cost, not an accuracy cost â is the most deployment-legible claim in the document.

**The CLIP results table is exact:** 0.9289 / 0.9285 / 0.9801 / 0.8898 / 11.83e-3 / 0.0050 / 1.98%, n = 15,316. Breakouts exact: synthetic 0.9518/0.9317/0.9962/0.9215/5.47/0.79%, tampered 0.9095/0.8638/0.9640/0.8612/19.63/3.56%. 19.63/5.47 = 3.59 â "3.6Ã" is right.

**The leakage control is the most defensible part of the design and it fully checks out.** manifest_v1: 26,756 rows, uid unique, fit 6,756 all `train:`, calib 4,684 and test 15,316 all `validation:`; test = 7,658 / 3,829 / 3,829 exactly 50/50 with fakes split evenly; 230 img_id collisions inside the manifest (119 label-1, 111 label-2), zero among reals. calib and test uid **sequences** are byte-identical between v1 and v2, which is what makes the WP3 comparison legitimate. `manifest_v1.parquet` sha256 `f95e42e2â¦` is the `manifest_sha` in every sidecar.

**The trivial baselines are exact and one of them is stronger than stated.** `width == height â fake` = **0.9785191956124314** (14,987/15,316), independently rederived from the raw parquet, matching `runs/trivial_baselines.json` to all 16 digits. `PNG â fake` = **0.7673021676677984**. And the omitted fact is the strongest one available: **0% of real images are PNG** (9,997 JPEG + 3 MPO across the full validation split), so on the real-vs-fully-synthetic subset (n = 11,487) that one-line rule is a **perfect** classifier â accuracy 1.0000. "Format alone perfectly solves 2 of the 3 classes" is the sentence that lands.

**The metrics conventions are all correct, and several are exact rather than approximate.**
- Block-weighted AURC/AUGRC **equals** the textbook mean-over-k on deduplicated data â difference exactly 0.0 to 15 digits on CLIP (n=15,268) and EffB0 (n=15,264).
- `generalized == coverage Â· selective` to max abs error 5.551e-17 over all 44 npz.
- Random-ranker anchors hold: at n=15,316, r=0.1689 â AURC 0.1687, AUGRC 0.0844 (ratio 0.4997). Exact expectations are E[AURC]=r, E[AUGRC]=r(n+1)/(2n) â worth tightening the docstring, since at n=1000 the difference (2.5e-4) exceeds several effects in the D3 table.
- The E-AURC empirical-oracle policy is right, and the gap to the closed form is exactly r/(2n).
- ECE: binary MSP â [0.5,1] leaves exactly 7 of 15 equal-width bins empty; floor 0.0112 at n=5000 â "~0.01" is fair.
- **The temperature theorem itself is correct and holds bit-exactly where the contract holds.** CLIP clean: AURC 0.0118269301, AUGRC 0.0098105355, AUROC(fail) 0.8897681 at T â {0.5, 1, 2, 5, 10} â spread exactly 0.0, not merely small. EffB0 identically. It is the numerical *bound* that is wrong, not the mathematics.

**Hygiene is clean.** `python -m src.verify` exits 0 (with the four honest cached-feature warnings). 19 tests pass, no GPU/data/network. Zero bulk artifacts tracked by git (5 `.pt`, 44 `.npz`, 3 `.parquet` all correctly ignored). `notes/` â including `proposal_extracted.txt`, the PIs' unpublished proposal â and `BUILD.md` are fully untracked, with the reasons written into `.gitignore`. The PGD step rule `alpha = 2.5*eps/steps` is recorded in every sidecar with `alpha_rule` spelled out, and the claimed 0.0002 delta from the convention change is verifiable from the stale-vs-current artifacts (0.7190 â 0.7192). `src/demo.py` genuinely serves stdlib `ThreadingHTTPServer` on :8471 and its sample pool is genuinely filtered to `sp == 'test'` â and its own HTML carries the validation caveat that the README's tables do not.

---

**Two things I could not check under the no-GPU constraint, marked UNVERIFIABLE rather than run:** the AutoAttack 2-class `IndexError` (no committed artifact; documented only in a docstring) and the `~8 min on a 3090` run_all timing. Also unrecoverable without the GPU: the exact fraction of *unquantised* ACE perturbations that would survive rounding â `src/run_attacks.py:126` stores only the mean and max of the per-sample array and discards the array. The ladder structure plus the stored mean bounds it at 2.3%â51.1% (eps=0.002) and 13.5%â70.2% (eps=0.005); both exclude zero, which is enough for Â§1.7, and the quantised runs give the exact realisable figure (25.4%). **Cheap fix worth making before the camp:** persist the per-sample effective-epsilon array in the npz (one float32 column, ~60 KB). It costs nothing, it lets anyone verify the threat-model claim without a GPU, and it lets you show a histogram instead of a mean that hides the tail doing all the work.",
    "failing_count": 44,
    "families": [
      "WP1 metrics claims (temperature invariance, AURC convention, E-AURC, random-ranker anchors, n_op)",
      "dataset shortcut + geometry control (README lines 102â131, src/baselines.py, src/features.py::apply_condition)",
      "Confidence-attack (ACE) effective-epsilon and quantisation threat-model claims â README.md lines 41-48 and 134-157, checked against runs/predictions/*.npz + *.json, src/attack_suite.py, src/metrics.py, src/moderation.py. No GPU used: all numbers recomputed with numpy/sklearn/scipy from the stored logits and sidecars.",
      "WP3/D3 comparative table + WP4/D4 moderation table (README.md lines 244-305), incl. the shared-initialisation/budget-parity claim, the Ledda et al. non-reproduction claim, and the PGD zero-accuracy / nan-AUROC claims",
      "Reproducibility and self-description: does the repo run as advertised, is it honest about what it is, and does every README number have an artifact behind it"
    ]
  },
  "workflowProgress": [
    {
      "type": "workflow_phase",
      "index": 1,
      "title": "Audit"
    },
    {
      "type": "workflow_phase",
      "index": 2,
      "title": "Synthesize"
    },
    {
      "type": "workflow_agent",
      "index": 1,
      "label": "audit:metrics",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "ad4f9779642691d90",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787514012076,
      "queuedAt": 1787514007742,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "WP1 metrics claims (temperature invariance, AURC conventionâ¦",
      "promptPreview": "You are auditing claims that will be presented at a research camp (TReC 2026, Lausanne,
24 Aug - 4 Sep) to two trustworthy-AI researchers who specialise in adversarial robustness and
uncertainty. Every claim in /home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/wp1/README.md must be checked against the artifacts and code
actually on disk. Assume the author is capable and wrong somewhere.

*** â¦",
      "lastProgressAt": 1787514485794,
      "tokens": 132523,
      "toolCalls": 21,
      "durationMs": 473268,
      "resultPreview": "{"family":"WP1 metrics claims (temperature invariance, AURC convention, E-AURC, random-ranker anchors, n_op)","verdict":"SOME_FAIL","checks":[{"claim":"\"temperature cannot reorder samples â it slides the operating point along a fixed curve\" / metrics.py: \"It provably cannot move binary AURC, AUGRC or AUROC(failure)\"","status":"HOLDS","recomputed":"Bit-identical (max-min spread exactly 0.0) acrâ¦"
    },
    {
      "type": "workflow_agent",
      "index": 2,
      "label": "audit:dataset-shortcut",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "a6af7757ad7aafdb5",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787514012009,
      "queuedAt": 1787514007742,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "dataset shortcut + geometry control (README lines 102â131, â¦",
      "promptPreview": "You are auditing claims that will be presented at a research camp (TReC 2026, Lausanne,
24 Aug - 4 Sep) to two trustworthy-AI researchers who specialise in adversarial robustness and
uncertainty. Every claim in /home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/wp1/README.md must be checked against the artifacts and code
actually on disk. Assume the author is capable and wrong somewhere.

*** â¦",
      "lastProgressAt": 1787514588484,
      "tokens": 128751,
      "toolCalls": 35,
      "durationMs": 575957,
      "resultPreview": "{"family":"dataset shortcut + geometry control (README lines 102â131, src/baselines.py, src/features.py::apply_condition)","verdict":"SOME_FAIL","checks":[{"claim":"`width == height` -> fake (no model) | **0.9785** â \"Measured on the exact test split (n=15,316, 50/50 prior)\"","status":"HOLDS","recomputed":"0.9785191956124314 (14,987 / 15,316 correct). Identical to runs/trivial_baselines.json to â¦"
    },
    {
      "type": "workflow_agent",
      "index": 3,
      "label": "audit:threat-model",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "aa1f28d1540616bb6",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787514012269,
      "queuedAt": 1787514007742,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "Confidence-attack (ACE) effective-epsilon and quantisation â¦",
      "promptPreview": "You are auditing claims that will be presented at a research camp (TReC 2026, Lausanne,
24 Aug - 4 Sep) to two trustworthy-AI researchers who specialise in adversarial robustness and
uncertainty. Every claim in /home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/wp1/README.md must be checked against the artifacts and code
actually on disk. Assume the author is capable and wrong somewhere.

*** â¦",
      "lastProgressAt": 1787514499221,
      "tokens": 125533,
      "toolCalls": 22,
      "durationMs": 486695,
      "resultPreview": "{"family":"Confidence-attack (ACE) effective-epsilon and quantisation threat-model claims â README.md lines 41-48 and 134-157, checked against runs/predictions/*.npz + *.json, src/attack_suite.py, src/metrics.py, src/moderation.py. No GPU used: all numbers recomputed with numpy/sklearn/scipy from the stored logits and sidecars.","verdict":"SOME_FAIL","checks":[{"claim":"\"Unquantised ACE produces â¦"
    },
    {
      "type": "workflow_agent",
      "index": 4,
      "label": "audit:defense-results",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "a069676c3857299f7",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787514011398,
      "queuedAt": 1787514007742,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "WP3/D3 comparative table + WP4/D4 moderation table (README.â¦",
      "promptPreview": "You are auditing claims that will be presented at a research camp (TReC 2026, Lausanne,
24 Aug - 4 Sep) to two trustworthy-AI researchers who specialise in adversarial robustness and
uncertainty. Every claim in /home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/wp1/README.md must be checked against the artifacts and code
actually on disk. Assume the author is capable and wrong somewhere.

*** â¦",
      "lastProgressAt": 1787514665577,
      "tokens": 141637,
      "toolCalls": 24,
      "durationMs": 654179,
      "resultPreview": "{"family":"WP3/D3 comparative table + WP4/D4 moderation table (README.md lines 244-305), incl. the shared-initialisation/budget-parity claim, the Ledda et al. non-reproduction claim, and the PGD zero-accuracy / nan-AUROC claims","checks":[{"claim":"The full D3 table, all 18 cells: `| clean | 0.8435 / 0.7945 / 52.5 | 0.8177 / 0.8006 / 62.3 | 0.7877 / 0.7761 / 82.5 |` ... through `| adversarial: oveâ¦"
    },
    {
      "type": "workflow_agent",
      "index": 5,
      "label": "audit:reproducibility",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "afbe814f1898b6789",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787514011725,
      "queuedAt": 1787514007742,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "Reproducibility and self-description: does the repo run as â¦",
      "promptPreview": "You are auditing claims that will be presented at a research camp (TReC 2026, Lausanne,
24 Aug - 4 Sep) to two trustworthy-AI researchers who specialise in adversarial robustness and
uncertainty. Every claim in /home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/wp1/README.md must be checked against the artifacts and code
actually on disk. Assume the author is capable and wrong somewhere.

*** â¦",
      "lastProgressAt": 1787515037631,
      "tokens": 198658,
      "toolCalls": 62,
      "durationMs": 1025105,
      "resultPreview": "{"family":"Reproducibility and self-description: does the repo run as advertised, is it honest about what it is, and does every README number have an artifact behind it","verdict":"SOME_FAIL","checks":[{"claim":"`PYTHONPATH=. .venv/bin/python -m src.verify` â 'artifact self-consistency check; run it before believing any table'","status":"HOLDS","recomputed":"Exit 0. Prints 4 warnings then: 'OK â eâ¦"
    },
    {
      "type": "workflow_agent",
      "index": 6,
      "label": "synthesize:corrections",
      "phaseIndex": 2,
      "phaseTitle": "Synthesize",
      "agentId": "ad1b2a751d6e9e9d5",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787515041804,
      "queuedAt": 1787515039477,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "cat > /tmp/claude-1000/-home-samuel-renteria-Desktop/29f64aâ¦",
      "promptPreview": "You are auditing claims that will be presented at a research camp (TReC 2026, Lausanne,
24 Aug - 4 Sep) to two trustworthy-AI researchers who specialise in adversarial robustness and
uncertainty. Every claim in /home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/wp1/README.md must be checked against the artifacts and code
actually on disk. Assume the author is capable and wrong somewhere.

*** â¦",
      "lastProgressAt": 1787515871948,
      "tokens": 195706,
      "toolCalls": 47,
      "durationMs": 830143,
      "resultPreview": "# TrustFake WP1 README â correction list, verified against artifacts on disk

Method: every number recomputed from `runs/predictions/*.npz` with `src/metrics.py` and `src/moderation.py` in float64, CPU only. GPU untouched. Where the five audits disagreed I recomputed and say which reading I believe. Scratch scripts: `/tmp/claude-1000/-home-samuel-renteria-Desktop/29f64a6d-3439-4bcc-8e45-e43e6291bcâ¦"
    }
  ],
  "totalTokens": 922808,
  "totalToolCalls": 211
