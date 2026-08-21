# TrustFake WP1 — evaluation harness

Data-agnostic scoring harness for selective deepfake detection: risk-coverage, AURC/AUGRC,
failure prediction, calibration, and confidence attacks. Built pre-camp (19 Aug 2026) on
real SID-Set images.

Everything below was produced on this box and is reproducible with
`HF_HUB_OFFLINE=1 python -m src.run_all`.

## The interface — publish this on Day 1

```
runs/predictions/predictions_{model}_{split}_{condition}.npz
    uid     str          THE key: "<source_split>:<img_id>"  -- join on this
    img_id  str          dataset id, traceability only (NOT unique, see below)
    y       int8         0 = real, 1 = fake (synthetic or tampered)
    label   int8         original 3-class label, so results break out by difficulty
    logits  float32[n,2] [0, margin];  softmax -> p(fake) = sigmoid(margin)
+ JSON sidecar {model_id, manifest_sha, condition, split, seed, timestamp, ...}
```

`condition` is required from line one. WP2 and WP3 hand over arrays from attacked or
robustly-trained models and WP1 scores them without touching their code. `metrics.py`
consumes `(scores, labels)` and nothing else, so a second dataset is a data adapter, not a
rebuild.

## Results — SID-Set validation slice, n = 15,316

Detector: frozen `vit_large_patch14_clip_224.openai` + linear probe (UniversalFakeDetect
recipe). Temperature fitted on the clean calibration split and frozen.

|                    | acc | F1 | AUROC(det) | AUROC(fail) | AURC ×10⁻³ | ECE | risk@cov.8 |
|---|---|---|---|---|---|---|---|
| clean              | 0.9289 | 0.9285 | 0.9801 | 0.8898 | 11.83 | 0.0050 | 1.98% |
| real vs synthetic  | 0.9518 | 0.9317 | 0.9962 | 0.9215 | 5.47  | | 0.79% |
| real vs tampered   | 0.9095 | 0.8638 | 0.9640 | 0.8612 | 19.63 | | 3.56% |

The clean risk-coverage curve is **not** degenerate, and tampered (local edits) is 3.6×
harder than fully-synthetic by AURC. That is where the interesting curve lives.

### The confidence attack — ACE on `tf_efficientnet_b0.ns_jft_in1k` + linear head

|              | acc | AUROC(fail) | AURC ×10⁻³ | AUGRC ×10⁻³ | ECE | risk@cov.5 | eff. ε |
|---|---|---|---|---|---|---|---|
| clean        | **0.8311** | 0.7703 | 67.05  | 46.52  | 0.008 | 5.77%  | — |
| ACE ε=0.0005 | **0.8311** | 0.1295 | 353.83 | 136.47 | 0.326 | 32.76% | 0.00040 |
| ACE ε=0.002  | **0.8311** | 0.0030 | 466.86 | 154.23 | 0.473 | 33.77% | 0.00102 |
| ACE ε=0.005  | **0.8311** | 0.0002 | 469.14 | 154.63 | 0.488 | 33.78% | 0.00175 |

Accuracy is identical to 4 dp in every row and label preservation is exactly 1.0000. That
is the regression check: the whole point of ACE is that accuracy does not move, so an
accuracy-based test proves nothing, and without accuracy printed beside AURC the claim is
unfalsifiable (PGD also wrecks AURC — by destroying accuracy).

**Headline.** Selective risk at 50% coverage goes 5.77% → 33.78% against a full-coverage
error rate of 16.89%. Abstaining on the half of the traffic the model is least sure about
leaves you with **2.0× the error you would have had by not abstaining at all.** AUROC of
failure prediction goes 0.7703 → 0.0002: confidence now ranks the model's mistakes above
its correct answers, near-perfectly inverted. `runs/fig2_confidence_hist.png` shows the two
distributions swapping places.

**And calibration does not help.** Every attacked row is bit-identical between T = 1.2785
and T = 1 on AURC, AUGRC, AUROC(failure) and risk@coverage, while ECE moves 0.33 → 0.47.
For a 2-class model MSP = σ(|z₁−z₀|/T) is strictly monotone in the logit margin, so
temperature cannot reorder samples — it slides the operating point along a fixed curve.
This is `tests/test_metrics.py::test_temperature_cannot_change_binary_selective_metrics`,
an assert rather than a citation. Bound: it is exact down to T = 0.5; below T ≈ 0.2 float64
softmax saturates MSP to exactly 1.0 and manufactures tie blocks, at which point AURC moves
by ~4e-3 — the representation degrades, not the ordering. Hence `n_op` (distinct confidence
values) is printed beside AURC in every table.

## Two dataset findings that fail silently

**1. `img_id` is not a key.** 8,634 img_ids appear in BOTH the train and validation splits —
4,496 synthetic and 4,138 tampered — because those classes are numbered sequentially and the
counter restarts per split. `full_synthetic_000155` exists in each, with different bytes
(verified: 0/6 byte-identical on inspection). The real class is content-hash named and has
**zero** collisions, so spot-checking reals hides the problem entirely. A flat feature cache
keyed on `img_id` silently overwrites fit images with test images; a join between two npz
files silently merges wrong rows. The key is `uid = "<source_split>:<img_id>"`. This
manifest alone contains 230 such collisions.

**2. The accept-check forward must be the reported forward.** ACE accepts a step only when
the argmax is unchanged, so label preservation is 1.0000 by construction — but re-running
the model on `x_adv` afterwards uses a different batch shape, and EfficientNet under cuDNN
is not bit-identical across batch shapes. A sample on the decision boundary then flips and
preservation reads 0.9980 for a reason that has nothing to do with the attack. `ace()`
returns the logits from the accept-check forward.

## Leakage control

Structural, not disciplinary (`src/manifest.py`):

- **fit** rows come only from `train-*` shards; **calib** and **test** only from
  `validation-*` shards, and calib and test never share a shard.
- The probe is fitted on fit, the temperature on calib, everything is reported on test.
  Fitting temperature on the reported split is impossible without editing the manifest.
- 50/50 binary prior by construction: all real rows kept, fakes sampled to match, half
  label-1 and half label-2. The policy is in the sidecar because the base rate sets what
  accuracy and F1 mean.

## The dataset has a trivial shortcut — and we verified we are not using it

Measured on the exact test split (n=15,316, 50/50 prior):

| rule | accuracy |
|---|---|
| `width == height` -> fake (no model) | **0.9785** |
| CLIP ViT-L/14 probe | 0.9289 |
| EfficientNet-B0 probe | 0.8311 |
| original format PNG -> fake | 0.7673 |
| majority class | 0.5000 |

Label 1 (fully synthetic) is 100% square 1024x1024 PNG, label 2 (tampered) is 100% square, and
only ~5% of real images are square. A one-line metadata rule with no learning in it beats every
model here. **On SID-Set, an accuracy is not evidence of forensic capability until it is read
against 0.9785, not against 0.5.** (Grommelt et al., "Fake or JPEG? Revealing Common Biases in
Generated Image Detection Datasets".) `src/baselines.py` computes this and every table prints it.

Our models do not read geometry — images arrive as 224x224 centre crops — but the cache
re-encodes everything to JPEG q95, which turns PNG-vs-JPEG provenance into a single-vs-double
compression trace a CNN could learn. So the shortcut is laundered, not removed, and the question
had to be settled by measurement rather than argument:

| CLIP ViT-L/14 probe | accuracy | AUROC(det) | AURC x10^-3 |
|---|---|---|---|
| uncontrolled | 0.9289 | 0.9801 | 11.83 |
| **geometry-controlled** (`squarecrop`, fit AND eval) | **0.9291** | 0.9802 | 11.75 |

Unchanged. The detector was never riding the artifact. Run any condition under `squarecrop` to
reproduce; `tests/test_attacks.py` locks the property that the control actually removes geometry.

## Two threat models for the confidence attack, and only one is realisable

Unquantised ACE produces mean perturbations of **0.102 / 0.261 / 0.447 grey levels** at
eps = 0.0005 / 0.002 / 0.005 — below one 8-bit quantisation step (0.5/255 = 0.00196). The decode
cache is uint8 and `ToTensor` puts x exactly on the k/255 grid, so rounding a sub-step
perturbation back to uint8 erases it. Unquantised ACE therefore describes an attacker with
post-decode **tensor** access; `ace_uint8` describes one who can only upload a **file**, which is
the moderation threat model that matters. (This is why integer-constrained attacks exist in
forensics: Tondi, Electronics Letters 54(21), 2018.)

| condition | acc | AUROC(fail) | AURC x10^-3 | residual risk |
|---|---|---|---|---|
| clean | 0.8311 | 0.7703 | 67.05 | 4.65% |
| ACE eps=0.005, float32 (tensor access) | 0.8311 | 0.0002 | 469.14 | 98.13% |
| ACE eps=0.0005, **uint8** | 0.8311 | 0.7703 | 67.05 | 4.65% |
| ACE eps=0.002-0.005, **uint8** | 0.8311 | 0.0471 | 433.87 | **33.06%** |

The sub-step row vanishes exactly as predicted, which is the control. The realisable row is the
result: a **one-grey-level** perturbation, invisible to a human and survivable in a saved PNG,
takes a policy tuned to a 5% residual-risk SLA to **33% wrong auto-decisions at bit-identical
accuracy** — and coverage RISES to 50.4%, so the system auto-decides more while being far more
wrong. That is Carlini & Farid's least-significant-bit result (CVPRW 2020, arXiv:2004.00622)
moved from the prediction axis to the confidence axis.

Report which threat model a row belongs to. Never quote the float32 number alone.

## Honest limitations — state these on every slide

- **This is not the SID-Set test set.** The official test split is withheld by the authors
  to prevent leakage. Everything here is carved from the **validation** split and labelled
  `SID-Set validation split, held-out slice`. If the authors grant the test set, re-running
  the manifest against it is ~20 minutes.
- **Tampered images cannot be deduplicated against reals.** They derive from
  COCO/OpenImages/Flickr30k and the dataset exposes no key linking a tampered image to its
  source, so near-duplicate *content* can straddle any split and no split rule can prevent
  it. This is a property of the dataset, not of the protocol.
- **The 256px JPEG q95 cache is lossy** and unsuitable for frequency-domain detectors that
  key on compression artefacts. The raw parquet is retained for that reason.
- **The probes are not tuned detectors.** A frozen backbone + linear head is the reference
  baseline, deliberately: WP3's robust training is what the comparison is for.
- Single split, single seed. No bootstrap CIs yet.

## WP2 — attack package (D2)

`src/attack_suite.py` holds seven attacks in two families, and the family distinction is the
scientific spine: a prediction-targeted attack collapses AURC *as a side effect* of destroying
accuracy, while a confidence-targeted attack collapses it at accuracy that does not move.

| attack | family | direction | uses labels |
|---|---|---|---|
| fgsm, pgd_linf, pgd_l2, autoattack | prediction | n/a | yes |
| ace | confidence | both (down where correct, up where wrong) | **yes** |
| overconf (Ledda et al. 2025) | confidence | over_confidence | **no** |
| underconf | confidence | under_confidence | no |

`overconf` minimizes H(f(x+d), onehot(yhat)) with yhat frozen at the clean prediction: label
preservation is exact *by construction*, not by an accept test, and it needs no ground truth —
which is ACE's practical weakness. `underconf` is the mirror and is NOT label-preserving, so the
runner measures preservation rather than assuming it.

`frac_perturbed` is recorded per run because AutoAttack and APGD write back **only the samples
they flipped**, so their output is a clean/adversarial mixture. Robust accuracy stays meaningful
on a mixture; AURC and ECE do not.

## WP3 — defense package (D3)

Standard / PGD-AT / TRADES on a fixed backbone, fixed data protocol, fixed evaluation, matched on
**optimizer steps** (not wall clock — PGD-k costs k+1 forwards per step, so matching wall clock
would hand the clean arm ~8x the weight updates and the comparison would measure budget). Model
selection uses a held-out slice of *fit*; calib and test are never seen.

**The epsilon finding.** PGD-AT at the standard ImageNet budget eps=8/255 COLLAPSED: loss pinned
at ln 2, clean and robust accuracy both at the majority-class rate, constant output
(`runs/train_wp3_eps8_255_collapsed.log`). Deepfake evidence is small-amplitude high-frequency
residue — generator fingerprints, resampling traces, blending seams — and an 8/255 ball is wide
enough to erase it, so the robust-optimal classifier inside that ball really can be a constant.
At a forensic budget the same code trains fine:

| | fit_val clean | fit_val robust (PGD-10) |
|---|---|---|
| PGD-AT eps=8/255 | 0.4943 (collapsed) | 0.4943 |
| PGD-AT eps=2/255, 4-epoch warm-up | 0.8251 | 0.7359 |

Robustness budgets transplanted from object recognition, where the signal is large-amplitude
semantic structure, do not transfer to forensics. Report at forensic epsilons and say why.

## WP4 — selective moderation (D4)

Two thresholds on p(fake): below t_low ALLOW, above t_high FLAG, between them HUMAN REVIEW.
Fitted on the clean calibration split by minimizing review rate subject to a residual-risk SLA,
then frozen and applied unchanged to every condition. A single confidence threshold is the
symmetric special case, and moderation costs are not symmetric.

| condition | acc | coverage | review | residual risk | missed fakes |
|---|---|---|---|---|---|
| clean | 0.8311 | 43.2% | 56.8% | **4.65%** | 2.34% |
| jpeg_q50 | 0.7875 | 33.6% | 66.4% | 7.45% | 3.98% |
| downscale_0.5 | 0.6762 | 58.2% | 41.8% | 22.46% | 0.18% (25.95% false flags) |
| ACE eps=0.005 | **0.8311** | 17.1% | 82.9% | **98.13%** | 17.16% |

A policy tuned to a 5% SLA on clean data (4.95% on calib, 4.65% on test — it generalizes) ships
**98% wrong auto-decisions** under a confidence attack at bit-identical accuracy. No accuracy
monitor would fire. That is the project's thesis in one row.

An empty auto-decide zone reports residual risk as **NaN, never 0.0**: a policy that refuses to
work has no residual risk to measure, and 0.0 makes total refusal look like perfect safety.

`src/demo.py` serves it on stdlib http.server (nothing else is installed and nothing else will
be): image, label, p(fake), MSP confidence, decision band, verdict, plus sample buttons that draw
from the **test** split only and reveal ground truth after the verdict.

## WP3 / D3 — comparative results

All three arms share one initialisation (`phase1_init.pt`) and one 12-epoch schedule on an
identical data protocol and evaluation, so a difference between rows is a difference of METHOD,
not of budget. Each model's temperature is fitted on ITS OWN clean calib split and frozen before
any attacked row is scored. Cells: **accuracy | AUROC(failure) | AURC x10^-3**.

| condition | standard | PGD-AT (2/255) | TRADES (2/255) |
|---|---|---|---|
| clean | 0.8435 / 0.7945 / 52.5 | 0.8177 / 0.8006 / 62.3 | 0.7877 / 0.7761 / 82.5 |
| realistic: JPEG q50 | 0.7198 / 0.7188 / 140.4 | 0.8169 / 0.7997 / 62.8 | 0.7876 / 0.7730 / 83.4 |
| realistic: 0.5x resize | 0.7987 / 0.7793 / 75.7 | 0.7857 / 0.7853 / 80.6 | 0.7649 / 0.7586 / 100.9 |
| adversarial: PGD 2/255 (label) | **0.0000** / nan / 1000.0 | 0.7192 / 0.7054 / 146.8 | 0.7062 / 0.6799 / 170.3 |
| adversarial: ACE uint8 (confidence) | 0.8435 / **0.1093** / 329.5 | 0.8177 / 0.6659 / 94.5 | 0.7877 / 0.6541 / 117.9 |
| adversarial: over-confidence (label-free) | 0.8435 / 0.4135 / 194.2 | 0.8177 / 0.7770 / 68.8 | 0.7877 / **0.7789** / 81.3 |

PGD uses the standard step rule `alpha = 2.5*eps/steps` (Madry; the RobustBench convention),
recorded in every sidecar so it can never be inferred wrongly from the epsilon. The earlier
default, `max(eps/4, 1/255)`, was a *stronger* attack — alpha = eps/2 crosses the ball in two
steps — so it understated the defenses rather than flattering them. Re-running under the
convention moved robust accuracy by 0.0002: the conclusions do not depend on the step rule.

Fitted temperatures: standard **3.338**, PGD-AT 0.903, TRADES 0.757. The undefended model needs
heavy smoothing; the defended ones are close to calibrated already.

Four results, and the third contradicts the hypothesis we set out to test:

1. **Robust training works on the label axis.** PGD at 2/255 takes the standard model to
   accuracy **0.0000** — AUROC(failure) is `nan` because there are no correct predictions left
   to rank, which is the empty-case guard firing rather than a bug. Both defenses hold ~0.71.
2. **It costs clean accuracy**: -2.6 pp for PGD-AT, -5.6 pp for TRADES, and the standard model
   has the best clean AURC (52.5) of the three. The trade-off is real and should be stated.
3. **It also protects CONFIDENCE, which we did not expect.** Under the confidence attack,
   AUROC(failure) collapses to 0.109 undefended but holds at 0.666 / 0.654 with AT/TRADES. Under
   the label-free over-confidence attack the defended models barely move at all
   (0.801 -> 0.777, 0.776 -> 0.779). The Ledda et al. hypothesis — that adversarial training
   protects against under-confidence but not over-confidence attacks, because the
   entropy-minimising region does not lie on the decision boundary — does **not** reproduce in
   this setting. That is a result worth putting to the PIs rather than filing away.
4. **Realistic robustness comes along for free**: JPEG q50 costs the standard model 12 pp of
   accuracy and costs PGD-AT essentially nothing (0.8177 -> 0.8169).

## WP4 / D4 — the same three models through the moderation layer

Policy fitted per model on its own clean calib at a 5% residual-risk SLA, then frozen.

| model | clean residual risk | review rate | PGD | ACE uint8 | over-confidence |
|---|---|---|---|---|---|
| standard | 5.47% | 27.1% | **100.00%** | 19.18% | 15.65% (coverage -> **100%**) |
| PGD-AT | 4.72% | 50.1% | 13.01% | 8.05% | 12.63% |
| TRADES | 4.98% | 55.4% | 13.53% | 8.47% | 12.89% |

The deployment reading: adversarial training takes residual risk under a label attack from
**100% to 13%** and under a confidence attack from 19.2% to 8.1%, and it holds the SLA under
JPEG q50 (11.17% -> 4.74%). **The price is the review bill roughly doubling**, 27% -> 50%, which
is a moderation-staffing cost, not an accuracy cost, and it belongs in the same sentence as the
safety gain.

The residual threat is the **label-free over-confidence attack**: it is the only one that drives
coverage UP (to 100% undefended, 80.8% with AT). It does not make the system abstain — it stops
the system abstaining, by making it confidently wrong. It also needs no ground-truth labels, so
it is the one a real attacker can actually run.

## Batch size is part of the experimental condition

EfficientNet under cuDNN is not bit-identical across batch shapes. Regenerating the clean
baseline at batch 16 instead of 32 moved accuracy by one sample in 15,316 — enough to break a
headline that reads "accuracy identical across every row" for a reason having nothing to do with
any attack. Every prediction sidecar records `batch`, and `score.py` prints a loud warning when
rows being compared were produced at different ones.

The same effect is why `ace()` returns the logits from its accept-check forward rather than
re-running the model on x_adv.

## Layout

```
src/metrics.py    the deliverable: rc_curve, aurc, augrc, eaurc, auroc_{detection,failure},
                  ece, nll, brier, fit_temperature, risk_at_coverage, coverage_at_risk
src/models.py     one constructor: "probe:<timm_id>" | "ckpt:<path>" -> [0,1] -> logits[n,2]
src/attack_suite.py  7 attacks, direction taxonomy, condition grammar (k_255 is exact)
src/run_attacks.py   the {model x attack x eps} matrix -> npz
src/train.py      standard / PGD-AT / TRADES, matched on optimizer steps, eps warm-up
src/moderation.py WP4 policy: fit on clean calib, freeze, evaluate everywhere
src/demo.py       stdlib demonstrator on :8471
src/manifest.py   leakage-free split manifest (uid key, 50/50 balance, shard-disjoint)
src/decode.py     parquet -> flat JPEG cache (short side 256, q95), 4 workers, resumable
src/features.py   frozen backbone features + the perturbation ladder (jpeg_q*, downscale_*, webp)
src/probe.py      linear probe -> predictions npz + sidecar
src/attacks.py    NormalizedModel (normalization inside forward) + ACE + condition runner
src/score.py      the condition table, temperature frozen on calib
src/figures.py    fig1 risk-coverage, fig2 confidence histograms
src/run_all.py    driver; --smoke is the offline pre-flight
src/baselines.py  trivial metadata baselines; the number every accuracy is read against
src/compare.py    the D3 comparative table across defenses x conditions
src/verify.py     artifact self-consistency check; run it before believing any table
tests/            19 tests (9 metrics + 10 attacks/hardening), no GPU, no data, no network, <2 s
```

## Run

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/ -q
HF_HUB_OFFLINE=1 .venv/bin/python -m src.run_all --smoke     # offline pre-flight
HF_HUB_OFFLINE=1 .venv/bin/python -m src.run_all             # full, ~8 min on a 3090
```

Measured on this box: decode 214 img/s (4 workers), CLIP L/14 features 232 img/s,
EfficientNet-B0 features 784 img/s, ACE over 15,316 images 41–62 s per epsilon.

## Conventions that must travel with the numbers

- selective risk = `Σ loss·[g≥τ] / Σ[g≥τ]`; generalized risk = `Σ loss·[g≥τ] / n`;
  `generalized == coverage · selective` elementwise (asserted).
- AURC/AUGRC = **block-size-weighted mean over distinct operating points**. Tie blocks are
  collapsed: a threshold inside a tie block selects a permutation-dependent set. Naive
  cumulative AURC varies 5.6e-3 across permutations with 10 distinct confidence levels.
  Trapezoid-over-coverage is a third convention in the literature and is not used here.
- E-AURC uses the **empirical** oracle at the same n, never the asymptotic closed form
  `r + (1−r)ln(1−r)` (9.65e-05 off at n=1000 — the size of the effects being reported).
- AUROC is always two numbers: detection (p_fake vs y) and failure (conf vs correct).
- ECE is reported with its binning scheme and domain. Binary MSP lives in [0.5, 1], so
  equal-width bins over [0,1] leave 7 of 15 empty; the floor is ~0.01 at n≈5000. NLL and
  Brier are reported alongside (unbiased, no binning).

## Known gotcha for the camp

AutoAttack's targeted stages **cannot run on a 2-class model**: targeted DLR reads
`x_sorted[:,-3]` and `[:,-4]` and needs ≥4 logits, so `apgd-t`/`APGDT` raise
`IndexError: index -3 is out of bounds`. `checks.py` warns for `n_cls<=2` and runs the
attack anyway, and `version="standard"` only crashes once points *survive* apgd-ce — so it
appears to work against a weak detector and dies against a robust one. Safe composition:
`AutoAttack(version="custom", attacks_to_run=["apgd-ce","fab-t","square"])` with
`aa.fab.n_target_classes=1`. Setting `n_classes=3` on a 2-logit model does not work around it.
