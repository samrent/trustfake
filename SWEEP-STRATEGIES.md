# Which robustness-training combination to sweep — the case for each

You asked which *combination* of robustness training works best, on a small data slice. There
are three genuinely different sweeps we could run. This document argues each so you can pick.
The sweep infrastructure (`src/sweep.py`) is built and proven end-to-end; only the **grid** and,
for B/C, a **new training objective** depend on this choice.

**How every option is judged** (the ranking objective): not robust accuracy, but **confidence
resilience** — the config whose failure-AUROC stays highest under the confidence attacks (ACE,
over-confidence) while holding a clean-accuracy floor. Rationale: robust accuracy under
PGD/AutoAttack is the crowded label axis the camp can't out-compete; the *confidence* axis is
this project's thesis and the PIs' own research area, and it's what keeps residual risk low
under attack. Every strategy is scored on the fresh **selval** slice (never trained on); the
single winner is confirmed once on test; the sealed holdout is reserved for 4 Sep.

---

## A. Hyperparameter grid of the methods we already have

**What it is.** Sweep the knobs of standard / PGD-AT / TRADES already in `train.py`: epsilon
{1, 2, 4/255} × inner-steps {3, 7} × TRADES-beta {3, 6} (± epsilon warm-up, ± warm-start from
`phase1_init`). ~12 configs.

**What it proves.** "Given the methods that exist, which *settings* keep confidence most
trustworthy under attack?" It turns the current three arbitrary arms (one eps, one beta, one
step count) into a properly-tuned frontier — and it may well show that a *smaller* forensic
epsilon or fewer inner steps buys better confidence resilience than the defaults we happened to
pick.

**Cost.** Zero new training code. ~1–1.2 GPU-hours at a 3k slice. **Runnable right now.**

**Risk.** Finds the best of *known* methods — no novelty. But it's the honest baseline: you
can't claim a hybrid or a new defense beats tuning until you've actually tuned.

**Long-term fit.** Necessary groundwork. Cheap, defensible, and it de-risks B and C by telling
you what a well-tuned single method actually achieves.

---

## B. Stacked / hybrid defenses

**What it is.** New *combined* objectives: PGD-AT + a TRADES KL term (α·CE_adv + β·KL in one
loss), adversarial training + input purification (a JPEG/blur pre-processing defense), or MART.
~4–6 hand-picked hybrids at the best epsilon from A.

**What it proves.** "Does *layering* defenses beat any single one?" It's the natural next
question after A, and 'we combined two defenses' is an intuitive story for a talk.

**Cost.** A new loss branch in `train.py` per hybrid (moderate build). ~0.5–1 GPU-hour once
written.

**Risk — the real one.** In the literature, carefully-combined defenses frequently **do not**
beat a well-tuned single method, and several famous "stacked" defenses (especially input
purification) were later shown to fail under adaptive attack — obfuscated-gradient territory
(Athalye et al. 2018). So B is the strategy most likely to produce a result that looks good and
then dissolves when a skeptic pushes on it. If you run B, it must be evaluated adaptively, or it
becomes a liability in a room of robustness researchers.

**Long-term fit.** Medium. Novel-sounding, but the downside risk is real and the payoff is
uncertain. Best only *after* A establishes the baseline it has to beat.

---

## C. Confidence-targeted defense — the gap the whole project points at

**What it is.** Train *against the confidence attack itself*, which no current arm does. Two
shapes: (i) adversarial training where the inner maximization is the **ACE / over-confidence**
objective rather than cross-entropy — i.e. hardening the model against perturbations that move
confidence without moving the label; or (ii) a **confidence/uncertainty-regularized loss** that
penalizes confident mistakes directly. ~4–6 configs.

**What it proves.** "Can we train a detector whose *confidence* survives a confidence attack?"
Every other page of this project *diagnoses* the confidence vulnerability; C is the only
strategy that tries to *fix* it. If it works, it's the result the whole harness was built to set
up — and it lands exactly on the PIs' evidential-uncertainty turf.

**Cost.** A genuinely new training objective (highest build of the three). ~0.5–1 GPU-hour once
written.

**Risk.** Hardest to get right; must be framed carefully as non-adaptive vs adaptive (an
attacker who then optimizes against *this* defense). But even a negative result — "confidence
adversarial training does not restore failure-AUROC under an adaptive attack" — is a genuine,
publishable finding, because it's the open question the field hasn't answered.

**Long-term fit.** Highest. It's the novel contribution, the thesis-aligned one, and the
strongest single thing to hand two trustworthy-AI PIs. It also composes with the σ-seam result
you already have (an independent uncertainty that survives the attack) into one story:
*diagnose the confidence vulnerability → defend it two ways (train-time C, decision-time σ) →
measure which holds.*

---

## Recommendation

**A first, then C.** Run A now (it's free and de-risks everything by giving the tuned baseline
any new method must beat), then invest the build in C (the novel, thesis-aligned contribution).
Skip B unless A and C leave time — its downside risk is the worst and its payoff the least
certain. This ordering also sequences the build: A needs no code, C needs one new objective, and
you'd only write B's objectives if you decide it's worth it after seeing A and C.

**One decision for you:** run **A alone** now, **A then C**, or **all three** — and confirm the
slice size (default ~3k fit) and clean-accuracy floor (default 0.75). Then I define the grid and
launch it.
