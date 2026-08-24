# The sealed holdout

**Seal:** `runs/holdout_seal.json` — sha256 `d10bc0a552012580…`, n = 3376 (50/50), 6 train
shards disjoint from the 30 fit shards. Manifest: `runs/manifest_v3_holdout.parquet` (its
fit / calib / test uids are byte-identical to `manifest_v2`, so every result already reported
still stands — the holdout is purely additive).

## Why it exists

Everything in this repo — every attack, the temperature and thresholds, the σ experiment, the
audit — was computed on the **test** split, and we iterated against those numbers repeatedly.
That is F2 (test-set peeking) at the meta level: not fitting on test, but *selecting and
framing* against it. A single number is only trustworthy if it comes from data that no
decision here has ever seen.

The official SID-Set test split is withheld by the authors, and all 34 validation shards are
consumed by calib + test. So the holdout is drawn from **unused train shards** — same
generation family (OpenImages / FLUX / latent-diffusion), never touched by the probe fit, the
temperature, any threshold, any attack, or any narrative.

## The discipline (do not break it)

1. The holdout is **never scored during development.** `run_attacks.py` refuses `--split
   holdout` unless `--unseal-holdout` is passed; `verify.py` fails if holdout predictions
   exist without an unseal flag.
2. It is evaluated **once**, at the very end, with the model and every threshold **frozen**.
3. Any model, temperature or threshold change **after** unsealing invalidates the number.
   Re-sealing requires a fresh tranche from other unused train shards.
4. Report the holdout number **as-is**, beside the 0.9785 trivial-baseline floor, with the
   train-family caveat stated.

## The one final run (when the model is frozen for the 4 Sep talk)

    python -m src.features --manifest runs/manifest_v3_holdout.parquet --splits holdout ...
    python -m src.run_attacks --split holdout --unseal-holdout --manifest runs/manifest_v3_holdout.parquet ...

Then it is spent. There is no second holdout evaluation.
