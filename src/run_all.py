"""End-to-end driver. `--smoke` is the offline pre-flight: green with the network
unplugged, or the harness is not done.

    HF_HUB_OFFLINE=1 python -m src.run_all --smoke

Smoke uses the 'smoke' manifest profile (2/1/2 shards) and a 256-image limit on the attack,
and writes EVERY artifact -- manifest, features, predictions, table -- under runs/smoke/.

That namespacing is load-bearing, not tidiness: feature and prediction files are named
{model}_{split}_{condition}, which is identical for a smoke run and a full run. Sharing a
directory means a 60-second smoke test silently overwrites the real numbers with 256-row
versions, and nothing errors. (It did, once, during the build.) The decode cache IS shared
deliberately -- it is keyed by uid, so smoke images are a subset, not a collision.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")


def step(name: str, args: list[str]) -> None:
    print(f"\n=== {name} " + "=" * (60 - len(name)))
    t0 = time.time()
    r = subprocess.run([PY, "-m"] + args, cwd=ROOT,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    if r.returncode:
        sys.exit(f"FAILED at {name}")
    print(f"--- {name}: {time.time() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        sm = ROOT / "runs" / "smoke"
        man, fdir, pdir = sm / "manifest_smoke.parquet", sm / "features", sm / "predictions"
        ns = ["--features-dir", str(fdir), "--pred-dir", str(pdir)]
        step("tests", ["pytest", "-q", "tests/"])
        step("manifest", ["src.manifest", "--profile", "smoke", "--out", str(man)])
        step("decode", ["src.decode", "--manifest", str(man)])
        step("features", ["src.features", "--manifest", str(man), "--features-dir", str(fdir)])
        step("features(target)", ["src.features", "--manifest", str(man), "--splits", "fit",
                                  "--features-dir", str(fdir),
                                  "--model", "tf_efficientnet_b0.ns_jft_in1k"])
        step("probe", ["src.probe", "--manifest", str(man)] + ns)
        step("attack", ["src.attacks", "--manifest", str(man), "--limit", "256",
                        "--conditions", "clean", "ace_eps0.002"] + ns)
        step("score", ["src.score", "--model", "tf_efficientnet_b0",
                       "--conditions", "clean", "ace_eps0.002", "--pred-dir", str(pdir),
                       "--out", str(sm / "condition_table.md")])
        print("\nSMOKE OK -- full-run artifacts untouched")
        return

    step("tests", ["pytest", "-q", "tests/"])
    step("manifest", ["src.manifest", "--profile", "full"])
    step("decode", ["src.decode"])
    step("features", ["src.features"])
    step("features(target)", ["src.features", "--splits", "fit", "--model", "tf_efficientnet_b0.ns_jft_in1k"])
    step("probe", ["src.probe"])
    step("score(clip)", ["src.score", "--out", str(ROOT / "runs" / "condition_table_clip.md")])
    step("attack", ["src.attacks"])
    step("score(target)", ["src.score", "--model", "tf_efficientnet_b0", "--conditions",
                           "clean", "ace_eps0.0005", "ace_eps0.002", "ace_eps0.005",
                           "--out", str(ROOT / "runs" / "condition_table_ace.md")])
    step("figures", ["src.figures"])


if __name__ == "__main__":
    main()
