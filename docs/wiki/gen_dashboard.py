#!/usr/bin/env python3
"""Regenerate the TrustFake experiment dashboard from the artifacts on disk.

Reads only files -- every metric is recomputed from runs/predictions/*.npz through the
harness's own metrics.py, so the dashboard cannot drift from the numbers the harness reports.
No GPU, no network. Run it again after any new experiment lands:

    python3 wiki/gen_dashboard.py   # writes wiki/dashboard.html

The temperature for each model is fitted on THAT model's clean calibration split, exactly as
score.py does, and frozen for its attacked rows. Rows with no calib_clean fall back to T=1
and are flagged.
"""
from __future__ import annotations

import base64
import json
import pathlib
import sys

WP1 = pathlib.Path("/home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/wp1")
sys.path.insert(0, str(WP1))
import numpy as np
from src import metrics as M
from src.attack_suite import parse_condition

PRED = WP1 / "runs" / "predictions"
RUNS = WP1 / "runs"
CKPT = RUNS / "checkpoints"
OUT = pathlib.Path(__file__).resolve().parent / "dashboard.html"

DETECTOR = {
    "tf_efficientnet_b0": "EfficientNet-B0 + linear probe",
    "vit_large_patch14_clip_224": "CLIP ViT-L/14 probe (fp16 features)",
    "vit_l14_e2e": "CLIP ViT-L/14 end-to-end (fp32, 4k sample)",
    "effb0_standard": "EffNet-B0 fine-tuned — standard (interim)",
    "effb0_standard_eps2_255": "EffNet-B0 fine-tuned — standard",
    "effb0_at_pgd_eps2_255": "EffNet-B0 fine-tuned — PGD-AT 2/255",
    "effb0_trades_eps2_255": "EffNet-B0 fine-tuned — TRADES 2/255",
}


def load(stem):
    z = np.load(PRED / f"{stem}.npz", allow_pickle=False)
    meta = json.loads((PRED / f"{stem}.json").read_text())
    return z, meta


def temp_for(model):
    calib = PRED / f"predictions_{model}_calib_clean.npz"
    if not calib.exists():
        return 1.0, False
    z, _ = load(f"predictions_{model}_calib_clean")
    return M.fit_temperature(z["logits"].astype(np.float64), z["y"].astype(int)), True


def esc(x):
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def b64img(p):
    return base64.b64encode(pathlib.Path(p).read_bytes()).decode() if pathlib.Path(p).exists() else ""


# ---------------------------------------------------------------- gather prediction rows

sidecars = sorted(PRED.glob("predictions_*_test_*.json"))
models = sorted({json.loads(j.read_text())["model_id"] for j in sidecars},
                key=lambda m: (0 if m.startswith("tf_") else 1 if m.startswith("vit") else 2, m))
temps = {m: temp_for(m) for m in models}

rows_by_model: dict[str, list] = {m: [] for m in models}
for j in sidecars:
    meta = json.loads(j.read_text())
    m = meta["model_id"]
    z, _ = load(j.stem)
    T, has_calib = temps[m]
    s = M.summary(z["logits"].astype(np.float64), z["y"].astype(int), temperature=T)
    pc = parse_condition(meta["condition"])
    rows_by_model[m].append({
        "condition": meta["condition"], "family": pc["family"], "n": meta["n"],
        "batch": meta.get("batch"), "eff_eps": meta.get("eps_effective_mean"),
        "preserved": meta.get("label_preservation"), "frac": meta.get("frac_perturbed"),
        "acc": s["accuracy"], "auroc_fail": s["auroc_failure"], "aurc": s["aurc"] * 1e3,
        "augrc": s["augrc"] * 1e3, "ece": s["ece_15_equal_mass"], "r5": s.get("risk@cov0.5"),
        "n_op": s["n_operating_points"], "T": T, "has_calib": has_calib,
    })

FAMILY_ORDER = {"clean": 0, "realistic": 1, "confidence": 2, "prediction": 3, "unknown": 9}
for m in rows_by_model:
    rows_by_model[m].sort(key=lambda r: (FAMILY_ORDER.get(r["family"], 5), r["condition"]))

n_pred = len(sidecars)
n_cond = len({json.loads(j.read_text())["condition"] for j in sidecars})

# ------------------------------------------------------------------------------ render

def cell(v, fmt="{:.4f}", nan="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return nan
    return fmt.format(v) if isinstance(v, float) else esc(v)


def fam_pill(f):
    color = {"clean": "", "realistic": "warn", "confidence": "bad", "prediction": "bad"}.get(f, "")
    return f'<span class="pill">{f}</span>'


parts = []
parts.append(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dashboard · TrustFake wiki</title>
<link rel="stylesheet" href="style.css"></head><body>
<nav>
<a href="index.html">Overview</a>
<a href="project.html">The project</a>
<a href="harness.html">The harness</a>
<a href="dataset.html">The dataset</a>
<a href="attacks.html">Attacks</a>
<a href="defenses.html">Defenses</a>
<a href="moderation.html">Moderation</a>
<a href="epistemics.html">How experiments lie</a>
<a href="implications.html">Implications</a>
<a href="pathways.html">To 4 Sep</a>
<a href="glossary.html">Glossary</a>
<a href="dashboard.html" class="here">Dashboard</a>
</nav>
<main>
<h1>Experiment dashboard</h1>
<p class="lead">Every number here is recomputed from <code>runs/predictions/*.npz</code> through
<code>src/metrics.py</code> at generation time — no copied figures. Regenerate with
<code>python3 wiki/gen_dashboard.py</code>.</p>
<div class="kv">
<b>Prediction runs</b><span>{n_pred} scored .npz files across {len(models)} detectors and {n_cond} conditions</span>
<b>Dataset</b><span>SID-Set validation slice (official test split withheld)</span>
<b>Scoring</b><span>temperature per model on its own clean calib, frozen; AURC block-weighted, tie-safe</span>
</div>""")

# trivial baseline banner
if (RUNS / "trivial_baselines.json").exists():
    b = json.loads((RUNS / "trivial_baselines.json").read_text())
    parts.append(f"""<div class="box bad"><div class="t">Read every accuracy against this floor, not against 0.5</div>
The no-model rule <code>width == height → fake</code> scores <b>{b['B0_square_is_fake']:.4f}</b> on the
test split; the decode-scale rule <b>{b.get('B0_shortside1024_is_fake', float('nan')):.4f}</b>; majority
class {b['B0_majority_class']:.4f}. A model accuracy is not evidence of forensic capability until it
clears {b['B0_square_is_fake']:.4f}. See <a href="dataset.html">the dataset page</a>.</div>""")

# per-model prediction tables
parts.append('<h2>All prediction runs, scored</h2>')
for m in models:
    T, has_calib = temps[m]
    tnote = f"T = {T:.3f}" + ("" if has_calib else " (no calib — fell back to 1.0)")
    parts.append(f'<h3>{esc(DETECTOR.get(m, m))} <span class="pill">{esc(m)}</span> <span class="pill">{tnote}</span></h3>')
    parts.append('<table><tr><th>condition</th><th>family</th><th class="num">n</th>'
                 '<th class="num">batch</th><th class="num">acc</th><th class="num">AUROC(fail)</th>'
                 '<th class="num">AURC e-3</th><th class="num">AUGRC e-3</th><th class="num">ECE</th>'
                 '<th class="num">risk@.5</th><th class="num">n_op</th><th class="num">eff ε (grey)</th>'
                 '<th class="num">preserved</th></tr>')
    for r in rows_by_model[m]:
        eff = f"{r['eff_eps']*255:.3f}" if r["eff_eps"] is not None else "—"
        pres = f"{r['preserved']:.4f}" if r["preserved"] is not None else "—"
        r5 = f"{r['r5']*100:.2f}%" if r["r5"] is not None else "—"
        parts.append(
            f'<tr><td>{esc(r["condition"])}</td><td>{fam_pill(r["family"])}</td>'
            f'<td class="num">{r["n"]}</td><td class="num">{cell(r["batch"], nan="—")}</td>'
            f'<td class="num">{r["acc"]:.4f}</td><td class="num">{cell(r["auroc_fail"])}</td>'
            f'<td class="num">{r["aurc"]:.1f}</td><td class="num">{r["augrc"]:.1f}</td>'
            f'<td class="num">{r["ece"]:.4f}</td><td class="num">{r5}</td>'
            f'<td class="num">{r["n_op"]}</td><td class="num">{eff}</td><td class="num">{pres}</td></tr>')
    parts.append('</table>')

# D3 comparative
d3 = RUNS / "d3_comparative.json"
if d3.exists():
    d = json.loads(d3.read_text())
    tv = d.get("temperatures", {})
    parts.append('<h2>D3 — defense comparison (matched budget, one shared init)</h2>')
    parts.append('<p>Cells: accuracy | AUROC(failure) | AURC×10⁻³. Same architecture, same 12-epoch '
                 'schedule; differences are of method. See <a href="defenses.html">defenses</a>.</p>')
    ms = list(tv.keys())
    parts.append('<table><tr><th>condition</th>' + "".join(f'<th class="num">{esc(x.replace("effb0_",""))}</th>' for x in ms) + '</tr>')
    conds = sorted({k.split("|", 1)[1] for k in d["rows"]})
    order = ["clean", "jpeg_q50", "downscale_0.5", "pgd_linf_eps0.00784_s10",
             "ace_uint8_eps0.005", "overconf_eps0.0157_s20"]
    conds = [c for c in order if c in conds] + [c for c in conds if c not in order]
    for c in conds:
        cells = []
        for x in ms:
            r = d["rows"].get(f"{x}|{c}")
            if r:
                af = "nan" if r["auroc_failure"] is None or (isinstance(r["auroc_failure"], float) and np.isnan(r["auroc_failure"])) else f'{r["auroc_failure"]:.4f}'
                cells.append(f'<td class="num">{r["accuracy"]:.4f} / {af} / {r["aurc"]*1e3:.1f}</td>')
            else:
                cells.append('<td class="num">—</td>')
        parts.append(f'<tr><td>{esc(c)}</td>' + "".join(cells) + '</tr>')
    parts.append('</table>')

# D4 moderation
mods = []
for name, path in [("EfficientNet-B0", RUNS / "moderation_threatmodel.json"),
                   ("CLIP ViT-L/14", RUNS / "moderation_clip.json"),
                   ("EffNet standard", RUNS / "d4_standard_eps2_255.json"),
                   ("EffNet PGD-AT", RUNS / "d4_at_pgd_eps2_255.json"),
                   ("EffNet TRADES", RUNS / "d4_trades_eps2_255.json")]:
    if path.exists():
        mods.append((name, json.loads(path.read_text())))
if mods:
    parts.append('<h2>D4 — selective moderation (5% SLA, frozen policy)</h2>')
    parts.append('<p>Residual risk = errors among the auto-decided; review rate = the human cost. '
                 'See <a href="moderation.html">moderation</a>.</p>')
    for name, d in mods:
        parts.append(f'<h3>{esc(name)} <span class="pill">t_low={d["t_low"]:.3f} t_high={d["t_high"]:.3f} T={d["temperature"]:.3f}</span></h3>')
        parts.append('<table><tr><th>condition</th><th class="num">acc</th><th class="num">coverage</th>'
                     '<th class="num">review</th><th class="num">residual risk</th>'
                     '<th class="num">missed fakes</th></tr>')
        for r in d["rows"]:
            rr = "—" if r["residual_risk"] is None or (isinstance(r["residual_risk"], float) and np.isnan(r["residual_risk"])) else f'{r["residual_risk"]*100:.2f}%'
            parts.append(f'<tr><td>{esc(r["condition"])}</td><td class="num">{r["accuracy"]:.4f}</td>'
                         f'<td class="num">{r["coverage"]*100:.1f}%</td><td class="num">{r["review_rate"]*100:.1f}%</td>'
                         f'<td class="num">{rr}</td><td class="num">{r["missed_fake_rate"]*100:.2f}%</td></tr>')
        parts.append('</table>')

# training checkpoints
cks = sorted(CKPT.glob("*.json"))
if cks:
    parts.append('<h2>Training runs (WP3 checkpoints)</h2>')
    parts.append('<table><tr><th>checkpoint</th><th>method</th><th class="num">eps (/255)</th>'
                 '<th class="num">epochs</th><th class="num">sel. epoch</th>'
                 '<th class="num">fit_val clean</th><th class="num">fit_val robust</th>'
                 '<th>init from</th><th class="num">min</th></tr>')
    def f4(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
    for ck in cks:
        d = json.loads(ck.read_text())
        h = (d.get("history") or [{}])[-1]
        parts.append(
            f'<tr><td>{esc(ck.stem)}</td><td>{esc(d.get("method","?"))}</td>'
            f'<td class="num">{d.get("eps",0)*255:.1f}</td><td class="num">{d.get("epochs","?")}</td>'
            f'<td class="num">{d.get("selected_epoch", d.get("epoch","?"))}</td>'
            f'<td class="num">{f4(h.get("fit_val_clean"))}</td>'
            f'<td class="num">{f4(h.get("fit_val_robust_pgd"))}</td>'
            f'<td>{esc(d.get("init_from") or "—")}</td>'
            f'<td class="num">{d.get("wall_clock_min","—")}</td></tr>')
    parts.append('</table>')

# figures
f1, f2 = b64img(RUNS / "fig1_risk_coverage.png"), b64img(RUNS / "fig2_confidence_hist.png")
if f1 or f2:
    parts.append('<h2>Figures</h2>')
    if f1:
        parts.append(f'<p><b>Risk–coverage</b> — clean vs the realisable confidence attack.</p>'
                     f'<img src="data:image/png;base64,{f1}" style="max-width:100%;border:1px solid var(--line);border-radius:8px">')
    if f2:
        parts.append(f'<p><b>Confidence by correctness</b> — the distributions swapping places under attack.</p>'
                     f'<img src="data:image/png;base64,{f2}" style="max-width:100%;border:1px solid var(--line);border-radius:8px">')

# provenance
prov = []
for tag, path in [("manifest v1", RUNS / "manifest_v1.json"), ("manifest v2", RUNS / "manifest_v2.json")]:
    if path.exists():
        prov.append(f"{tag} {json.loads(path.read_text()).get('manifest_sha256','?')[:16]}")
coll = RUNS / "img_id_collisions.json"
if coll.exists():
    prov.append(f"img_id collisions {json.loads(coll.read_text())['collisions_total']:,}")
parts.append(f'<div class="box"><div class="t">Provenance</div>{esc(" · ".join(prov))}</div>')
parts.append('<div class="next">→ Back to <a href="index.html">Overview</a> · '
             'read the <a href="implications.html">Implications</a>.</div>')
parts.append('</main><footer>TrustFake · TReC 2026 · dashboard regenerated from artifacts by '
             'wiki/gen_dashboard.py · sources under FILES/PROJECTS/trustfake/wp1</footer></body></html>')

OUT.write_text("\n".join(parts))
print(f"wrote {OUT}  ({OUT.stat().st_size//1024} KB, {n_pred} runs, {len(models)} detectors)")
