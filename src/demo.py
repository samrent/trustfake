"""WP4 end-to-end demonstrator: upload an image, see the moderation decision.

Deliberately stdlib-only (http.server). No Flask/FastAPI is installed in this venv, and
adding one would make the demonstrator the thing that breaks when the camp wifi does. Runs
fully offline, loads the detector once, falls back to CPU if the GPU is busy -- the 3090
here is usually holding a dictation model.

Raw-body upload rather than multipart: the page POSTs the File object straight to /analyze,
so there is no multipart parser to get wrong (and `cgi`, the stdlib one, is deprecated and
gone in 3.13).

THE POLICY IS LOADED, NOT COMPUTED. Thresholds come from runs/moderation_report.json, where
they were fitted on the clean calibration split and frozen. A demonstrator that re-derived
its own thresholds from whatever it has seen would be a different system from the one the
report measures, and the numbers on the slide would not describe the thing on the screen.

This is a research prototype over a held-out slice of the SID-Set VALIDATION split. It is
not a deployable moderation system and the page says so.
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

from . import metrics as M
from .decode import CACHE, cache_path
from .models import build
from .moderation import ACTION_NAMES, actions

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATE: dict = {}

PAGE = """<!doctype html><meta charset=utf-8><title>TrustFake - selective moderation demo</title>
<style>
 :root{--bg:#0e1116;--fg:#e6edf3;--mut:#8b949e;--line:#30363d;--card:#161b22}
 @media(prefers-color-scheme:light){:root{--bg:#fff;--fg:#1f2328;--mut:#656d76;--line:#d0d7de;--card:#f6f8fa}}
 body{background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,sans-serif;
      max-width:860px;margin:0 auto;padding:32px 20px}
 h1{font-size:20px;margin:0 0 4px} .mut{color:var(--mut);font-size:13px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin:18px 0}
 .drop{border:1.5px dashed var(--line);border-radius:10px;padding:38px;text-align:center;cursor:pointer}
 .drop:hover{border-color:#58a6ff}
 .row{display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap}
 img{max-width:300px;max-height:300px;border-radius:8px;border:1px solid var(--line)}
 .verdict{font-size:26px;font-weight:650;margin:2px 0 10px}
 .allow{color:#3fb950}.review{color:#d29922}.flag{color:#f85149}
 table{border-collapse:collapse;font-size:13px;margin-top:8px}
 td{padding:3px 14px 3px 0}  td:first-child{color:var(--mut)}
 .bar{height:7px;background:var(--line);border-radius:4px;overflow:hidden;width:230px;margin:5px 0}
 .bar>i{display:block;height:100%;background:#58a6ff}
 code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:12px}
 button.samp{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;
   padding:4px 11px;margin-right:6px;cursor:pointer;font-size:13px}
 button.samp:hover{border-color:#58a6ff}
 .truth{font-size:13px;margin-top:9px;padding:7px 11px;border-radius:6px;border:1px solid var(--line)}
 .ok{border-color:#3fb950}.bad{border-color:#f85149}
</style>
<h1>TrustFake &mdash; selective moderation</h1>
<div class=mut>Detector <code id=mid>?</code> &middot; policy frozen on the clean calibration split
 (<code id=pol>?</code>) &middot; <span id=dev>?</span></div>

<div class=card><div class=drop id=drop>Drop an image here, or click to choose one
<input type=file id=f accept="image/*" hidden></div>
<div class=mut style="margin-top:12px">or try a held-out SID-Set image:
 <button class=samp data-k=real>real</button>
 <button class=samp data-k=synthetic>fully synthetic</button>
 <button class=samp data-k=tampered>tampered</button></div></div>
<div id=out></div>

<div class=card><b>What this is not.</b> <span class=mut>A research prototype scored on a held-out
slice of the SID-Set <i>validation</i> split (the official test split is withheld by the dataset
authors). It is not a deployable moderation system: it is a fixed backbone with a linear head,
measured on one dataset, and its confidence is <b>known to be attackable</b> &mdash; a perturbation
that leaves accuracy bit-identical drives the residual risk of this very policy from 4.7% to
98.1%.</span></div>

<script>
const drop=document.getElementById('drop'),f=document.getElementById('f'),out=document.getElementById('out');
drop.onclick=()=>f.click();
drop.ondragover=e=>{e.preventDefault();drop.style.borderColor='#58a6ff'};
drop.ondragleave=()=>drop.style.borderColor='';
drop.ondrop=e=>{e.preventDefault();drop.style.borderColor='';go(e.dataTransfer.files[0])};
f.onchange=()=>go(f.files[0]);
document.querySelectorAll('button.samp').forEach(b=>b.onclick=async()=>{
  const r=await fetch('/sample?kind='+b.dataset.k);
  const truth=r.headers.get('X-Truth'), name=r.headers.get('X-Name');
  go(new File([await r.blob()],name||'sample.jpg',{type:'image/jpeg'}),truth);});
fetch('/health').then(r=>r.json()).then(h=>{
  document.getElementById('mid').textContent=h.model_id;
  document.getElementById('pol').textContent='t_low='+h.t_low.toFixed(3)+', t_high='+h.t_high.toFixed(3);
  document.getElementById('dev').textContent=h.device;});
async function go(file,truth){
  if(!file) return;
  out.innerHTML='<div class=card>scoring&hellip;</div>';
  const r=await fetch('/analyze',{method:'POST',body:file});
  if(!r.ok){out.innerHTML='<div class=card>error: '+await r.text()+'</div>';return}
  const d=await r.json(), url=URL.createObjectURL(file);
  out.innerHTML=`<div class=card><div class=row>
    <img src="${url}">
    <div>
      <div class="verdict ${d.action}">${d.action_label}</div>
      <div class=mut>${d.rationale}</div>
      <table>
        <tr><td>predicted</td><td><b>${d.label}</b></td></tr>
        <tr><td>p(fake)</td><td><b>${(d.p_fake*100).toFixed(1)}%</b>
            <div class=bar><i style="width:${(d.p_fake*100).toFixed(1)}%"></i></div></td></tr>
        <tr><td>confidence (MSP)</td><td>${(d.confidence*100).toFixed(1)}%</td></tr>
        <tr><td>decision band</td><td>allow &lt; ${d.t_low.toFixed(3)} &middot;
            flag &gt; ${d.t_high.toFixed(3)}</td></tr>
      </table>
      ${truth?`<div class="truth ${(truth!=='real')===(d.label==='fake')?'ok':'bad'}">
        ground truth: <b>${truth}</b> &mdash; the detector was
        ${(truth!=='real')===(d.label==='fake')?'right':'<b>wrong</b>'}</div>`:''}
    </div></div></div>`;
}
</script>"""

RATIONALE = {
    "allow": "p(fake) is below the allow threshold, so this is auto-accepted without a human.",
    "review": "p(fake) sits inside the uncertainty band, so this is escalated to a human moderator.",
    "flag": "p(fake) is above the flag threshold, so this is auto-flagged as manipulated.",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):            # keep the terminal usable during a live demo
        pass

    def do_GET(self):
        if self.path.startswith("/sample"):
            # Serve a random image from the TEST split -- never fit or calib. A demo that
            # showed a training image would be a demo of memorization.
            from urllib.parse import parse_qs, urlparse
            kind = parse_qs(urlparse(self.path).query).get("kind", ["real"])[0]
            pool = STATE["samples"].get(kind, [])
            if not pool:
                return self._send(404, b"no sample of that kind", "text/plain")
            i = STATE["rng"].integers(len(pool))
            uid = pool[int(i)]
            body = cache_path(uid, CACHE).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Truth", kind)
            self.send_header("X-Name", uid.split(":")[-1])
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self._send(200, json.dumps({
                "ok": True, "model_id": STATE["model_id"], "device": STATE["device"],
                "t_low": STATE["t_low"], "t_high": STATE["t_high"],
                "temperature": STATE["T"], "policy_source": STATE["policy_source"],
            }).encode(), "application/json")
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/analyze":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if not 0 < n <= 32 * 1024 * 1024:
            return self._send(413, b"image must be 1 byte - 32 MB", "text/plain")
        raw = self.rfile.read(n)
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:                                    # noqa: BLE001
            return self._send(400, f"cannot decode image: {type(e).__name__}".encode(), "text/plain")

        x = STATE["transform"](im).unsqueeze(0).to(STATE["device"])
        with torch.no_grad():
            logits = STATE["model"](x).float().cpu().numpy().astype(np.float64)
        p = M.softmax(logits, STATE["T"])[0]
        p_fake = float(p[1])
        act = int(actions(np.array([p_fake]), STATE["t_low"], STATE["t_high"])[0])
        name = ACTION_NAMES[act]
        self._send(200, json.dumps({
            "p_fake": p_fake, "confidence": float(p.max()),
            "label": "fake" if p_fake > 0.5 else "real",
            "action": name,
            "action_label": {"allow": "ALLOW", "review": "HUMAN REVIEW", "flag": "AUTO-FLAG"}[name],
            "rationale": RATIONALE[name],
            "t_low": STATE["t_low"], "t_high": STATE["t_high"],
        }).encode(), "application/json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="probe:tf_efficientnet_b0.ns_jft_in1k")
    ap.add_argument("--policy", type=pathlib.Path, default=ROOT / "runs" / "moderation_report.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8471)
    ap.add_argument("--manifest", type=pathlib.Path, default=ROOT / "runs" / "manifest_v1.parquet")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        det = build(a.model, device=device)
    except torch.OutOfMemoryError:
        device = "cpu"                        # the 3090 is usually holding a dictation model
        det = build(a.model, device=device)

    import pyarrow.parquet as pq
    t = pq.read_table(a.manifest, columns=["uid", "label", "split"])
    names = {0: "real", 1: "synthetic", 2: "tampered"}
    samples: dict[str, list[str]] = {"real": [], "synthetic": [], "tampered": []}
    for uid, lab, sp in zip(t.column("uid").to_pylist(), t.column("label").to_pylist(),
                            t.column("split").to_pylist()):
        if sp == "test":
            samples[names[lab]].append(uid)

    pol = json.loads(a.policy.read_text())
    STATE.update(samples=samples, rng=np.random.default_rng(0),model=det.module, transform=det.transform, model_id=det.model_id,
                 device=device, T=pol["temperature"], t_low=pol["t_low"], t_high=pol["t_high"],
                 policy_source=str(a.policy))
    print(f"samples: " + ", ".join(f"{k}={len(v)}" for k, v in samples.items()))
    print(f"detector {det.model_id} on {device}; policy t_low={pol['t_low']:.4f} "
          f"t_high={pol['t_high']:.4f} T={pol['temperature']:.4f} (frozen on clean calib)")
    print(f"http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
