#!/usr/bin/env python3
"""Static server for the TrustFake wiki + experiment dashboard. Offline, stdlib only.

Binds 0.0.0.0:8791 so the pages are reachable BOTH at http://127.0.0.1:8791 (local browser
pane) AND across the Tailscale tailnet at http://<this-host>:8791 from your phone or laptop.
0.0.0.0 also exposes it on the local LAN interface; on a trusted home network that is fine,
but if you want tailnet-ONLY, pass the tailnet IP as the first argument:

    python3 serve.py 100.69.174.6

This serves read-only static files and has no mutating endpoints, so no token is used; add
one if you ever put this on an untrusted network.
"""
import http.server, socketserver, pathlib, functools, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent
PORT = 8791
HOST = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)
    def log_message(self, *a):
        pass

def tailnet_ip():
    try:
        return subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True,
                              timeout=3).stdout.strip().splitlines()[0]
    except Exception:
        return None

if __name__ == "__main__":
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((HOST, PORT), Handler) as s:
        print(f"TrustFake wiki serving on {HOST}:{PORT}")
        print(f"  local:   http://127.0.0.1:{PORT}/")
        ip = tailnet_ip()
        if ip:
            print(f"  tailnet: http://{ip}:{PORT}/   (reachable from your tailnet devices)")
        s.serve_forever()
