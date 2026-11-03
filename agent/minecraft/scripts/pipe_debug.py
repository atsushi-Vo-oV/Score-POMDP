"""Drive sbmc/server.py directly (same venv) and report each stage with timestamps; stderr of the server to a file."""
import os
import pickle
import struct
import subprocess
import sys
import time

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python = sys.executable
cmd = [python, "-m", "sbmc.server", "--max-steps", "5", "--action-repeat", "4", "--seed", "1"]
if not os.environ.get("DISPLAY"):
    cmd = ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24", *cmd]
log = open(os.path.join(root, "logs", "pipe_debug_server.log"), "wb")
t0 = time.time()
p = subprocess.Popen(cmd, cwd=root, env=dict(os.environ, PYTHONPATH=root, CUDA_VISIBLE_DEVICES=""),
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
print("spawned pid", p.pid, flush=True)


def recv():
    h = p.stdout.read(4)
    if len(h) < 4:
        return None
    (n,) = struct.unpack("<I", h)
    return pickle.loads(p.stdout.read(n))


def send(m):
    d = pickle.dumps(m)
    p.stdin.write(struct.pack("<I", len(d)) + d)
    p.stdin.flush()


hello = recv()
print(f"hello after {time.time() - t0:.0f}s: {hello}", flush=True)
send({"cmd": "reset"})
r = recv()
print(f"reset after {time.time() - t0:.0f}s: obs {None if r is None else r['obs'].shape} info {None if r is None else r['info']}", flush=True)
for i in range(5):
    send({"cmd": "step", "action": 1})
    r = recv()
    print(f"step {i}: reward {r['reward']:.3f} done {r['done']} progress {r['info'].get('progress')}", flush=True)
send({"cmd": "close"})
print("exit code", p.wait(timeout=60), f"total {time.time() - t0:.0f}s")
