import sys
import time

sys.path.insert(0, ".")
from PIL import Image  # noqa: E402

from sbmc.walk_env import WalkForward  # noqa: E402

t0 = time.time()
env = WalkForward(max_steps=60, action_repeat=4, seed=1)
print("make", round(time.time() - t0, 1), flush=True)
for ep in range(2):
    t1 = time.time()
    obs = env.reset()
    print(f"ep{ep} reset {time.time() - t1:.1f}s heading {env.heading} start {env.start_position} shape {obs.shape}", flush=True)
    Image.fromarray(obs).save(f"logs/walk_ep{ep}_start.png")
    t2 = time.time()
    total = 0.0
    info = {}
    for t in range(60):
        a = 1 if ep == 0 else env.action_space.sample()  # ep0: always forward; ep1: random
        obs, r, done, info = env.step(a)
        total += r
        if t % 10 == 0:
            print(f"  t={t} a={info['action']:14s} r={r:+.3f} progress={info['progress']:+.2f} y={info['ypos']:.1f} yaw={info['yaw']:.0f}", flush=True)
        if done:
            print("  done at", t)
            break
    dt = time.time() - t2
    print(f"ep{ep} return {total:+.2f} progress {info['progress']:+.2f}  {t + 1} agent steps in {dt:.1f}s = {(t + 1) / dt:.1f} steps/s ({(t + 1) * 4 / dt:.0f} ticks/s)", flush=True)
    Image.fromarray(obs).save(f"logs/walk_ep{ep}_end.png")
env.close()
print("closed", flush=True)
