# Minecraft "walk in one direction" task (2026-11-03)

Image-in, discrete-action-out POMDP on a real Minecraft (MineRL 1.0.2, Minecraft 1.16.5), run on the math server
under Xvfb.  Everything here is user-space under `~/workspace/agent/minecraft` on that server (MineRL venv,
Python 3.10, `src/minerl-1.0.2` built with gradle); the training side lives in `src/sb_pomdp/minecraft_env.py`.

## Task (`sbmc/walk_env.py`, `WalkForward`)

- Episode: a freshly generated world (random spawn, plains biome preferred, noon, no mob spawning), the agent is
  turned to a uniformly random yaw at reset and the heading unit vector `h` is fixed from that yaw.
- Observation: RGB 128x128 point of view.  Minecraft renders 128x240 with GUI scale 1 and the frame is cropped to
  rows 64..192 so that the toast notifications (top) and the HUD (hearts, hotbar; bottom) never appear.
  `tutorialStep:none` is set in MineRL's `MCP-Reborn/options.txt`.
- Actions (9, one agent step = 4 Minecraft ticks): noop, forward, forward+sprint, forward+jump, back, turn left,
  turn right, forward+left, forward+right (turns are 5 deg per tick).
- Reward per agent step: `<p_t - p_{t-1}, h>` in blocks (horizontal displacement along the initial heading).
  Position / yaw come from MineRL's location stats and are returned in `info` only (oracle), never observed.
- Speed: about 4 agent steps / s per instance (Minecraft runs at real time, 20 ticks / s); reset 15-45 s.

## Pipeline

1. `scripts/collect_frames.py data/frames_<i>.npz 10000 <seed>` - random, action-persistent policy; several
   instances in parallel under `xvfb-run -a`.
2. `scripts/train_vae.py data/vae32.pt "data/frames_*.npz" --latent 32 --epochs 40` - convolutional VAE
   (4 stride-2 convs 128 -> 8, latent 32); writes the weights, the latent mean / std for normalisation and a
   reconstruction sheet.
3. Training: `config/minecraft_tf-svgd.json` (light TF-SVGD) and `config/minecraft_base.json` (GRU / RNN / PF),
   task `minecraft_walk` with `backend: minerl`, `vae_checkpoint`, `minecraft_python`, `minecraft_root`.  The
   trainer (torch venv) starts `sbmc/server.py` in the MineRL venv per environment and drives it over a pipe
   (length-prefixed pickles); the VAE encoder runs in the trainer on the CPU.  4 environments x 128 steps per
   update, prediction pretraining 25,600 steps, 300 PPO updates.
4. `backend: fake` (`config/minecraft_*-fake.json`) is an in-process kinematic stand-in with a synthetic image
   and a block-mean "encoder" (48-d) for tests and CPU smoke runs (`src/tests/test_minecraft_env.py`).

Caveats: a Minecraft world cannot be restored from a checkpoint, so a resumed run continues with fresh episodes
(`load_state_dict` leaves `needs_reset` set; both trainers reset such environments at resume and drop the
interrupted episode from the accumulators, logging a warning; `src/tests/test_minecraft_resume.py`); FLOPs of the
baselines are not matched to the transformer here yet; the
crosshair stays in the frame.
