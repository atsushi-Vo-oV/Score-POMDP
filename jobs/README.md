# Genkai jobs

The active production comparison contains two methods:

- `score_transformer`: the score-belief operator policy.
- `gru`: the learned recurrent POMDP baseline.

Both `full` and `tbptt_1` temporal-gradient modes remain enabled for both
methods. For score belief, `full` replays the episode prefix and `tbptt_1`
detaches the incoming belief at every environment transition. GRU uses the
matched definition: `full` replays the observation/action prefix across
rollout boundaries, while `tbptt_1` detaches the incoming hidden state at
every transition. The two modes have identical forwards and different
temporal gradient ranges; neither GRU condition is a duplicate control label.

On continuous tasks both active methods use the same `DiffusionPolicy`,
reverse schedule, per-denoising-step PPO ratios and advantage weights, and
the `mean_chain` deterministic readout. CartPole uses the same categorical
head. Entropy is disabled (`entropy_coef=0.0`). Production GRU capacity is
matched to the score model with encoder widths `[348,185]` and hidden width
`176`; the maximum parameter-count difference over the four tasks is 0.116%.

All four configured tasks are included:

- `masked_cartpole`
- `masked_pendulum`
- `masked_mountain_car_continuous`
- `light_dark`, with `dimension=5` in production

One production seed therefore contains 16 conditions: 2 methods x 2 gradient
modes x 4 tasks. Seeds 10--14 give 80 conditions in the complete campaign.

Bulk submission is disabled. Production uses independent jobs at the shard
level so different conditions can run concurrently as the scheduler permits.
For each seed, the four `full/score_transformer` conditions use independent
25-step PJM chains; every step advances the same shard by at most ten PPO
updates and resumes its committed state. The other 12 conditions are submitted
as independent one-shot jobs. The routine short debug uses
`jobs/genkai_debug_short.sh` to attempt all 16 active conditions with four
independent GPU workers in one B node. The existing exact one-update scripts
remain available for timing calibration and failure isolation.

All scripts use `config/production.json` and Genkai's Python 3.11 / PyTorch
2.3.1 module stack. Score runs split belief replay into sequence memory
microbatches; GRU runs preserve the complete time axis and split PPO
minibatches only over environments. The active matrices are:

- Debug: 16 shards = 2 modes x 2 methods x 4 tasks, seed 0, one PPO update.
- Stability: 48 shards = 2 modes x 2 methods x 4 tasks x seeds 0--2, ten PPO
  updates.
- Production: 80 shards = 2 modes x 2 methods x 4 tasks x seeds 10--14.

Short debug requests one B node without a `gpu` or `vnode-core` selector for at
most 30 minutes. This makes it eligible for Genkai's available short-job nodes,
although queue start and completion of all shards are not guaranteed. Four
full GPUs execute independent conditions and the request ceiling is 60 pt.
Earlier score jobs hit a PyTorch/NVML allocator assertion on MIG, so debug does
not use `b-batch-mig`. Exact debug and stability still request one full GPU for
two hours. Every production job or step reserves one full GPU
for up to 168 hours, or a 5,040 pt request ceiling. A 25-step chain has a much
larger aggregate request ceiling, although charging follows actual use.
Timing or point estimates measured before GRU received the matched diffusion
head and full episode-prefix replay are not valid for the current comparison;
re-run the combined debug and one-shard production pilot before extrapolating
the complete campaign cost.

## Debug commands

The routine command attempts all 16 conditions with a 28-minute internal
watchdog and a 30-minute PJM limit:

```bash
pjsub -x "SB_POMDP_CAMPAIGN_ID=debug-short-$(date -u +%Y%m%dT%H%M%SZ)" jobs/genkai_debug_short.sh
```

The short-only overrides are `num_envs=4`, `epochs=1`, `max_updates=1`, and one
evaluation episode. The production network, action head, K=32 particles, L=8
Langevin steps, rollout length 128, logical minibatch 512, and sequence
microbatch 128 remain unchanged.

### Combined debug + pilot in one submission

`jobs/genkai_debug_then_pilot.sh` requests one B node (4 full GPUs) for two
hours and chains both validation steps into a single job. Phase 1 runs the same
16-condition short debug as `genkai_debug_short.sh` (28-minute internal
watchdog); any shard failure fails the job and skips phase 2. Phase 2 spends
the remaining wall time (minus a 6-minute teardown margin) running the
calibration shard 11 (`full/score_transformer/MountainCar`, seed 10) through
the segmented production path on one GPU, with campaign id `<id>-pilot` and
`SB_POMDP_PILOT_SEGMENT_UPDATES` (default 10) updates requested.

```bash
pjsub -x "SB_POMDP_CAMPAIGN_ID=debug-pilot-$(date -u +%Y%m%dT%H%M%SZ)" jobs/genkai_debug_then_pilot.sh
```

A pilot stopped by the wall-time budget exits 0: the committed-update count
printed in the job output is the speed measurement, and the pilot shard stays
resumable under the same campaign id, shard index, and source. Only a debug
failure or a pilot crash before the budget produces a nonzero exit.

### Experimental benefits and limitations

- Benefit: every active task, method, and temporal-gradient mode traverses the
  production CUDA, autograd, optimizer, diagnostics, evaluation, and artifact
  paths. Independent condition-level parallelism does not mix gradients across
  experiments.
- Limitation: each update has 512 rather than 2,048 transitions and one rather
  than 16 optimizer steps. Seed 0, one update, and one evaluation episode do
  not establish performance, convergence, variance, multi-epoch KL behavior,
  long-term stability, cross-rollout full replay, resume correctness, peak
  production memory, or production time/point cost. Never aggregate short-debug
  returns into the final comparison.

For exact one-update timing, the original production-sized DEBUG remains:

```bash
pjsub -x "SB_POMDP_CAMPAIGN_ID=debug-heavy-$(date -u +%Y%m%dT%H%M%SZ)" jobs/genkai_debug_heavy.sh
pjsub -x "SB_POMDP_CAMPAIGN_ID=debug-two-method-$(date -u +%Y%m%dT%H%M%SZ)" jobs/genkai_debug_all.sh
```

These exact jobs retain `num_envs=16` and `epochs=4`, request one full B GPU for
two hours, and are not the routine low-wait path. `genkai_debug_heavy.sh` fixes
the condition to full score/MountainCar; `genkai_debug_all.sh` runs shards 3,
11, 7, and 15 sequentially.

For failure isolation, a single condition can still be submitted directly:

```bash
# full / score_transformer / MountainCar / seed 0
pjsub -x "SB_POMDP_SHARD_INDEX=3,SB_POMDP_CAMPAIGN_ID=debug-two-method-20260810T153036Z" jobs/genkai_debug.sh

# tbptt_1 / score_transformer / same task/seed
pjsub -x "SB_POMDP_SHARD_INDEX=11,SB_POMDP_CAMPAIGN_ID=debug-two-method-20260810T153036Z" jobs/genkai_debug.sh
```

The complete debug matrix remains selectable as indices 1--16.

## Production campaign

`jobs/submit_production_campaign.sh` submits exactly one production seed. The
offsets 0--4 select seeds 10--14. Always give dry-run and real submission the
same explicit campaign ID; otherwise their independently generated timestamps
will differ.

```bash
campaign_id="prod-two-method-$(date -u +%Y%m%dT%H%M%SZ)"

# Print all pjsub commands without submitting anything.
bash jobs/submit_production_campaign.sh --dry-run --seed-offset 0 "$campaign_id"

# Submit seed 10 after inspecting the dry-run output.
bash jobs/submit_production_campaign.sh --seed-offset 0 "$campaign_id"
```

This submits four 25-step chains (one chain for each task under
`full/score_transformer`) and 12 independent jobs. The four chains and the 12
jobs are independent of one another and may overlap; the 25 segments inside
one chain execute sequentially. Each segment runs at most ten additional PPO
updates, so 25 successful segments reach the configured 250 updates. This is
an operational wall-time split only: `config/production.json`, the total
environment-step budget, optimizer settings, model, tasks, and seeds are not
changed.

On Genkai, the site `/usr/local/bin/pjsub` wrapper accepts only one positional
`script_file` per invocation, despite the older `/usr/bin/pjsub` manual
describing multi-script invocation. The launcher therefore submits the first
subjob with `-z jid`, extracts its step-job ID, and attaches every later subjob
with `jid=<step-job-id>`. Each later subjob explicitly names the preceding
step in `sd=ec!=0:all:<previous-step>`. It never passes multiple script paths
to one call.

After a pilot supports continuing, the other seeds can be submitted into the
same campaign by changing only `--seed-offset`:

```bash
for seed_offset in 1 2 3 4; do
  bash jobs/submit_production_campaign.sh \
    --seed-offset "$seed_offset" \
    "$campaign_id"
done
```

### One-shard, ten-update pilot

To validate the production checkpoint/resume path with two one-update step
subjobs first, run this single shell command:

```bash
pilot_id="prod-resume-smoke-$(date -u +%Y%m%dT%H%M%SZ)"; bash jobs/submit_production_chain.sh --shard-index 11 --segment-updates 1 --segments 2 "$pilot_id"
```

Shard 11 is the seed-10 `full/score_transformer` MountainCar condition and is
the most important measured wall-time pilot. This command submits only its
first ten-update segment:

```bash
pilot_id="prod-pilot-mountaincar-$(date -u +%Y%m%dT%H%M%SZ)"
pjsub -x "SB_POMDP_SHARD_INDEX=11,SB_POMDP_CAMPAIGN_ID=${pilot_id},SB_POMDP_SEGMENT_UPDATES=10" \
  jobs/genkai_production.sh
```

Submitting the exact same campaign/shard/segment tuple again resumes from
`results/campaigns/$pilot_id/shard011/.../checkpoints/latest.pt` and advances
by at most ten more updates:

```bash
pjsub -x "SB_POMDP_SHARD_INDEX=11,SB_POMDP_CAMPAIGN_ID=${pilot_id},SB_POMDP_SEGMENT_UPDATES=10" \
  jobs/genkai_production.sh
```

Do not change `config/production.json`, the source tree, campaign ID, or shard
index between segments. Resume checks these identities, and a per-shard lock
rejects overlapping segments for the same campaign. Reusing the command after
all 250 updates are complete is also rejected rather than starting a new run.
The current checkpoint format is version 4; checkpoints written by earlier
formats cannot be resumed by this source version.

Use expanded PJM status output to see the individual members of step chains:

```bash
pjstat -E
# Once an entire chain has left the active listing:
pjstat -E -H
```

`partial` in the shard metadata, manifest, or summary means that the segment
ended normally and committed resumable state, but fewer than the configured
250 updates have completed. It is not a finished comparison result.
`completed` is written only after all configured updates and final evaluations
finish. A nonzero step exit is a real failure; the campaign launcher uses
`sd=ec!=0:all` so PJM deletes the remaining members of that chain.

The 25-way split avoids Genkai's 168-hour limit but does not reduce total GPU
time or points, and it adds small restart overhead. With a 90,000 pt budget,
run and inspect the one-shard ten-update pilot before submitting even one full
16-condition seed. Parallel submission shortens calendar time only; it does
not make the underlying computation cheaper.

For stability inspection, `metrics.csv` records the pre-clip gradient norms
of the initial (`g0`), observation (`f`), and transition (`u`) energy modules,
initial/recursive score and particle norms, their sample counts, and
non-finite counts. These are diagnostics only; they do not reject an optimizer
step. `mean_chain` and the Gaussian-only `mean_action` are action-readout
labels, not metrics. The recommended primary endpoint is final stochastic
evaluation return aggregated over held-out episodes and training seeds.

To transfer the one-shard pilot, use `--expected-shard-indices 11`. A complete
five-seed campaign uses `--expected-shards 80`. A partial campaign is a
calibration run and cannot be presented as the complete production aggregate.
