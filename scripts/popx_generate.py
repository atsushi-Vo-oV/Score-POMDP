"""Regenerate 36 light + 6 GRU configs, with separate seed-major queues."""
import json
from pathlib import Path
from sb_pomdp.popexplore.experiments import PRODUCTION_ARMS, GRU_ARMS, arm_config
from sb_pomdp.popexplore.config import PopConfig
from sb_pomdp.popexplore.runner import configured_updates


def write_text(path, text, *, lf=False):
    # Preserve the bytes and checkout line endings of identical existing assets.
    if path.exists() and path.read_text(encoding='utf-8') == text:
        return
    path.write_text(text, encoding='utf-8', newline='\n' if lf else None)


def main():
    root = Path(__file__).absolute().parents[1]
    jobs = root / 'jobs/popx'
    jobs.mkdir(parents=True, exist_ok=True)
    manifest = []
    for task in ('carflag', 'mirror2', 'tagr'):
        for arm in (*PRODUCTION_ARMS, *GRU_ARMS):
            source_name = f'trysvgd_{task}_gru' if arm in GRU_ARMS else f'trysvgdlight_{task}_tf-svgd'
            source = json.loads((root / f'config/{source_name}.json').read_text(encoding='utf-8-sig'))
            cfg = json.loads(json.dumps(source))
            cfg['experiment'].update(seeds=list(range(10, 15)), eval_interval_updates=25, checkpoint_interval_updates=25)
            cfg['ppo'].update(num_envs=64, minibatch_size=2048, sequence_microbatch_size=2048,
                              total_steps=(1000 if task == 'tagr' else 200) * 64 * 128)
            cfg['model'].update(score_implementation='autograd', belief_cuda_graphs=False, belief_compile='off',
                                init_from_checkpoint=None, prediction_pretrain_steps=98 * 64 * 128)
            cfg['popexplore'] = {**arm_config(arm), 'group': 'scratch', 'arm': arm, 'freeze_belief': False}
            updates = configured_updates(PopConfig(cfg))
            assert updates == (1293 if task == 'tagr' else 333)
            name = f'popx_scratch_{task}_{arm}'
            config = f'config/{name}.json'
            write_text(root / config, json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
            campaign_arm = arm.replace('_', '-') if arm in GRU_ARMS else arm
            campaign = f'popx-{task}-{campaign_arm}-v1'
            for bulk, seed in enumerate(range(10, 15), 1):
                job = f'jobs/popx/{name}_r{seed}.sh'
                script = f'''#!/usr/bin/env bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=168:00:00
#PJM -j
#PJM -S
set -euo pipefail
export SB_POMDP_ENTRY=popexplore
export SB_POMDP_CONFIG={config}
export SB_POMDP_CAMPAIGN_ID={campaign}
export SB_POMDP_SHARD_INDEX={bulk}
export SB_POMDP_BULK_INDEX={bulk}
export SB_POMDP_SEGMENT_UPDATES="${{SB_POMDP_SEGMENT_UPDATES:-{updates}}}"
if [[ "${{SB_POMDP_SOURCE_ONLY:-0}}" != 1 ]]; then
  cd "${{PJM_O_WORKDIR:-$PWD}}"
  exec bash jobs/popx/launch.sh
fi
'''
                write_text(root / job, script, lf=True)
                manifest.append(dict(group='scratch', task=task, arm=arm, checkpoint_seed=-1, seed=seed,
                                     config=config, campaign=campaign, bulk_index=bulk, shard_index=bulk,
                                     updates=updates, job=job))
    order = {task:i for i, task in enumerate(('carflag', 'mirror2', 'tagr'))}
    light = sorted((r for r in manifest if r['arm'] in PRODUCTION_ARMS),
                   key=lambda r:(r['seed'], order[r['task']], PRODUCTION_ARMS.index(r['arm'])))
    gru = sorted((r for r in manifest if r['arm'] in GRU_ARMS),
                 key=lambda r:(r['seed'], order[r['task']], GRU_ARMS.index(r['arm'])))
    assert len(light) == 180 and len(gru) == 30
    write_text(jobs / 'manifest.json', json.dumps(light + gru, indent=2) + '\n')
    write_text(jobs / 'scratch.queue', ''.join(f"TREE {r['job']}\n" for r in light), lf=True)
    write_text(jobs / 'gru.queue', ''.join(f"TREE {r['job']}\n" for r in gru), lf=True)
    print('Generated 36 light + 6 GRU configs, 180 + 30 jobs, manifest and separate queues')


if __name__ == '__main__':
    main()
