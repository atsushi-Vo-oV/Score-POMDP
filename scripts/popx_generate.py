"""Regenerate 36 scratch configs / 180 jobs, seed-major and short-task-first."""
import json
from pathlib import Path
from sb_pomdp.popexplore.experiments import PRODUCTION_ARMS, arm_config
from sb_pomdp.popexplore.config import PopConfig
from sb_pomdp.popexplore.runner import configured_updates


def main():
    root = Path(__file__).absolute().parents[1]
    jobs = root / 'jobs/popx'
    jobs.mkdir(parents=True, exist_ok=True)
    # Nonrecursive deletion of obsolete generated files, only inside this worktree.
    for path in list((root / 'config').glob('popx_frozen_*.json')) + list(jobs.glob('*frozen*')):
        if path.is_file():
            path.unlink()
    manifest = []
    for task in ('carflag', 'mirror2', 'tagr'):
        source = json.loads((root / f'config/trysvgdlight_{task}_tf-svgd.json').read_text(encoding='utf-8-sig'))
        for arm in PRODUCTION_ARMS:
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
            (root / config).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            campaign = f'popx-{task}-{arm}-v1'
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
                (root / job).write_text(script, encoding='utf-8', newline='\n')
                manifest.append(dict(group='scratch', task=task, arm=arm, checkpoint_seed=-1, seed=seed,
                                     config=config, campaign=campaign, bulk_index=bulk, shard_index=bulk,
                                     updates=updates, job=job))
    order = {task:i for i, task in enumerate(('carflag', 'mirror2', 'tagr'))}
    manifest.sort(key=lambda r:(r['seed'], order[r['task']], PRODUCTION_ARMS.index(r['arm'])))
    assert len(manifest) == 180
    (jobs / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (jobs / 'scratch.queue').write_text(''.join(f"TREE {r['job']}\n" for r in manifest), encoding='utf-8', newline='\n')
    print('Wrote 36 configs, 180 jobs, manifest and scratch.queue; removed frozen assets')


if __name__ == '__main__':
    main()
