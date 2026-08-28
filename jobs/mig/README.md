# MIG (b-batch-mig) の利用

`b-batch-mig` は H100 を 7 分割した sub-GPU(1g.12gb、10.75 GiB、CPU 4 コア)を
`-L gpu=1` 単位で割り当て、ポイントは full GPU の **1/7**。

## 検証結果(2026-08-28、debug_mig.sh、job 6631172)
- 3 アーム全て exit 0(素の allocator で score OK、cudaMallocAsync でも OK、gru OK)。
  8/10 に見られた NVML assert は現行 PyTorch 2.3.1 モジュール + 現コードでは**再現せず**。
- **PPO update 時間: MIG 37.4 s vs full GPU 38.3 s(同速)**。launch-bound な本ジョブは
  スライスで速度が落ちない → **update あたりのコストは 1/7**。
- 以後の production は原則 MIG で流す。

## 使い方(job ファイルは書き換え不要)
```bash
bash jobs/mig/pjsub_mig.sh jobs/ld2d_p3o/score_p3o.sh jobs/ld2d_p3o/alpha_p3o.sh
# = pjsub -L rscgrp=b-batch-mig -L gpu=1 <job.sh>
```
`genkai_production.sh` は `PJM_RSCGRP` が mig のとき `OMP_NUM_THREADS=$(nproc)`(=4)
に自動調整する。投入後 `pjstat` の CORE 列が 4 なら MIG 割当、30 なら full GPU。
MIG ノードは 2 台(56 スライス)なので満杯のときは待ち行列になる。
