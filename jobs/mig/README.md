# MIG (b-batch-mig) の利用

`b-batch-mig` は H100 を 7 分割した sub-GPU を `-L gpu=1` 単位で割り当て、
ポイントは full GPU の **1/7**。本プロジェクトのジョブはピーク GPU メモリ
2.5 GB・GPU 使用率 ~12%(launch-bound)なので、スライスでも速度低下は
小さい見込み。

## 過去の失敗と対策
以前 score 系が MIG 上で
`NVML_SUCCESS == r INTERNAL ASSERT FAILED (CUDACachingAllocator.cpp:844)`
で即死した(PyTorch caching allocator が MIG デバイスで NVML を呼ぶ既知問題)。
`debug_mig.sh` で (A) 素のまま再現、(B) `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`
での回避、(C) 同設定で gru、を各 1 update 実行し、exit code と wall time を記録する。

## 本番ジョブを MIG で流す
既存の job ファイルを書き換えず、pjsub のコマンドライン指定で上書きする:

```bash
pjsub -L rscgrp=b-batch-mig -L gpu=1 jobs/<set>/<shard>.sh
```

(B) が有効だった場合は `jobs/genkai_production.sh` 側で
`PJM_RSCGRP` が `*mig*` のとき自動で allocator backend と
`OMP_NUM_THREADS=$(nproc)` を設定する(debug 結果確定後に追加)。
