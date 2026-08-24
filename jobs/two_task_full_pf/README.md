# two_task_full_pf job set

particle_filter baseline（決定論的 PF 型 ensemble: K=16 学習 anchor 仮説 +
観測適合度重み + soft 床 α=0.9、score との parameter 差 0.085%）を
two-task-full-v2 と同一プロトコル（lr 2.5e-4、entropy 0.01、max_grad_norm 10、
light_dark 報酬再スケール、200 updates、full mode、seed 10）で学習させるセット。

手順（リポジトリ root から）:

```bash
# 1. DEBUG gate（30分枠、両タスク 1 update + 1 評価）
pjsub jobs/two_task_full_pf/debug_pf.sh
# 2. 合格後に本番 2 shards
pjsub jobs/two_task_full_pf/shard01_pf_cartpole.sh
pjsub jobs/two_task_full_pf/shard04_pf_lightdark.sh
```

campaign id は `two-task-full-pf-v1` に固定。segmented 形式（毎 update
checkpoint）なので、中断後は同じファイルの再投入で厳密に続きから再開する。
seed 11/12 へは index +1/+2 の複製で拡張（cartpole: 2,3 / light_dark: 5,6）。
