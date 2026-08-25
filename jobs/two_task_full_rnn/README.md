# two_task_full_rnn job set

単純 RNN（Elman、tanh、ゲートなし）baseline を two-task-full-v2 と同一
プロトコル（lr 2.5e-4、entropy 0.01、max_grad_norm 10、light_dark 報酬
再スケール、200 updates、full mode、seed 10）で学習させるセット。

GRU との差はゲート機構の有無だけ（encoder→再帰セル→共通 head という構造・
入力 obs+prev_action・BPTT 経路は `GRUActorCritic(recurrent_cell="rnn")` で
完全共有）。パラメータは rnn_encoder_hidden=[149,177] / rnn_hidden_dim=64 で
score_transformer と一致（cartpole +0.06% / light_dark +0.02%）。ゲートが
ない分、同じパラメータ予算で再帰状態は 64 次元（GRU は 43）になる。

手順（リポジトリ root から）:

```bash
# 1. DEBUG gate（30分枠、両タスク 1 update + 1 評価）
pjsub jobs/two_task_full_rnn/debug_rnn.sh
# 2. 合格後に本番 2 shards
pjsub jobs/two_task_full_rnn/shard01_rnn_cartpole.sh
pjsub jobs/two_task_full_rnn/shard04_rnn_lightdark.sh
```

campaign id は `two-task-full-rnn-v1` に固定。segmented 形式（毎 update
checkpoint）なので、中断後は同じファイルの再投入で厳密に続きから再開する。
seed 11/12 へは index +1/+2 の複製で拡張（cartpole: 2,3 / light_dark: 5,6）。
