# two_task_full job set

masked_cartpole + light_dark で score/full と gru/full を学習させるセット。
config は `config/two_task_full.json`（lr 2.5e-4、entropy 0.01、200 updates =
409,600 steps、seed 10）。campaign id は `two-task-full-v1` に固定。

投入（リポジトリ root から、1 ファイル = 1 job）:

```bash
pjsub jobs/two_task_full/shard01_score_cartpole.sh
pjsub jobs/two_task_full/shard04_score_lightdark.sh
pjsub jobs/two_task_full/shard07_gru_cartpole.sh
pjsub jobs/two_task_full/shard10_gru_lightdark.sh
```

segmented 形式（毎 update checkpoint）なので、中断後は同じファイルを再投入
するだけで厳密に続きから再開する。完了済み shard への再投入は何も変更せず
exit 0。seed 11/12 へ拡張する場合は index を +1 / +2 した複製を作る
（score: 2,3 / 5,6、gru: 8,9 / 11,12）。
