# two_task_alpha job set

α-vector 型方策(`score_alpha` = 粒子位置のみの mean-pool trunk +
温度付き logsumexp の PWLC head、Transformer なし)と従来の
`score_transformer` を、**cold(通常)belief と warm belief の両設定**で
比較するセット。プロトコルは two-task-full-v2 と同一(200 updates、
lr 2.5e-4、entropy 0.01、max_grad_norm 10、LD 報酬再スケール、full mode、
seed 10)。

## 比較行列と対照

| 設定 | score_transformer | score_alpha |
|---|---|---|
| cold(τ=1、cold init、α 固定) | **v2 完走結果を流用**(再実行不要) | shard01 / shard04 |
| warm(warm start + τ=0.1 + α_l 学習) | shard01_warmtr / shard04_warmtr | shard07_warmalpha / shard10_warmalpha |

- config: `two_task_alpha_cold.json`(methods=[score_alpha]、index 1=cartpole, 4=LD)/
  `two_task_alpha_warm.json`(methods=[score_transformer, score_alpha]、
  index 1/4=transformer、7/10=alpha)。`--validate-only --bulk-index` 検証済み。
- warm 設定は τ study の先行アーム warm_tau010 と同一
  (`langevin_warm_start=true, langevin_temperature=0.1,
  langevin_step_size_learnable=true`)。τ study の最終結果で最良設定が
  変わった場合は warm config を更新してから投入する。
- **パラメータ数は意図的に不一致**(score_deepsets と同じ「構造 ablation は
  次元を共有し、数は報告する」方針): cartpole 74,202 vs **20,775**、
  LD 77,405 vs **37,644**。α-pool が 3.6 倍小さいまま拮抗すれば、
  必要なのは attention ではなく「凸クラス + 生きた粒子」という主張になる。

## 手順(リポジトリ root から)

```bash
# 1. DEBUG gate(30分枠: cold_alpha / warm_alpha / warm_transformer × 両タスク、各1 update)
pjsub jobs/two_task_alpha/debug_alpha.sh

# 2. 合格後(計画では warm 系のみ先行投入)
pjsub jobs/two_task_alpha/shard01_warmtr_cartpole.sh
pjsub jobs/two_task_alpha/shard04_warmtr_lightdark.sh
pjsub jobs/two_task_alpha/shard07_warmalpha_cartpole.sh
pjsub jobs/two_task_alpha/shard10_warmalpha_lightdark.sh

# (cold-alpha は必要になったら)
pjsub jobs/two_task_alpha/shard01_alpha_cartpole.sh
pjsub jobs/two_task_alpha/shard04_alpha_lightdark.sh
```

segmented(毎 update checkpoint)なので、24h 上限で止まったら同じファイルの
再投入で厳密に続きから再開する(score 系 full mode の実測: LD ~10分/update ≈
34h、cartpole ~18分/update ≈ 60h — 途中再投入前提。alpha 版は encoder が
軽い分やや速い見込み)。解析は return に加えて piece 勝率分布(実効 piece 数
= タスクが要求する凸性の診断)と表現プローブを併用する。
