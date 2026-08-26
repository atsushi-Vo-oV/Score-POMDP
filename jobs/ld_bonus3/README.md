# ld_bonus3 job set

タスク妥当性監査(scratchpad/ld_task_audit.py)の結果を受けた **light_dark の
goal_bonus 1.0 → 3.0** 修正版。監査では、厳密ベイズ追跡つきスクリプト方策で
bonus 1.0 のとき情報収集(光訪問)の margin が **−0.11**(dead-reckon
−0.294 vs light-visit −0.401、goal 到達率 0.03 vs 0.85)で最適にならず、
**bonus 3.0 で margin +1.54** と明確に最適になることを確認した。
Light-Dark 5D は本実装独自の拡張(原典 Platt 2010 は 2D・planning)なので
外部比較の制約はない。他は two-task-full-v2 プロトコル(200 updates、
seed 10、full)。

## 構成

- `ld_bonus3_warm.json`(warm belief: warm start + τ0.1 + α_l 学習):
  index 1 = score_transformer、4 = score_alpha、**7 = score_deepsets**
  (campaign ld-bonus3-warm-v1)
- `ld_bonus3_refs.json`(cold belief 参照):
  index 1 = score_transformer、4 = score_alpha、7 = gru、10 = rnn、
  13 = particle_filter、**16 = score_deepsets**(campaign ld-bonus3-refs-v1)

score_deepsets は encoder 梯子の中段(alpha_pool ⊂ deep_sets ⊂ transformer)。
methods リストへの**末尾追加**なので既存 index(走行中 shards 含む)は不変。
alpha 優位が続いた場合に「attention を外した効果」と「後段非線形・score
入力を外した効果」を分離する。

DEBUG ゲートは不要(全機構はゲート通過済み、変更は報酬定数のみ)。
CartPole 側は bonus と無関係なので、走行中の two-task-alpha-warm-v1 の
cartpole shards と v2/PF/RNN の cartpole 結果をそのまま使う。
bonus 1.0 の LD 既存結果(v2/PF/RNN/τ study)は「受動 filtering
ベンチマーク」として解釈を限定して保持する。

## 手順(リポジトリ root から)

```bash
# warm 比較(bonus1 で走り始めた LD 2本を pjdel した後に)
pjsub jobs/ld_bonus3/warmtr_lightdark.sh
pjsub jobs/ld_bonus3/warmalpha_lightdark.sh
# cold 参照(gru/rnn/pf は速い; score 系は ~34h で要再投入)
pjsub jobs/ld_bonus3/score_lightdark.sh
pjsub jobs/ld_bonus3/alpha_lightdark.sh
pjsub jobs/ld_bonus3/gru_lightdark.sh
pjsub jobs/ld_bonus3/rnn_lightdark.sh
pjsub jobs/ld_bonus3/pf_lightdark.sh
```

判定は return に加えて **光訪問(max x₀)・goal 到達率**を必ず見る:
bonus 3.0 でも探索が光ルートを発見できない可能性は残る(bonus は goal に
入って初めて得られる深い探索課題)。その場合は entropy・初期分布側の介入を
検討する。スクリプト参照値: dead-reckon −0.239 / light-visit +1.304(bonus 3)。
