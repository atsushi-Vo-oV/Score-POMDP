# 05 try-svgd-td3: PPO を TD3 に置き換える

- **目的**: 方策最適化器の違い（オンポリシー PPO vs オフポリシー TD3）が信念の学習に与える影響。モデル・FLOPs・予測スケジュールは 01 と同じ。
- **課題**: mirror2 / tps。
- **設定とジョブ**: `../config/trysvgdtd3_*.json`（`ppo.algorithm=td3`、実装 `../src/sb_pomdp/td3.py`）、`../jobs/trysvgd_td3/`。
- **状態（2026-11-07）**: mirror2 TF-SMI など一部完走（Genkai / math の batch 木）。残りは math の待ち行列（leak セットと big の後）。
- **制約**: prediction_head=condition のみ。`policy_observation_input` と `belief_training=prediction_only` は非対応。

## 結果の解説

- 評価値: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)（campaign `trysvgdtd3-*`）

## 結論メモ

- 完走分の mirror2 TF-SMI は −1892（崩壊）。まとまった比較は待ち行列の完走後。
