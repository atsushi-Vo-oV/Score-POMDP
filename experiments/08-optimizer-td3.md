# 08 最適化器を TD3 にすると（PPO vs TD3）

- **問い**: オンポリシーの PPO をオフポリシーの TD3 に替えると、スコア信念と基準線の順位は変わるか。
- **腕（campaign）**: `trysvgdtd3-<task>-tf-{ula,svgd,smi}-v1`、基準線 `trysvgdtd3-<task>-{gru,rnn,pf}-v1` vs PPO 版（[01](01-sampler-energy.md)）。
- **課題**: mirror2 / tps。
- **設定とジョブ**: `ppo.algorithm=td3`（`../src/sb_pomdp/td3.py`）、`../config/trysvgdtd3_*.json`、`../jobs/trysvgd_td3/`。prediction_head=condition のみ対応。
- **状態**: mirror2 TF-SMI など一部完走、残り 48 本は待ち行列（leak / 予測専用の後）。

## 結果の解説

- 評価値一覧: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)（campaign `trysvgdtd3-*`）

## 結論メモ

- 完走分の mirror2 TF-SMI は −1892（崩壊）。まとまった比較は完走後。
