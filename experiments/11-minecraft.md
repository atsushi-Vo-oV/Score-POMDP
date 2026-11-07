# 11 画像観測（Minecraft）で動くか

- **問い**: 画像観測（128 × 128 → VAE 潜在）の 1 方向歩行で、GRU と TF-SVGD が学習できるか。
- **腕（campaign）**: `minecraft-base-v1`（GRU）、`minecraft-tf-svgd-v1`。
- **課題**: MineRL 1.0.2、ランダムスタートで一方向に進んだ分が報酬。405 update。
- **設定とジョブ**: `../agent/minecraft/`（環境・VAE・パイプ接続）、`../config/minecraft_{base,tf-svgd}.json`、math の `run1102/mc/`。
- **状態**: TF-SVGD s03 249 / 405、GRU s01 129 / 405 で走行中（1 update 数分、サーバー再起動で遅れることがある）。

## 結果の解説

- 環境と手順: [agent/minecraft/README.md](../agent/minecraft/README.md)
- 評価値（途中）: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)

## 結論メモ

- 結果待ち。スケジューラは Minecraft の 2 本も同時実行数に数える。
