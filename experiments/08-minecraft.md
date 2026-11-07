# 08 Minecraft: 画像観測の 1 方向歩行（MineRL 1.0.2、VAE 潜在）

- **目的**: 画像観測（128 × 128 → VAE 潜在）で GRU と TF-SVGD の動作確認。ランダムスタートで一方向に進んだ分が報酬。
- **設定とジョブ**: `../agent/minecraft/`（環境・VAE・パイプ接続、README あり）、`../config/minecraft_{base,tf-svgd}.json`、math の `run1102/mc/`。405 update。
- **状態（2026-11-07）**: TF-SVGD s03 が 249 / 405、素 GRU s01 が 129 / 405 で走行中（1 update 数分、Minecraft サーバー再起動で遅れることがある）。

## 結果の解説

- 環境と手順: [agent/minecraft/README.md](../agent/minecraft/README.md)
- 評価値（途中）: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)（campaign `minecraft-*`）

## 結論メモ

- 結果待ち。スケジューラは Minecraft の 2 本も同時実行数に数える。
