# 11 画像観測（Minecraft）で動くか

- **問い**: 画像観測（128 × 128 → VAE 潜在）の 1 方向歩行で、GRU と TF-SVGD が学習できるか。
- **腕（campaign）**: `minecraft-base-v1`（GRU / RNN / PF、各 3 seed）、`minecraft-tf-svgd-v1`（3 seed）。
- **課題**: MineRL 1.0.2、ランダムスタートで一方向に進んだ分が報酬。405 update。
- **設定とジョブ**: `../agent/minecraft/`（環境・VAE・パイプ接続）、`../config/minecraft_{base,tf-svgd}.json`、math の `run1102/mc/`。
- **状態**: TF-SVGD 3 seed 完走、基準線は GRU 3 本・RNN 2 本・PF 2 本が 400 update 以上（2026-11-09）。残り 2 本走行中。

## 結果の解説

- 環境と手順: [agent/minecraft/README.md](../agent/minecraft/README.md)
- 評価値（確率的評価のみ）: [results_summary_1109.md](../results/reports/summary_2026-11-09/results_summary_1109.md)

## 結論メモ

- 確率的評価: TF-SVGD 65 / 27 / 25、GRU 29 / 15 / 35、PF 31 / 67 / 74、RNN 119 / 71 / 82。RNN が最も良く、TF-SVGD は GRU と同程度。seed 分散が大きく 3 seed では順位を確定できない。スケジューラは Minecraft の 2 本も同時実行数に数える。
