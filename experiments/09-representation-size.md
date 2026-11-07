# 09 表現の次元を GRU に合わせると（K × D = 隠れ次元）

- **問い**: 粒子数 K × 潜在次元 D を GRU の隠れ次元に合わせれば（mirror2 80 × 2、tagr 36 × 6）、「表現の次元が足りない」説は消えるか。
- **腕（campaign）**: `trysvgdbig-<task>-{svgd,ula}-v1`（light）vs 素の light（K = 16、[02](02-light-vs-energy.md)）と GRU。
- **課題**: mirror2 / tagr × 5 seed。
- **設定とジョブ**: `../config/trysvgdbig_*.json`、`../jobs/trysvgd_big/`（commit 5138bff）。1 update は tagr 40〜55 秒、mirror2 140〜170 秒。
- **状態**: mirror2 SVGD 3 本走行中、残り 17 本は待ち行列。

## 結果の解説

- 評価値（途中）: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)
- 勾配帰属（big の観測 / 遷移比は 2.5〜3.0）: [GRAD_REPORT.md §8.2](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)

## 結論メモ

- 結果待ち。
