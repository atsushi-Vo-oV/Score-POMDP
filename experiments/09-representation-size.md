# 09 表現の次元を GRU に合わせると（K × D = 隠れ次元）

- **問い**: 粒子数 K × 潜在次元 D を GRU の隠れ次元に合わせれば（mirror2 80 × 2、tagr 36 × 6）、「表現の次元が足りない」説は消えるか。
- **腕（campaign）**: `trysvgdbig-<task>-{svgd,ula}-v1`（light）vs 素の light（K = 16、[02](02-light-vs-energy.md)）と GRU。
- **課題**: mirror2 / tagr × 5 seed。
- **設定とジョブ**: `../config/trysvgdbig_*.json`、`../jobs/trysvgd_big/`（commit 5138bff）。1 update は tagr 40〜55 秒、mirror2 140〜170 秒。
- **状態**: 20 本完走（2026-11-09）。

## 結果の解説

- 評価値: [results_summary_1109.md](../results/reports/summary_2026-11-09/results_summary_1109.md)
- 勾配帰属（big の観測 / 遷移比は 2.5〜3.0）: [GRAD_REPORT.md §8.2](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)

## 結論メモ

- tagr: big SVGD 2.49 / 2.33 / 2.65 / 4.27 / 4.14（平均 3.2）、big ULA 2.21 / 2.31 / 2.51 / 4.17 / 4.77（3.2）。K16 の light 変種（2.8〜3.4）と同程度で GRU 4.2 に届かない。表現の次元は律速ではない。
- mirror2: big SVGD 平均 −388（K16 light −368 と同等）、big ULA は s13 が崩壊（−1906）。
