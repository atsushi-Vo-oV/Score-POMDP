# 03 try-svgd-alpha: α-vector 型の読み出し（score_alpha）

- **目的**: 信念に対して凸（区分線形）な価値読み出し（AlphaLSE ヘッド、alpha_pool 符号化器）を、同じ FLOPs の TF 読み出しと比べる。
- **課題**: tagr（math、v2）。mirror2 / tps は Genkai。
- **設定とジョブ**: `../config/trysvgdalpha_*.json`、`../jobs/trysvgd_alpha/`。campaign `trysvgdalpha-tagr-energy-{svgd,ula}-v2`。
- **状態（2026-11-07）**: tagr v2 完走（3 seed）。

## 結果の解説

- 設計と「連続行動では α-vector 方策ではない（actor は拡散、alpha_lse は critic のみ）」の説明: [score_alpha_20261102.pdf](../results/reports/score_alpha_20261102.pdf)
- 評価値: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)（energy-svgd 2.39、energy-ula 4.07、3 seed）

## 結論メモ

- ULA 版は seed 分散が大きい（5.43 / 2.45 / 4.34）。離散化や区分線形混合に進めるかは未決。
