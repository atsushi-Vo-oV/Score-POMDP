# 07 凸な α-vector 型読み出しは TF 読み出しに勝つか

- **問い**: 信念に対して凸（区分線形）な価値読み出し（AlphaLSE ヘッド、alpha_pool 符号化器）は transformer 読み出しより良いか。
- **腕（campaign）**: `trysvgdalpha-tagr-energy-{svgd,ula}-v2`（3 seed）vs `trysvgd-tagr-tf-{svgd,ula}-v2`（[01](01-sampler-energy.md)）。mirror2 / tps と direct 版は待ち行列。
- **設定とジョブ**: `../config/trysvgdalpha_*.json`、`../jobs/trysvgd_alpha/`。幅は同じ種類の TF に FLOPs 一致。
- **状態**: tagr energy 完走。

## 結果の解説

- 設計と注意（連続行動では actor は拡散方策、alpha_lse は critic のみ）: [score_alpha_20261102.pdf](../results/reports/score_alpha_20261102.pdf)
- 評価値一覧: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)

## 結論メモ

- tagr: energy-ULA 4.07（5.43 / 2.45 / 4.34）、energy-SVGD 2.39。TF-ULA 2.45、TF-SVGD 2.26 より上だが seed 分散が大きい。
