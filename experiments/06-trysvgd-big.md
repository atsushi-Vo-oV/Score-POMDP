# 06 try-svgd-big: 粒子数 × 潜在次元 = GRU の隠れ次元

- **目的**: 「表現の次元が足りない」説の検証。粒子数 K と潜在次元 D の積を GRU の隠れ次元に合わせる（mirror2 80 × 2、tagr 36 × 6）light ULA / SVGD。
- **課題**: mirror2 / tagr。
- **設定とジョブ**: `../config/trysvgdbig_*.json`、`../jobs/trysvgd_big/`（20 本、commit 5138bff）。
- **状態（2026-11-07）**: mirror2 SVGD 3 本が走行中、残り 17 本は math の待ち行列（leak セットの直後）。1 update は tagr 40〜55 秒、mirror2 140〜170 秒。

## 結果の解説

- 評価値（途中）: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)（campaign `trysvgdbig-*`）
- 勾配帰属での位置づけ（big は観測 / 遷移の勾配比が 2.5〜3.0 と大きめ）: [GRAD_REPORT.md §8.2](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)

## 結論メモ

- 結果待ち。
