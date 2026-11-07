# 03 joint 変種（予測と PPO の同時学習）は効くか

- **問い**: 予測フェーズと PPO を交互にする既定に対し、同時に学ぶ joint 版（light ULA / SVGD / SMI）は良くなるか。
- **腕（campaign）**: `trysvgdlight-<task>-tf-{ula,svgd,smi}-joint-v1` vs 素の light（[02](02-light-vs-energy.md)）。
- **課題**: mirror2 / tps / tagr。
- **状態**: mirror2 / tps 完走、tagr は待ち行列（3 + 2 seed）。
- **設定とジョブ**: `../config/trysvgdlight_<task>_tf-*-joint.json`、`../jobs/trysvgd_light/*joint*`。

## 結果の解説

- 評価値一覧: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)
- 勾配帰属（joint の観測 / 遷移比など）: [GRAD_REPORT.md §8.2](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)

## 結論メモ

- mirror2 では SVGD-joint は素の light SVGD と同等（light_vs_pf の表）。tagr は結果待ち。
