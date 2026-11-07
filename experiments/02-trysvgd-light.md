# 02 try-svgd-light: direct スコア（light）の ULA / SVGD / SMI と変種

- **目的**: スコアをエネルギーの勾配ではなく直接出す「light」版（計算が約 1/3）で、サンプラーと変種（joint、prior、noanchor、flat）を基準線と比べる。
- **課題**: mirror2 / tps / tagr（update は 01 と同じ）。
- **設定とジョブ**: `../config/trysvgdlight_<task>_*.json`、`../jobs/trysvgd_light/`。campaign `trysvgdlight-<task>-tf-<sampler>[-variant]-v1`。
- **状態（2026-11-07）**: mirror2 / tps は完走。tagr は変種 6 種（param / func / prev / prior / noanchor / flat）が完走、素の light tagr は [11 leak セット](11-obs-leak.md) の ctl 腕として走る。

## 結果の解説

- light vs PF の比較（2026-11-04、Holm 補正あり）: [light_vs_pf_1104.md](../results/reports/lightpf_2026-11-04/light_vs_pf_1104.md)、Codex の独立再解析 [LIGHTPF_REPORT.md](../results/reports/lightpf_2026-11-04/LIGHTPF_REPORT.md)
- 12 段は少なすぎるか（段数診断、2026-11-04）: [STEPS_RESULT.md](../results/reports/steps_2026-11-04/STEPS_RESULT.md)、改善案 [IMPROVE_REPORT.final.md](../results/reports/steps_2026-11-04/IMPROVE_REPORT.final.md)
- 信念の中身（埋め込みの解析、2026-11-03）: [ANALYSIS_REPORT_astra.md](../results/reports/belief_2026-11-03/ANALYSIS_REPORT_astra.md)、[ANALYSIS_REPORT_terra.md](../results/reports/belief_2026-11-03/ANALYSIS_REPORT_terra.md)
- 全 run の最新評価: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)
- ビューア（粒子・復号・run 横断表）: https://claude.ai/artifact/9DeCqi4xsok7n1VCPY5Rf3

## 結論メモ

- mirror2: light SVGD −368 は PF v1（5/5 崩壊）より良いが GRU には届かず、PF より有意に良い腕は Holm 補正後にはない。
- tagr: light 変種は 2.8〜4.2（noanchor は s13 の 9.4 が引き上げ）、GRU 4.2 との差は 8 比較すべて非有意。
- 段数は律速ではない（0 / 6 / 12 / 24 段で同じ収益、スコア特徴をゼロにすると崩壊）。
