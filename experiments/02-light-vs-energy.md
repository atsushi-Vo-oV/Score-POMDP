# 02 スコアを直接出す light は energy / PF と比べてどうか

- **問い**: スコアをエネルギーの勾配ではなく直接出す direct パラメータ化（light、計算約 1/3）は、energy 版と基準線（特に PF）に対してどうか。
- **腕（campaign）**: `trysvgdlight-<task>-tf-{ula,svgd,smi}-v1`（素）、比較対象 `trysvgd-<task>-tf-*-v1`（energy、[01](01-sampler-energy.md)）、`trysvgd-<task>-base-v2`、PF v3（[06](06-pf-observation-function.md)）。tagr の素 light SVGD は [12](12-obs-leak.md) の `trysvgdlight-tagr-tf-svgd-ctl-v1`。
- **課題**: mirror2 / tps / tagr。
- **設定とジョブ**: `../config/trysvgdlight_<task>_tf-*.json`、`../jobs/trysvgd_light/`。
- **状態**: mirror2 / tps 完走。tagr の素 light SVGD は leak 比較の ctl 腕として走行待ち（v5 木の同設定 5 本も待ち行列に残る）。

## 結果の解説

- light vs PF（2026-11-04、Holm 補正）: [light_vs_pf_1104.md](../results/reports/lightpf_2026-11-04/light_vs_pf_1104.md)、Codex 再解析 [LIGHTPF_REPORT.md](../results/reports/lightpf_2026-11-04/LIGHTPF_REPORT.md)
- 12 段の輸送の性質（段数診断）: [STEPS_RESULT.md](../results/reports/steps_2026-11-04/STEPS_RESULT.md)、[IMPROVE_REPORT.final.md](../results/reports/steps_2026-11-04/IMPROVE_REPORT.final.md)
- 埋め込みの解析（2026-11-03）: [ANALYSIS_REPORT_astra.md](../results/reports/belief_2026-11-03/ANALYSIS_REPORT_astra.md)、[ANALYSIS_REPORT_terra.md](../results/reports/belief_2026-11-03/ANALYSIS_REPORT_terra.md)
- direct 場の curl（反対称成分 0.47〜0.57、energy は 0）: [GRAD_REPORT.md §7.4](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)
- 評価値一覧: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)

## 結論メモ

- mirror2: light SVGD −368 は PF v1（5/5 崩壊）より良いが GRU に届かず、11 比較の Holm 補正では PF より有意な腕なし。コストは PF の 5〜20 倍。
- light の 12 段は等速直線の輸送で、延長すると発散。energy では粒子がほとんど動かず読み出しはスコア値を使う。
