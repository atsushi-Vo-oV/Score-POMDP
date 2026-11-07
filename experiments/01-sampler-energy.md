# 01 サンプラーの違いは効くか（energy スコア信念 vs FLOPs 一致の基準線）

- **問い**: energy パラメータ化のスコア粒子信念で、ULA / SVGD / SMI のどれが良いか。FLOPs をそろえた GRU / RNN / PF に届くか。
- **腕（campaign）**: `trysvgd-<task>-tf-{ula,svgd,smi}-v1`、基準線 `trysvgd-<task>-base-v2`（GRU / RNN / PF v2、2026-11-02 の FLOPs 再計数後の幅）、tagr は `trysvgd-tagr-tf-{ula,svgd}-v2`。
- **課題**: mirror2（333 update）、tps（213）、tagr（1293）。5 seed（tagr の TF は 3）。
- **設定とジョブ**: `../config/trysvgd_<task>_*.json`、`../jobs/trysvgd/`、`../jobs/trysvgd_tagr/`。
- **状態**: mirror2 / tps / tagr の ULA・SVGD と基準線は完走。SMI は mirror2 / tps の一部と tagr が待ち行列。

## 結果の解説

- 評価値一覧: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)
- 中間報告: [trysvgd_interim_20261102.pdf](../results/reports/trysvgd_interim_20261102.pdf)、[trysvgd_interim_20261105.pdf](../results/reports/trysvgd_interim_20261105.pdf)
- FLOPs 一致の根拠と計数漏れ（Codex 2026-11-02）: [codex-review-20261102.md](../agent/codex-review-20261102.md)、実測 [flops_tagre_1107.json](../results/reports/codex_why_2026-11-07/flops_tagre_1107.json)
- 統計（Holm 補正）と原因候補: [WHY_REPORT.md](../results/reports/codex_why_2026-11-07/WHY_REPORT.md)
- 設計メモ: [belief-aux-losses.pdf](../docs/belief-aux-losses.pdf)、関連研究 [related-work.pdf](../docs/related-work.pdf)

## 結論メモ

- tagr: GRU 4.2、PF v3 3.5、RNN 3.2 に対し energy TF-SVGD 2.3、TF-ULA 2.5（3 seed）。energy tagr の方策は加速ゼロで止まる seed が多い。
- mirror2: energy TF 系は PF v1 より良いが GRU には届かない。tps: 全腕同等。
