# 01 try-svgd（本体）: energy スコア信念 × サンプラー vs FLOPs 一致の基準線

- **目的**: energy パラメータ化のスコア粒子信念（TF-ULA / TF-SVGD / TF-SMI、transformer 読み出し）を、行動 1 step の FLOPs を transformer に合わせた GRU / RNN / 粒子フィルタ（DPFRL）と PPO で比べる。
- **課題**: mirror2（333 update）、tps（213 update）、tagr（1293 update）。
- **設定とジョブ**: `../config/trysvgd_<task>_{tf-ula,tf-svgd,tf-smi,base}.json`、`../jobs/trysvgd/`（基準線 v2 = 2026-11-02 の FLOPs 再計数後の幅）、`../jobs/trysvgd_tagr/`。
- **状態（2026-11-07）**: mirror2 / tps / tagr の TF-SVGD・ULA と基準線 v2 は完走。SMI の一部は math の待ち行列。

## 結果の解説

- 全 run の最新評価（決定的 mean_chain / 確率的）: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)
- 中間報告 PDF: [trysvgd_interim_20261102.pdf](../results/reports/trysvgd_interim_20261102.pdf)、[trysvgd_interim_20261105.pdf](../results/reports/trysvgd_interim_20261105.pdf)
- FLOPs 一致の根拠と計数漏れの指摘（Codex、2026-11-02）: [codex-review-20261102.md](../agent/codex-review-20261102.md)、計測値 [flops_tagre_1107.json](../results/reports/codex_why_2026-11-07/flops_tagre_1107.json)
- 統計的な勝敗判定（Holm 補正、tagr 8 比較で有意差なし）: [WHY_REPORT.md §2.5](../results/reports/codex_why_2026-11-07/WHY_REPORT.md)
- 補助損失の設計メモ: [belief-aux-losses.pdf](../docs/belief-aux-losses.pdf)、関連研究: [related-work.pdf](../docs/related-work.pdf)

## 結論メモ

- tagr では GRU 4.2、PF v3 3.5、RNN 3.2、energy TF-SVGD 2.3（3 seed）。mirror2 では energy TF 系は PF v1 より良いが GRU には届かない。tps は全腕同等。
- `results/` は git 管理外なので、上のリンクは手元の作業ツリーで開く。
