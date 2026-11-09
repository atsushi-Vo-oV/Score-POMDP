# 06 PF 基準線の観測適合度を非線形にすると（v1 / v2 / v3 / v3b）

- **問い**: DPFRL 粒子フィルタ基準線の観測適合度 `Linear([粒子, 観測特徴])` は観測項が粒子間で共通になり log_softmax で相殺する（構造的欠陥）。MLP（v3）/ 双線形（v3b）に直すと PF はどこまで伸びるか。
- **腕（campaign）**: `trysvgd-<task>-base-v2`（PF v2、線形）、`trysvgd-<task>-pf-v3`（mlp）、`trysvgd-mirror2-pf-v3b`（bilinear）。PF v1 は旧木。
- **課題**: mirror2 / tps / tagr × 5 seed。
- **設定とジョブ**: `comparison.pf_observation_function`、`../config/trysvgd_<task>_pf_v3.json`、`../jobs/trysvgd/*_pf_v3*`、`../jobs/trysvgd_tagr/*_pf_v3_s*.sh`。幅は energy TF-ULA に再一致。
- **状態**: 完走。

## 結果の解説

- 欠陥の指摘: [LIGHTPF_REPORT.md](../results/reports/lightpf_2026-11-04/LIGHTPF_REPORT.md)
- 評価値一覧: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)
- 復号（PF v3 の自己位置は最良、鬼の距離は GRU に劣る）: [MDN_REPORT.md](../results/reports/mdn_2026-11-07/MDN_REPORT.md)

- mirror2 の腕ごとの動きの比較（軌跡・m 別リターン・信念の動き・学習曲線、日英）: https://claude.ai/artifact/12CJkE2XUB5wLjhk8a7Zh9

## 結論メモ

- tagr: v3 3.53 > v2 3.02。mirror2: v3 −1180、v3b −1336 で v2 −770 より悪い。tps: 同等。
