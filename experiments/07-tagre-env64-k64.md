# 07 tagre env64 / k64: 向き依存の楕円距離観測、環境 64、粒子 64、PPO 2 倍

- **目的**: 距離を体座標の楕円（前方 1.0、横 0.6）で測ることで、体を回して見回す能動的センシングが要る課題にする。GPU の 1 カーネルの仕事を太らせるため環境 64 / 粒子 64。
- **課題**: tagre（`observation_mode=elliptic_ranges`、2420 update = 25 + 2000 + 395）。
- **設定とジョブ**: `../config/trysvgd_tagre_{gru,pf_v3,tf-svgd}.json`、`../config/trysvgdlight_tagre_tf-svgd{,-func-init,-param-init}.json`、`../jobs/trysvgd_tagre/`（README に詳細）。
- **状態（2026-11-07）**: GRU 2 本完走（評価 7.05 / 4.06）、func-init s10 完走（1.99、update 1500 の 5.7 から崩壊）、PF v3 / param-init / func-init s11 / 素 light s10・s11 走行中。energy TF-SVGD は未投入。
- **注意**: **FLOPs は未一致**。K = 64 でスコア腕が GRU の 9〜13 倍（light 20.0 / energy 27.9 / PF v3 6.4 / GRU 2.16 MFLOP）。比較には K16 対照か幅合わせ GRU（encoder [3810, 1467] / hidden 700）が要る。

## 結果の解説

- FLOPs 計測と再一致幅: [flops_tagre_1107.json](../results/reports/codex_why_2026-11-07/flops_tagre_1107.json)、[rematch_math.json](../results/reports/codex_why_2026-11-07/rematch_math.json)、議論 [WHY_REPORT.md §2.2–2.3](../results/reports/codex_why_2026-11-07/WHY_REPORT.md)
- 評価値（途中）: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)
- func-init の勾配爆発（tagre でも 10¹⁰ 台）: [GRAD_REPORT.md §2.2](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)

## 結論メモ

- 訓練中の `episode_return_mean` は評価値ではない（GRU 7.40 / 5.00 は訓練指標、評価は 7.05 / 4.06）。
