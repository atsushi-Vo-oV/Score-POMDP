# 10 楕円距離観測（能動的センシング）で信念は効くか

- **問い**: 距離を体座標の楕円（前方 1.0、横 0.6）で測ると、体を回して見回す必要が生まれる。この課題で信念（light 変種、PF v3）は GRU に対してどうか。
- **腕（campaign）**: `trysvgd-tagre-gru-env64-k64-v1`、`trysvgd-tagre-pf-v3-env64-k64-v1`、`trysvgdlight-tagre-tf-svgd-{func-init,param-init}-env64-k64-v1`、`trysvgdlight-tagre-tf-svgd-env64-k64-v1`（素）。energy `trysvgd-tagre-tf-svgd-env64-k64-v1` は未投入。
- **課題**: tagre（`observation_mode=elliptic_ranges`、環境 64、粒子 64、2420 update = 25 + 2000 + 395）。2 seed（10、11）。
- **設定とジョブ**: `../config/*tagre*.json`、`../jobs/trysvgd_tagre/`（README 参照）。
- **状態**: 2 seed とも全腕完走（2026-11-09）。energy TF-SVGD は未投入。
- **注意**: **FLOPs 未一致**。K = 64 でスコア腕は GRU の 9〜13 倍（light 20.0 / energy 27.9 / PF v3 6.4 / GRU 2.16 MFLOP）。比較には K16 対照か幅合わせ GRU（encoder [3810, 1467] / hidden 700）が要る。

## 結果の解説

- FLOPs 計測と再一致幅: [flops_tagre_1107.json](../results/reports/codex_why_2026-11-07/flops_tagre_1107.json)、[rematch_math.json](../results/reports/codex_why_2026-11-07/rematch_math.json)、議論 [WHY_REPORT.md §2.2–2.3](../results/reports/codex_why_2026-11-07/WHY_REPORT.md)
- 評価値（途中）: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)
- func-init の勾配爆発（tagre でも 10¹⁰ 台）: [GRAD_REPORT.md §2.2](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)

## 結果メモ（seed 10 / 11、決定的評価、[results_summary_1109.md](../results/reports/summary_2026-11-09/results_summary_1109.md)）

| 腕 | s10 | s11 | 平均 | 行動 FLOPs |
|---|---|---|---|---|
| GRU | 7.05 | 4.06 | 5.6 | 2.2 MFLOP |
| PF v3（K64） | 6.13 | 4.84 | 5.5 | 6.4 |
| light TF-SVGD 素（K64） | 5.00 | 5.47 | 5.2 | 20.0 |
| light param-init | 6.17 | 2.14 | 4.2 | 20.0 |
| light func-init | 1.99 | 6.87 | 4.4 | 20.3 |

## 結論メモ

- 楕円観測 + K64 + PPO 2 倍で light 素は 5.2 と GRU（5.6）に近づいたが、9 倍の計算を使って届かない。seed 分散（func 2.0 / 6.9、param 6.2 / 2.1）が腕間差より大きい。
- 訓練中の `episode_return_mean` は評価値ではない（GRU 7.40 / 5.00 は訓練指標、評価は 7.05 / 4.06）。
