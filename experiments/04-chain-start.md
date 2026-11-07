# 04 連鎖の開始分布を学べば良くなるか（param / func / prev）

- **問い**: 毎時刻 N(0, I) から始める SVGD 連鎖の開始を学習すれば信念が良くなるか。param = 大域の N(μ, σ)、func = 前粒子・前行動・現在観測から粒子ごとに出す網、prev = 前時刻の粒子から開始（warm start）。
- **腕（campaign）**: `trysvgdlight-<task>-tf-svgd-{param,func,prev}-init-v1` vs 素の light SVGD。
- **課題**: mirror2 / tps / tagr × 5 seed = 45 本。
- **設定とジョブ**: `model.langevin_proposal` ∈ {noise, affine, network}、`langevin_warm_start`、`../config/trysvgdlight_<task>_tf-svgd-{param,func,prev}-init.json`、`../jobs/trysvgd_light/*-{param,func,prev}init_s*.sh`。
- **状態**: 完走。

## 結果の解説

- 学習不足か（Codex #1）: [ASTRA1_VERIFY_REPORT.md](../results/reports/codex_2026-11-06/ASTRA1_VERIFY_REPORT.md)
- 記憶を使わない理由（Codex #2）: [ASTRA2_MEMORY_REPORT.md](../results/reports/codex_2026-11-06/ASTRA2_MEMORY_REPORT.md)
- 統合報告と近傍検査: [JOINT_REPORT.md](../results/reports/codex_2026-11-06/JOINT_REPORT.md)、[NEIGHBORHOOD_RESULTS.md](../results/reports/codex_2026-11-06/NEIGHBORHOOD_RESULTS.md)
- 重みの動きと勾配爆発（func は全体勾配 10〜10¹⁰、Adam 不足 step 38〜504）: [GRAD_REPORT.md §2–3](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)
- 復号（MDN）: [MDN_REPORT.md](../results/reports/mdn_2026-11-07/MDN_REPORT.md)、ビューア https://claude.ai/artifact/9DeCqi4xsok7n1VCPY5Rf3
- 評価値一覧: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)

## 結論メモ

- param は素と同じ（μ ≈ 0、σ 0.8〜1.0、PPO は提案分布を押していない）。func は full BPTT で勾配爆発し seed 差が大きい。prev は座標が流れ（最大 2419）復号も崩れる。
- tagr の評価: func 3.41、param 3.05、prev 2.77（GRU 4.21）。
