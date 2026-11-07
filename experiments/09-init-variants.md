# 09 初期化変種: SVGD 連鎖の開始分布を学ぶ（param / func / prev）

- **目的**: 毎時刻 N(0, I) から始める連鎖の開始を学習すれば信念が良くなるか。param = 大域の N(μ, σ) を学習、func = 前粒子・前行動・現在観測から粒子ごとに出す網、prev = 前時刻の粒子から開始（warm start）。
- **課題**: mirror2 / tps / tagr × 5 seed = 45 本。
- **設定とジョブ**: `model.langevin_proposal` ∈ {noise, affine, network}、`langevin_warm_start`、`../config/trysvgdlight_<task>_tf-svgd-{param,func,prev}-init.json`、`../jobs/trysvgd_light/*-{param,func,prev}init_s*.sh`。
- **状態（2026-11-07）**: 完走。

## 結果の解説

- 「学習不足か」の検証（Codex、2026-11-05〜06）: [ASTRA1_VERIFY_REPORT.md](../results/reports/codex_2026-11-06/ASTRA1_VERIFY_REPORT.md)、記憶を使わない理由 [ASTRA2_MEMORY_REPORT.md](../results/reports/codex_2026-11-06/ASTRA2_MEMORY_REPORT.md)、統合報告 [JOINT_REPORT.md](../results/reports/codex_2026-11-06/JOINT_REPORT.md)、近傍検査 [NEIGHBORHOOD_RESULTS.md](../results/reports/codex_2026-11-06/NEIGHBORHOOD_RESULTS.md)
- 開始分布の学習量と勾配爆発（func は Adam の不足 step 38〜504、全体勾配 10〜10¹⁰）: [GRAD_REPORT.md §2–3](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)
- 信念の復号（MDN）: [MDN_REPORT.md](../results/reports/mdn_2026-11-07/MDN_REPORT.md)、ビューア https://claude.ai/artifact/9DeCqi4xsok7n1VCPY5Rf3
- 評価値: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)

## 結論メモ

- param は素と同じ（μ ≈ 0、σ 0.8〜1.0 のまま。PPO は提案分布を押していない）。func は full BPTT で勾配爆発し seed 差が大きい。prev は座標が流れて復号も崩れる。
- 復号: 粒子集合は現在観測に含まれる量だけを符号化し、隠れ変数（鏡の符号、鬼の位置）は復号できない。
