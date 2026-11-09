# 04 連鎖の開始分布を変えれば良くなるか（param / func / prev / uniform）

- **問い**: 毎時刻 N(0, I) から始める SVGD 連鎖の開始を変えれば信念が良くなるか。param = 大域の N(μ, σ)（学習）、func = 前粒子・前行動・現在観測から粒子ごとに出す網（学習）、prev = 前時刻の粒子から開始（warm start）、uniform = 固定の一様分布 U(−√3, √3)（分散は N(0, I) と同じ、台が有界。学習なし）。
- **腕（campaign）**: `trysvgdlight-<task>-tf-svgd-{param,func,prev,uniform}-init-v1` vs 素の light SVGD。Car-Flag では基準線 `trysvgd-carflag-{gru,pf-v3}-v1` も同時に走らせる（幅は light TF-SVGD の 1.07 MFLOP に一致: GRU [800, 308] / 147、PF v3 [181, 322]）。
- **課題**: mirror2 / tps / tagr × 5 seed = 60 本（uniform 15 本を 2026-11-10 に追加）＋ **Car-Flag**（2026-11-10 追加、ユーザー「多峰性があるタスクをさらに追加」→ Car-Flag を選択）: light 5 腕 + GRU + PF v3 × 5 seed = 35 本。Car-Flag は緑旗が +1 か −1 かの 2 値の隠れ変数を、青旗（priest、0.5 ± 0.2）の近くでしか観測できない課題。信念は 2 点の混合で、観測後はその符号を旗まで保持する必要がある（離散モードの記憶）。
- **設定とジョブ**: `model.langevin_proposal` ∈ {noise, affine, network, uniform}、`langevin_warm_start`、`langevin_uniform_half_width`（既定 √3）、`../config/trysvgdlight_<task>_tf-svgd-{param,func,prev,uniform}-init.json`、`../jobs/trysvgd_light/*-{param,func,prev,uniform}init_s*.sh`。Car-Flag は `../config/trysvgdlight_carflag_tf-svgd*.json`、`../config/trysvgd_carflag_{gru,pf_v3}.json`、`../jobs/trysvgd_light/trysvgdlight_carflag_*.sh`、`../jobs/trysvgd/trysvgd_carflag_*.sh`（333 update、horizon 160、観測ノイズ 0）。
- **状態**: param / func / prev は完走。uniform は 2026-11-10 01:10 に math へ投入（seed 10–12 の 9 本が未実行分の先頭、seed 13–14 の 6 本が最後尾。木 v18）。Car-Flag 35 本は 2026-11-10 02:00 に math へ投入（木 v19。seed 10–12 の 21 本が uniform-init s10–12 の直後、seed 13–14 の 14 本が最後尾）。

## 結果の解説

- 学習不足か（Codex #1）: [ASTRA1_VERIFY_REPORT.md](../results/reports/codex_2026-11-06/ASTRA1_VERIFY_REPORT.md)
- 記憶を使わない理由（Codex #2）: [ASTRA2_MEMORY_REPORT.md](../results/reports/codex_2026-11-06/ASTRA2_MEMORY_REPORT.md)
- 統合報告と近傍検査: [JOINT_REPORT.md](../results/reports/codex_2026-11-06/JOINT_REPORT.md)、[NEIGHBORHOOD_RESULTS.md](../results/reports/codex_2026-11-06/NEIGHBORHOOD_RESULTS.md)
- 重みの動きと勾配爆発（func は全体勾配 10〜10¹⁰、Adam 不足 step 38〜504）: [GRAD_REPORT.md §2–3](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)
- 復号（MDN）: [MDN_REPORT.md](../results/reports/mdn_2026-11-07/MDN_REPORT.md)、ビューア https://claude.ai/artifact/9DeCqi4xsok7n1VCPY5Rf3
- 評価値一覧: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)

- mirror2 の腕ごとの動きの比較（軌跡・m 別リターン・信念の動き・学習曲線、日英）: https://claude.ai/artifact/12CJkE2XUB5wLjhk8a7Zh9

## 結論メモ

- param は素と同じ（μ ≈ 0、σ 0.8〜1.0、PPO は提案分布を押していない）。func は full BPTT で勾配爆発し seed 差が大きい。prev は座標が流れ（最大 2419）復号も崩れる。
- tagr の評価: func 3.41、param 3.05、prev 2.77（GRU 4.21）。
