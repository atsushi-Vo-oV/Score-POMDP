# mountain_hike_v2 — score v2: 粒子を記憶担体にする再設計 + ハイパラ掃引 + 軽量網

共通(v2): `langevin_warm_start`(連鎖初期値 = 前粒子)、`langevin_observation_anchor`(エピソード先頭の連鎖初期値 = μ(o₀)+σ(o₀)ξ、ゼロ初期化)、`reward_prediction_coef=1`(粒子+行動 → 観測報酬の混合ガウス尤度)、`observation_prediction_coef=1`(次観測)。v2g はさらに `langevin_transition_proposal`(x_t⁰ = x_{t−1} + g(x_{t−1}, a_{t−1})、ゼロ初期化)。状態は学習にも評価にも使わない(方策勾配 + 自身の観測・報酬列のみ)。

| config | method | updates | 変更点 |
|---|---|---|---|
| mountain_hike_v2.json | score_transformer | 200 | warm start + observation anchor + reward/obs aux (no drift g) |
| mountain_hike_v2.json | score_alpha | 200 | warm start + observation anchor + reward/obs aux (no drift g) |
| mountain_hike_v2.json | score_deepsets | 200 | warm start + observation anchor + reward/obs aux (no drift g) |
| mountain_hike_v2g.json | score_transformer | 200 | v2 + learned transition drift g |
| mountain_hike_v2g.json | score_alpha | 200 | v2 + learned transition drift g |
| mountain_hike_v2g.json | score_deepsets | 200 | v2 + learned transition drift g |
| mountain_hike_v2g_a01.json | score_transformer | 100 | v2g, ULA step 0.02 -> 0.1 |
| mountain_hike_v2g_a03.json | score_transformer | 100 | v2g, ULA step 0.02 -> 0.3 |
| mountain_hike_v2g_t03.json | score_transformer | 100 | v2g, temperature 1 -> 0.3 |
| mountain_hike_v2g_t01.json | score_transformer | 100 | v2g, temperature 1 -> 0.1 |
| mountain_hike_v2g_lr1e3.json | score_transformer | 100 | v2g, lr 2.5e-4 -> 1e-3 |
| mountain_hike_v2g_ent03.json | score_transformer | 100 | v2g, entropy 0.01 -> 0.03 |
| mountain_hike_v2g_rw10.json | score_transformer | 100 | v2g, aux coefs 1 -> 10 |
| mountain_hike_v2g_rw01.json | score_transformer | 100 | v2g, aux coefs 1 -> 0.1 |
| mountain_hike_v2g_ns.json | score_transformer | 100 | v2g, encoder sees particle positions only (score shortcut cut) |
| mountain_hike_v2g_liteL16.json | score_transformer | 100 | v2g, small networks (energy [16,16], d_model 32, policy [48,48]) with L 4 -> 16 |
| mountain_hike_v2g_liteL16.json | score_alpha | 100 | v2g, small networks (energy [16,16], d_model 32, policy [48,48]) with L 4 -> 16 |
| mountain_hike_v2g_liteK32.json | score_transformer | 100 | v2g, small networks with K 16 -> 32 and L 4 -> 8 |
| mountain_hike_v2g_liteK32.json | score_alpha | 100 | v2g, small networks with K 16 -> 32 and L 4 -> 8 |
| mountain_hike_ab_warm.json | score_transformer | 100 | 対照: warm のみ(アンカー・補助・g なし) |
| mountain_hike_ab_aux.json | score_transformer | 100 | 対照: 補助損失のみ(cold、アンカー・g なし) |
| mountain_hike_v3.json | transformer / alpha / deepsets | 200 | v3 = v2 − warm start − 観測アンカー(cold 連鎖 + 補助損失のみ、g なし) |
| mountain_hike_v4.json | transformer / alpha / deepsets | 200 | v4 = v3 の逆: warm start + 観測アンカーあり、補助損失なし(g なし) |

```bash
pjsub jobs/mountain_hike_v2/debug_v2.sh        # 15 本の gate
pjsub jobs/mountain_hike_v2/debug_v2_lite.sh   # lite 4 本の gate(本番サイズ 1 update)
# 合格後:
pjsub jobs/mountain_hike_v2/v2_score.sh
pjsub jobs/mountain_hike_v2/v2_alpha.sh
pjsub jobs/mountain_hike_v2/v2_ds.sh
pjsub jobs/mountain_hike_v2/v2g_score.sh
pjsub jobs/mountain_hike_v2/v2g_alpha.sh
pjsub jobs/mountain_hike_v2/v2g_ds.sh
pjsub jobs/mountain_hike_v2/v2g_a01_score.sh
pjsub jobs/mountain_hike_v2/v2g_a03_score.sh
pjsub jobs/mountain_hike_v2/v2g_t03_score.sh
pjsub jobs/mountain_hike_v2/v2g_t01_score.sh
pjsub jobs/mountain_hike_v2/v2g_lr1e3_score.sh
pjsub jobs/mountain_hike_v2/v2g_ent03_score.sh
pjsub jobs/mountain_hike_v2/v2g_rw10_score.sh
pjsub jobs/mountain_hike_v2/v2g_rw01_score.sh
pjsub jobs/mountain_hike_v2/v2g_ns_score.sh
pjsub jobs/mountain_hike_v2/v2g_liteL16_score.sh
pjsub jobs/mountain_hike_v2/v2g_liteL16_alpha.sh
pjsub jobs/mountain_hike_v2/v2g_liteK32_score.sh
pjsub jobs/mountain_hike_v2/v2g_liteK32_alpha.sh
```

## 別エンコーダ(critic 専用)比較 — sepenc

`model.value_encoder = separate`: 価値ヘッドが粒子集合を自前の集合エンコーダで読む(方策エンコーダと belief は共有、価値損失は方策エンコーダに届かない)。v2 設定 × 3 エンコーダ × seeds 10–14 = 15 本(config/mountain_hike_v2_sepenc.json、campaign mountain-hike-v2-sepenc-v1)。DEBUG: jobs/mountain_hike_v2/debug_sepenc.sh(両課題共通)。

## 集合混合密度の補助予測網 — v5 / v6

`model.aux_predictor_kind = set_mixture`: 補助損失の予測網を「粒子ごとのガウスの等重み平均」から「粒子集合を DeepSets で読み、4 成分の対角ガウス混合を出す MDN」に置き換える(観測・報酬とも、隠れ層 [64, 64])。v5 = v2 + set_mixture(warm + アンカー + 補助)、v6 = v3 + set_mixture(cold + 補助)。3 エンコーダ × seeds 10–14、config/mountain_hike_v5.json / mountain_hike_v6.json、campaign mountain-hike-v5-v1 / -v6-v1。DEBUG: jobs/mountain_hike_v2/debug_mdn.sh(両課題共通)。

## FLOPs 一致版 — v2lf

v2 の連鎖を K=4 粒子・L=2 ステップ・エネルギー MLP [32,32] に縮めたもの(推論 0.68–0.72 MFLOP/step。gru 0.46、PF 原典準拠 0.87 の間。v2 本来は 14.7–16.2 MFLOP)。パラメータ数は transformer 72.5k、alpha 32.8k で既定とほぼ同じ。3 エンコーダ × seeds 10–14、campaign *-v2lf-v1。DEBUG: jobs/mountain_hike_v2/debug_lf.sh(3 課題共通)。

## FLOPs 一致版 — v2lf

v2 の連鎖を K=4 粒子・L=2 ステップ・エネルギー MLP [32,32] に縮めたもの(推論 0.68–0.72 MFLOP/step。gru 0.46、PF 原典準拠 0.87 の間。v2 本来は 14.7–16.2 MFLOP)。パラメータ数は transformer 72.5k、alpha 32.8k で既定とほぼ同じ。3 エンコーダ × seeds 10–14、campaign *-v2lf-v1。DEBUG: jobs/mountain_hike_v2/debug_lf2.sh(3 課題共通)。
