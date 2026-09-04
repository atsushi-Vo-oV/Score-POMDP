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
