# ld2d_v2 — Light-Dark 2D での score 設計格子(v1〜v4)× 5 seed

基底は config/ld2d_p3o_symlog.json(LD2D-P3O 課題: light_position 5.0、initial_std 1.0、goal_bonus 0、horizon 30、symlog 目標、200 updates)。v1 は 6 手法(pf は原典準拠 dpfrl、パラメータ一致 [108,192]、mgf 8)、v2〜v4 は score 3 エンコーダ。seeds 10〜14。shard index = method-major × seed。

| config | methods | seeds | shards |
|---|---|---|---|
| ld2d_v1.json | 6 手法 | 10–14 | 30 |
| ld2d_v2.json | score × 3 | 10–14 | 15 |
| ld2d_v3.json | score × 3 | 10–14 | 15 |
| ld2d_v4.json | score × 3 | 10–14 | 15 |

```bash
pjsub jobs/ld2d_v2/debug_ld2d.sh   # 合格後にゲートが 75 本を投入
```

## 別エンコーダ(critic 専用)比較 — sepenc

`model.value_encoder = separate`: 価値ヘッドが粒子集合を自前の集合エンコーダで読む(方策エンコーダと belief は共有、価値損失は方策エンコーダに届かない)。v2 設定 × 3 エンコーダ × seeds 10–14 = 15 本(config/ld2d_v2_sepenc.json、campaign ld2d-v2-sepenc-v1)。DEBUG: jobs/mountain_hike_v2/debug_sepenc.sh(両課題共通)。

## 集合混合密度の補助予測網 — v5 / v6

`model.aux_predictor_kind = set_mixture`: 補助損失の予測網を「粒子ごとのガウスの等重み平均」から「粒子集合を DeepSets で読み、4 成分の対角ガウス混合を出す MDN」に置き換える(観測・報酬とも、隠れ層 [64, 64])。v5 = v2 + set_mixture(warm + アンカー + 補助)、v6 = v3 + set_mixture(cold + 補助)。3 エンコーダ × seeds 10–14、config/ld2d_v5.json / ld2d_v6.json、campaign ld2d-v5-v1 / -v6-v1。DEBUG: jobs/mountain_hike_v2/debug_mdn.sh(両課題共通)。

## FLOPs 一致版 — v2lf

v2 の連鎖を K=4 粒子・L=2 ステップ・エネルギー MLP [32,32] に縮めたもの(推論 0.68–0.72 MFLOP/step。gru 0.46、PF 原典準拠 0.87 の間。v2 本来は 14.7–16.2 MFLOP)。パラメータ数は transformer 72.5k、alpha 32.8k で既定とほぼ同じ。3 エンコーダ × seeds 10–14、campaign *-v2lf-v1。DEBUG: jobs/mountain_hike_v2/debug_lf.sh(3 課題共通)。

## FLOPs 一致版 — v2lf

v2 の連鎖を K=4 粒子・L=2 ステップ・エネルギー MLP [32,32] に縮めたもの(推論 0.68–0.72 MFLOP/step。gru 0.46、PF 原典準拠 0.87 の間。v2 本来は 14.7–16.2 MFLOP)。パラメータ数は transformer 72.5k、alpha 32.8k で既定とほぼ同じ。3 エンコーダ × seeds 10–14、campaign *-v2lf-v1。DEBUG: jobs/mountain_hike_v2/debug_lf2.sh(3 課題共通)。
