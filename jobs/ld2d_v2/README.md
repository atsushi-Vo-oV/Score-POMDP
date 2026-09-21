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
