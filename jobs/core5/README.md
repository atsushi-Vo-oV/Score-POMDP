# core5 — core4 に ULA の step size α を変えた score transformer 2 腕を足した比較

core5 は core4 の 4 手法(gru、rnn、DPFRL 粒子フィルタ、score transformer k16f α=0.02 固定)に、α だけを変えた score transformer を 2 腕足したもの。K=16、L=12、エネルギー MLP [8]、d_model 32、ff 64、τ=1、belief 設計 v2、PPO 設定、seeds 10–14 は core4 と同一なので、推論 FLOPs も同じ 1.12 MFLOP/step。gru / rnn / pf / α 0.02 の列は既存の core4 と core4mm の結果を使う。

| 腕 | α | 1 時刻の輸送予算 α·L | ULA 雑音 √(2ατL) |
|---|---|---|---|
| core4 tr(既存) | 0.02 固定 | 0.24 | 0.69 |
| **a02** | 0.2 固定 | 2.4 | 2.19 |
| **alearn** | 12 step 別の α_l を学習。初期値 0.02、sigmoid で [1e-4, 0.5] に制限、学習率 0.025(方策の 100 倍、pendbal は 25 倍) | 0.24 → 最大 6 | 0.69 → 最大 3.5 |

α 学習版の学習率を上げる理由: 方策と同じ学習率では α は 0.020 → 0.023 しか動かなかった(8 月の τ study)。100 倍 + sigmoid で初めて動いた(Light-Dark で 0.497 まで)。clamp 境界は範囲外で勾配が消えて戻れないので sigmoid を使う。学習中の α は `.out` の `langevin step_size update=N mean= min= max=` 行で追える。

| task | config | campaign | shards | updates | 見込み時間/腕 | elapse |
|---|---|---|---|---|---|---|
| mh | config/core5_mh_a02.json | core5-mh-a02-v1 | 5 | 200 | ≈107 h | 120:00:00 |
| mh | config/core5_mh_alearn.json | core5-mh-alearn-v1 | 5 | 200 | ≈107 h | 120:00:00 |
| ld2d | config/core5_ld2d_a02.json | core5-ld2d-a02-v1 | 5 | 200 | ≈93 h | 120:00:00 |
| ld2d | config/core5_ld2d_alearn.json | core5-ld2d-alearn-v1 | 5 | 200 | ≈93 h | 120:00:00 |
| pendbal | config/core5_pendbal_a02.json | core5-pendbal-a02-v1 | 5 | 200 | ≈350 h(≈3 segment) | 168:00:00 |
| pendbal | config/core5_pendbal_alearn.json | core5-pendbal-alearn-v1 | 5 | 200 | ≈350 h(≈3 segment) | 168:00:00 |
| mirror | config/core5_mirror_a02.json | core5-mirror-a02-v1 | 5 | 200 | ≈113 h | 168:00:00 |
| mirror | config/core5_mirror_alearn.json | core5-mirror-alearn-v1 | 5 | 200 | ≈113 h | 168:00:00 |
| mirror2 | config/core5_mirror2_a02.json | core5-mirror2-a02-v1 | 5 | 200 | ≈113 h | 168:00:00 |
| mirror2 | config/core5_mirror2_alearn.json | core5-mirror2-alearn-v1 | 5 | 200 | ≈113 h | 168:00:00 |
| tps | config/core5_tps_a02.json | core5-tps-a02-v1 | 5 | 100 | ≈52 h | 72:00:00 |
| tps | config/core5_tps_alearn.json | core5-tps-alearn-v1 | 5 | 100 | ≈52 h | 72:00:00 |

tps は core4 と比べられるように core4 と同じ 100 updates。

`core4_ld2d_pf_s13.sh` は core4 の Light-Dark 2D DPFRL seed 13 の再実行(旧ツリーで update 3 に非有限勾配ノルムで停止; この木は該当 minibatch をスキップする)。

```bash
pjsub jobs/core5/debug_core5.sh   # 合格後にゲートが投入(gate_core5.sh <JID> all|nopendbal|<task>|ld2dpf)
```
