# core5 — core4 に ULA の step size α を変えた score transformer 2 腕を足し、情報収集タスク 3 つを加えた比較

core5 は core4 の 4 手法(gru、rnn、DPFRL 粒子フィルタ、score transformer k16f α=0.02 固定)に、α だけを変えた score transformer を 2 腕足したもの。K=16、L=12、エネルギー MLP [8]、d_model 32、ff 64、τ=1、belief 設計 v2、PPO 設定、seeds 10–14 は core4 と同一なので、推論 FLOPs も同じ 1.12 MFLOP/step。既存 6 タスクの gru / rnn / pf / α 0.02 の列は core4 と core4mm の結果を使う。

| 腕 | α | 1 時刻の輸送予算 α·L | ULA 雑音 √(2ατL) |
|---|---|---|---|
| core4 tr(既存) | 0.02 固定 | 0.24 | 0.69 |
| **a02** | 0.2 固定 | 2.4 | 2.19 |
| **alearn** | 12 step 別の α_l を学習。初期値 0.02、sigmoid で [1e-4, 0.5] に制限、学習率 0.025(方策の 100 倍、pendbal は 25 倍) | 0.24 → 最大 6 | 0.69 → 最大 3.5 |

α 学習版の学習率を上げる理由: 方策と同じ学習率では α は 0.020 → 0.023 しか動かなかった(8 月の τ study)。100 倍 + sigmoid で初めて動いた(Light-Dark で 0.497 まで)。clamp 境界は範囲外で勾配が消えて戻れないので sigmoid を使う。学習中の α は `.out` の `langevin step_size update=N mean= min= max=` 行で追える。

## 新タスク(情報収集が必要な POMDP)

| tag | 内容 | 出典 | 状態 / 観測 / 行動 | horizon |
|---|---|---|---|---|
| carflag | 1 次元の車。緑旗は +1 か −1(等確率)、赤旗は反対側。0.5 の青旗の ±0.2 以内でだけ緑の側が観測に出る。−0.01/step、緑 +1、赤 −1 | Nguyen et al. CoRL 2022、Baisero & Amato AAMAS 2022(公開実装 hai-h-nguyen/pomdp-domains の定数、報酬は論文の定義) | (位置, 速度, 側) / (位置, 速度/0.07, 合図) / 力 [−1, 1] | 160 |
| heavenhell3 | T 字の通路。天国は左右の端のどちらか(等確率)、地獄は反対。下の行き止まりの司祭でだけ天国の側が分かる。天国 +1、地獄 −1 | Geffner & Bonet 1998、Thrun 2000、gym-pomdps heavenhell_3(28 状態、15 観測) | (x, y, 側) / 観測記号の one-hot 15 / N S E W | 100 |
| heavenhell4 | 同上、通路長 4 | gym-pomdps heavenhell_4(36 状態、19 観測) | 同上、one-hot 19 | 100 |

Heaven-Hell の遷移・観測・報酬は gym-pomdps の .pomdp ファイルと全状態・全行動で一致を確認済み。gym-pomdps は端のマスで次の行動をとったときに報酬を払うが、ここでは端に入った時点で払う(割引 1 step 分の違いだけ)。
新タスクの設定は core4_mh.json と同じ(PPO、baseline、k16f)でタスクだけ差し替え。4 手法も走らせる(campaign core5-<tag>-v1、20 shards)。

## 参照値(脚本方策、報酬は入力に使わない、2000 episode)

| task | oracle(側を知る) | 合図を見に行ってから向かう | 決め打ち | 一様ランダム |
|---|---|---|---|---|
| carflag | 0.635 | 0.394 | −0.357 | −1.483 |
| heavenhell3 | 1.000(6 step) | 1.000(14 step) | 0.007 | −0.029 |
| heavenhell4 | 1.000(8 step) | 1.000(18 step) | 0.007 | −0.004 |

## campaign 一覧

| task | config | campaign | shards | updates | 見込み時間/腕 | elapse |
|---|---|---|---|---|---|---|
| mh | config/core5_mh_a02.json | core5-mh-a02-v1 | 5 | 200 | ≈107 h | 120:00:00 |
| mh | config/core5_mh_alearn.json | core5-mh-alearn-v1 | 5 | 200 | ≈107 h | 120:00:00 |
| ld2d | config/core5_ld2d_a02.json | core5-ld2d-a02-v1 | 5 | 200 | ≈93 h | 120:00:00 |
| ld2d | config/core5_ld2d_alearn.json | core5-ld2d-alearn-v1 | 5 | 200 | ≈93 h | 120:00:00 |
| pendbal | config/core5_pendbal_a02.json | core5-pendbal-a02-v1 | 5 | 200 | ≈350 h、3 segment | 168:00:00 |
| pendbal | config/core5_pendbal_alearn.json | core5-pendbal-alearn-v1 | 5 | 200 | ≈350 h、3 segment | 168:00:00 |
| mirror | config/core5_mirror_a02.json | core5-mirror-a02-v1 | 5 | 200 | ≈113 h | 168:00:00 |
| mirror | config/core5_mirror_alearn.json | core5-mirror-alearn-v1 | 5 | 200 | ≈113 h | 168:00:00 |
| mirror2 | config/core5_mirror2_a02.json | core5-mirror2-a02-v1 | 5 | 200 | ≈113 h | 168:00:00 |
| mirror2 | config/core5_mirror2_alearn.json | core5-mirror2-alearn-v1 | 5 | 200 | ≈113 h | 168:00:00 |
| tps | config/core5_tps_a02.json | core5-tps-a02-v1 | 5 | 100 | ≈52 h | 72:00:00 |
| tps | config/core5_tps_alearn.json | core5-tps-alearn-v1 | 5 | 100 | ≈52 h | 72:00:00 |
| carflag | config/core5_carflag.json | core5-carflag-v1 | 20 | 200 | tr ≈107 h、baseline 数時間 | 120:00:00 / 24:00:00 |
| carflag | config/core5_carflag_a02.json | core5-carflag-a02-v1 | 5 | 200 | ≈107 h | 120:00:00 |
| carflag | config/core5_carflag_alearn.json | core5-carflag-alearn-v1 | 5 | 200 | ≈107 h | 120:00:00 |
| heavenhell3 | config/core5_heavenhell3.json | core5-heavenhell3-v1 | 20 | 200 | tr ≈107 h、baseline 数時間 | 120:00:00 / 24:00:00 |
| heavenhell3 | config/core5_heavenhell3_a02.json | core5-heavenhell3-a02-v1 | 5 | 200 | ≈107 h | 120:00:00 |
| heavenhell3 | config/core5_heavenhell3_alearn.json | core5-heavenhell3-alearn-v1 | 5 | 200 | ≈107 h | 120:00:00 |
| heavenhell4 | config/core5_heavenhell4.json | core5-heavenhell4-v1 | 20 | 200 | tr ≈107 h、baseline 数時間 | 120:00:00 / 24:00:00 |
| heavenhell4 | config/core5_heavenhell4_a02.json | core5-heavenhell4-a02-v1 | 5 | 200 | ≈107 h | 120:00:00 |
| heavenhell4 | config/core5_heavenhell4_alearn.json | core5-heavenhell4-alearn-v1 | 5 | 200 | ≈107 h | 120:00:00 |

新タスクの 1 update あたりの時間は Hike の実測(≈32 min)からの見積もり。tps は core4 と比べられるように core4 と同じ 100 updates。

`core4_ld2d_pf_s13.sh` は core4 の Light-Dark 2D DPFRL seed 13 の再実行(旧ツリーで update 3 に非有限勾配ノルムで停止; この木は該当 minibatch をスキップする)。

```bash
pjsub jobs/core5/debug_core5.sh   # 合格後にゲートが投入(gate_core5.sh <JID> <scope>...; scope = all | nopendbal | info | ld2dpf | <task>)
```
