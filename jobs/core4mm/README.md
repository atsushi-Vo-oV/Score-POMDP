# core4mm — core4(gru / rnn / DPFRL 粒子フィルタ / score transformer k16f)を多峰タスクで走らせる

**core4** の設定(K=16、Langevin 12 step、エネルギー MLP [8]、d_model 32、ff 64 → 推論 1.11 MFLOP/step、belief 設計 v2)をそのまま、真 belief が多峰になる 3 タスクに適用する。seeds 10–14、200 updates。

| task | 内容 | config | campaign | shards |
|---|---|---|---|---|
| mirror | 鏡像世界 Hike、lf 格子と同じ(尾根 1、鏡像コーナー開始、標識幅 w=2、箱外 −1.5×4=−6/step) | config/core4_mirror.json | core4-mirror-v1 | 20 |
| mirror2 | 再修正版: 地形は同じ(尾根 1 本のまま。3 尾根にすると m を知る価値が 2.3→0.4/step に縮み多峰が行動に効かなくなる)、箱外罰則 outside_box_cost −5(= −20/step)、標識幅 w=4(可読半径 ≈2.7→5.4、箱面積の ≈23%) | config/core4_mirror2.json | core4-mirror2-v1 | 20 |
| tps | テレポート(p=0.03)+ \|s\| 観測 Hike(4 象限の多峰)。旧実験は seed 10 × 100 updates のみ | config/core4_tps.json | core4-tps-v1 | 20 |

score 腕の壁時間は horizon に比例(core4 実測: horizon 75 で ≈32 min/update)→ mirror/mirror2(horizon 100)は ≈143h で elapse 168h、tps(75)は ≈107h で elapse 120h。baseline は 24h。

```bash
pjsub jobs/core4mm/debug_core4mm.sh   # 合格後にゲートが投入(gate_core4mm.sh <JID> mirror|mirror2|tps|all)
```
