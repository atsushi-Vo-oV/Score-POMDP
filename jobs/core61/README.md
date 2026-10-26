# core6.1 — core6 blearn の ULA スケジュール学習率を 1/10 にした腕

core6 blearn では、step 別の α_l / β_l(初期 0.2 / 0.2、sigmoid 境界)を学習率 0.025 で学習した。
最初の 2 update で α は 0.2 → 0.08–0.11、β は 0.2 → 0.29–0.43 に動き、エネルギーが育つ前に
「ドリフトを弱めて雑音を強める」側へ寄った。core6.1 はスケジュール学習率だけを 0.0025
(`ppo.langevin_schedule_lr_multiplier` 100 → 10)にする。α と β は同じパラメータ群なので両方とも遅くなる。
それ以外(K=16、L=12、エネルギー MLP [8]、d_model 32、ff 64、belief 設計 v2、PPO 設定、seeds 10–14)は core6 blearn と同一。

| task | config | campaign | shards | updates | elapse |
|---|---|---|---|---|---|
| mirror | config/core61_mirror_blslow.json | core61-mirror-blslow-v1 | 5 | 200 | 168:00:00 |
| mirror2 | config/core61_mirror2_blslow.json | core61-mirror2-blslow-v1 | 5 | 200 | 168:00:00 |
| tps | config/core61_tps_blslow.json | core61-tps-blslow-v1 | 5 | 100 | 72:00:00 |

## 投入

```
cd ~/Belief-Score-Based-Model-core61-20261026
pjsub jobs/core61/debug_core61.sh
( setsid ~/gate_core61.sh <DEBUG の JID> > ~/gate_core61.log 2>&1 < /dev/null & )
```
