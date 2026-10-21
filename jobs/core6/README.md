# core6 — 多峰タスクで ULA のドリフト係数と雑音係数を分けた score transformer

core5 までの ULA は x ← x + α ∇log p̃ + √(2ατ) ξ で、雑音がドリフト係数 α に縛られていた。core4 / core5 の診断では
粒子の 1 時刻の移動量が雑音 √(2ατ) だけで説明でき(ドリフトは経路長の 3–6%、学習前と同じ)、t=0 の観測アンカーが
作る粒子集合に酔歩を足した記憶になっていた。core6 は x ← x + α_l ∇log p̃ + β_l ξ として α と β を独立に設定する。
K=16、L=12、エネルギー MLP [8]、d_model 32、ff 64、belief 設計 v2、PPO 設定、seeds 10–14 は core4 と同一。
対象は多峰 3 タスク(mirror / mirror2 / tps)のみ。gru / rnn / DPFRL / α 0.02 の列は core4mm の結果を使う。

| 腕 | α | β | 備考 |
|---|---|---|---|
| core4 tr(既存) | 0.02 | √(2·0.02) = 0.2 | 雑音は α に連動 |
| core5 a02(既存) | 0.2 | √(2·0.2) = 0.63 | 同上 |
| **b02** | 0.2 固定 | 0.2 固定 | 雑音は core4 と同じ、ドリフトだけ 10 倍(予算 α·L = 2.4) |
| **blearn** | 12 step 別に学習、初期 0.2 | 12 step 別に学習、初期 0.2 | sigmoid 境界 α ∈ [1e-4, 0.5]、β ∈ [1e-3, 2]、学習率 0.025 |
| **b02wd** | 0.2 固定 | 0.2 固定 | b02 + AdamW weight decay 0.01(ULA スケジュールには掛けない) |

学習中の β は `.out` の `langevin noise_scale update=N mean= min= max=` 行で追える(α は `langevin step_size`)。

## campaign 一覧

| task | config | campaign | shards | updates | 見込み時間/腕 | elapse |
|---|---|---|---|---|---|---|
| mirror | config/core6_mirror_b02.json | core6-mirror-b02-v1 | 5 | 200 | ≈117 h | 168:00:00 |
| mirror | config/core6_mirror_blearn.json | core6-mirror-blearn-v1 | 5 | 200 | ≈117 h | 168:00:00 |
| mirror | config/core6_mirror_b02wd.json | core6-mirror-b02wd-v1 | 5 | 200 | ≈117 h | 168:00:00 |
| mirror2 | config/core6_mirror2_b02.json | core6-mirror2-b02-v1 | 5 | 200 | ≈120 h | 168:00:00 |
| mirror2 | config/core6_mirror2_blearn.json | core6-mirror2-blearn-v1 | 5 | 200 | ≈120 h | 168:00:00 |
| mirror2 | config/core6_mirror2_b02wd.json | core6-mirror2-b02wd-v1 | 5 | 200 | ≈120 h | 168:00:00 |
| tps | config/core6_tps_b02.json | core6-tps-b02-v1 | 5 | 100 | ≈53 h | 72:00:00 |
| tps | config/core6_tps_blearn.json | core6-tps-blearn-v1 | 5 | 100 | ≈53 h | 72:00:00 |
| tps | config/core6_tps_b02wd.json | core6-tps-b02wd-v1 | 5 | 100 | ≈53 h | 72:00:00 |

## 投入

```
cd ~/Belief-Score-Based-Model-core6-20261022
pjsub jobs/core6/debug_core6.sh
( setsid ~/gate_core6.sh <DEBUG の JID> all > ~/gate_core6.log 2>&1 < /dev/null & )
```
