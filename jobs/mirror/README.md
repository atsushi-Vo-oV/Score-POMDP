# mirror — 鏡像世界 Mountain Hike: 多峰 belief が持続し、行動に効く課題 × 設計格子 v1〜v6 × 5 seed

`mountain_hike` に `mirror_symmetry=true` を指定したもの。エピソードごとに未知の反転 m ∈ {±1}² が引かれ、エージェントの身体座標系は世界座標系を m で鏡映したもの: 観測は m⊙s + N(0, 3²)、行動 a は世界状態を m⊙a だけ動かす。4 つの鏡像仮説はどんな行動列でも観測上区別できないので、世界位置の belief は 4 峰(2 次元の鏡像 4 点、各峰の中では σ=3 の観測を時間平均して 2 座標とも推定する必要がある)のまま持続する。原点の「標識」(観測の追加 2 成分 z = m·exp(−|s|²/2w²) + N(0, 0.2²)、w=2)を読みに行ったときだけ m が分かる。報酬は DVRL 地形の尾根 1 だけ(世界 y=5 の水平尾根、傾斜で x=8 付近が最良、他は −4.5 前後)なので、m を知らないと y=±5・x=±8 のどれに向かうべきか決まらない。開始は 4 つの鏡像コーナー(±8.5, ±8.5)の一つ(m とは独立)、100 step。

参照値(脚本方策、scratchpad/mirror_refs.py): oracle(m と位置既知)、m 既知+Kalman、標識戦略+Kalman(原点へ → z を読む → m̂⊙(8,5) へ)、標識戦略+生観測、m=(+,+) と決め打ち、m に依らない折衷 (0,5)、静止。

| config | methods | shards |
|---|---|---|
| mirror_v1.json | score_transformer, score_alpha, score_deepsets, gru, rnn, particle_filter | 30 |
| mirror_v2.json | score_transformer, score_alpha, score_deepsets | 15 |
| mirror_v3.json | score_transformer, score_alpha, score_deepsets | 15 |
| mirror_v4.json | score_transformer, score_alpha, score_deepsets | 15 |
| mirror_v5.json | score_transformer, score_alpha, score_deepsets | 15 |
| mirror_v6.json | score_transformer, score_alpha, score_deepsets | 15 |

```bash
pjsub jobs/mirror/debug_mirror.sh   # 合格後にゲートが全腕を投入(score 48h、baseline 24h、MIG)
```

## 参照値(200 episode、平均 return、scratchpad/mirror_refs.py)

| 方策 | 知っているもの | return | m̂ 正答率 |
|---|---|---|---|
| oracle | m と世界位置 | −116 | — |
| m 既知 + Kalman | m、身体位置は σ=3 観測を Kalman で平均 | −151 | — |
| 標識戦略 + Kalman | 原点へ行き z を 2 step 読んで m̂ を決め、m̂⊙(8,5) へ | −184 | 0.99 |
| 標識戦略 + 生観測 | 同上、フィルタなし | −240 | 0.89 |
| 折衷 (0,5) + Kalman | m を使わない最良の固定目標(m_y=+1 のときだけ尾根) | −309 | — |
| m=(+,+) 決め打ち + Kalman | 4 分の 1 で正解 | −349 | — |
| 静止 | — | −491 | — |

読み: m を解く価値(標識戦略 − 折衷)≈ 125、2 座標の時間平均の価値(Kalman − 生観測)≈ 56、標識への迂回コスト(m 既知 − 標識戦略)≈ 33。真 belief は標識を読むまで 4 峰(各峰は 2 次元ガウス)で、読んだ瞬間に 1 峰へ崩れる。

## FLOPs 一致版 — v1lf〜v6lf

score 系の連鎖を K=4・L=2・エネルギー MLP [32,32] に縮めたもの(推論 0.68〜0.72 MFLOP/step、gru 0.46 と PF 0.87 の間)。baseline は元から同予算なのでそのまま。config/mirror_v{1..6}lf.json、campaign mirror-vNlf-v1、v1lf は 6 手法(30 shards)、v2lf〜v6lf は score 3 エンコーダ(15 shards)× seeds 10〜14 = 105 腕。DEBUG: jobs/mirror/debug_mirror_lf.sh(合格後にゲートが全腕を投入、score 36h / baseline 24h)。
