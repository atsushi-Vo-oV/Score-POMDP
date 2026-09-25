# pendulum_v2 — 第 3 の課題: 速度隠蔽・ノイズつき振り子(連続状態)× 設計格子 v1〜v6 × 5 seed

`masked_pendulum`(既存実装): 状態 (θ, θ̇)、観測 (cos θ, sin θ) + N(0, 0.05²)、角速度は観測不可、トルク ±2 の連続行動、200 step、gym Pendulum と同じコスト(θ² + 0.1 θ̇² + 0.001 u²)。belief は角度の履歴から速度を推定する(平滑化ではなく微分)必要がある。プロトコルは LD2D 格子と同じ(symlog 目標、200 updates、seeds 10–14)。

| config | methods | shards |
|---|---|---|
| pendulum_v1.json | score_transformer, score_alpha, score_deepsets, gru, rnn, particle_filter | 30 |
| pendulum_v2.json | score_transformer, score_alpha, score_deepsets | 15 |
| pendulum_v3.json | score_transformer, score_alpha, score_deepsets | 15 |
| pendulum_v4.json | score_transformer, score_alpha, score_deepsets | 15 |
| pendulum_v5.json | score_transformer, score_alpha, score_deepsets | 15 |
| pendulum_v6.json | score_transformer, score_alpha, score_deepsets | 15 |

```bash
pjsub jobs/pendulum_v2/debug_pendulum.sh   # 合格後にゲートが全腕を投入
```

## FLOPs 一致版 — v2lf

v2 の連鎖を K=4 粒子・L=2 ステップ・エネルギー MLP [32,32] に縮めたもの(推論 0.68–0.72 MFLOP/step。gru 0.46、PF 原典準拠 0.87 の間。v2 本来は 14.7–16.2 MFLOP)。パラメータ数は transformer 72.5k、alpha 32.8k で既定とほぼ同じ。3 エンコーダ × seeds 10–14、campaign *-v2lf-v1。DEBUG: jobs/mountain_hike_v2/debug_lf.sh(3 課題共通)。
