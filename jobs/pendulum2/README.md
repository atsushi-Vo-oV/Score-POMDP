# pendulum2 — 振り子(振り上げ、速度隠蔽)を「完全観測で解ける PPO 設定」で取り直す

元の pendulum 格子(ppo は Hike/LD2D と同じ γ0.99・lr 2.5e-4・4 epochs・minibatch 512)は、真の状態を入力にしても 388 updates まで −1150〜−1380(無トルク −1277)で平坦だった。ローカル検証(oracle_state、seed 10)で **γ 0.9、lr 1e-3、10 epochs、minibatch 128、entropy 0、価値目標 symlog なし、GAE λ 0.9** にすると @75 −198、@200 −183(真状態の制御器 −160)まで学ぶ。効いたのは 1 update あたりの勾配ステップ数(16 → 160)と γ。この設定で格子を取り直す。

10 epochs は score 腕の PPO 相を 2.5 倍にする: lf ≈ 16 min/update(200 updates ≈ 55 h、elapse 72h)、本来サイズ warm 系 ≈ 30〜40 min/update(100〜130 h、elapse 96h + 再投入 1 回)。cold 本来サイズ(v1、v3、v6)は 150 min/update になるので lf のみ。

| config | methods | shards |
|---|---|---|
| pendulum2_v1lf.json | score_transformer, score_alpha, score_deepsets, gru, rnn, particle_filter | 30 |
| pendulum2_v2lf.json | score_transformer, score_alpha, score_deepsets | 15 |
| pendulum2_v2.json | score_transformer, score_alpha, score_deepsets | 15 |
| pendulum2_v3lf.json | score_transformer, score_alpha, score_deepsets | 15 |
| pendulum2_v4lf.json | score_transformer, score_alpha, score_deepsets | 15 |
| pendulum2_v4.json | score_transformer, score_alpha, score_deepsets | 15 |
| pendulum2_v5lf.json | score_transformer, score_alpha, score_deepsets | 15 |
| pendulum2_v5.json | score_transformer, score_alpha, score_deepsets | 15 |
| pendulum2_v6lf.json | score_transformer, score_alpha, score_deepsets | 15 |

```bash
pjsub jobs/pendulum2/debug_pendulum2.sh   # 合格後にゲートが投入(gate_pendulum2.sh <JID> lf|full|all)
```
