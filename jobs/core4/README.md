# core4 — 比較の中核 4 手法を同じ推論予算で並べる実験グループ

**core4** = gru、rnn、DPFRL 粒子フィルタ、score transformer の 4 手法。score 側は 16 粒子を保ったまま FLOPs を baseline に合わせた **k16f** 設定: K=16、Langevin 1 step、エネルギー MLP [12,12]、d_model 32、ff 64 → 推論 0.81 MFLOP/step(gru 0.46、rnn 0.46、PF 0.87; 本来サイズ v2 は 14.7、旧 lf の K=4 は 0.68)。パラメータ数は 38k(baseline 76k)。belief 設計は v2(warm start + 観測アンカー + 粒子別補助損失)。cold 連鎖は 1 step では意味を持たないので入れない。

| task | config | campaign | shards |
|---|---|---|---|
| mh | core4_mh.json | core4-mh-v1 | 20 |
| ld2d | core4_ld2d.json | core4-ld2d-v1 | 20 |
| pendbal | core4_pendbal.json | core4-pendbal-v1 | 20 |

seeds 10–14、200 updates。pendbal は固定 PPO 設定(γ0.9、lr 1e-3、10 epochs、minibatch 128、entropy 0、価値目標 symlog なし、GAE 0.9)。

```bash
pjsub jobs/core4/debug_core4.sh   # 合格後にゲートが投入(gate_core4.sh <JID> mh|ld2d|pendbal|all)
```
