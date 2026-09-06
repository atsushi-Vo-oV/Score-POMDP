# mountain_hike_q — 行動価値クリティック(Q ベース PPO)

`model.critic_kind=action`: 状態価値ヘッド V(b) を Q(b, a) ヘッドに置き換える。学習目標は実行した行動の on-policy λ-return(SARSA / Q(λ) 型)、方策の advantage は A = Q(b, a) − E_{a′∼π} Q(b, a′)(離散は厳密、連続は方策サンプル `q_value_samples` 本の平均)。`ppo.q_advantage_mix` で GAE advantage と混合(1 = Q のみ、0 = 従来 GAE、Q-Prop / IPG 型の補間)。bootstrap 値 V̂(b) = E_π Q も同じ期待値で置き換える。

| config | method | updates | 内容 |
|---|---|---|---|
| mountain_hike_v2q.json | score_transformer | 200 | v2 (no g) with the action-value critic Q(b,a); advantage = Q - E_pi Q |
| mountain_hike_v2q.json | score_alpha | 200 | v2 (no g) with the action-value critic Q(b,a); advantage = Q - E_pi Q |
| mountain_hike_v2q.json | score_deepsets | 200 | v2 (no g) with the action-value critic Q(b,a); advantage = Q - E_pi Q |
| mountain_hike_v2q_mix05.json | score_transformer | 200 | v2q with the Q-advantage mixed 50/50 with GAE (Q-Prop/IPG style) |
| mountain_hike_q_gru.json | gru | 200 | gru baseline with the action-value critic (fairness control) |

```bash
pjsub jobs/mountain_hike_q/debug_q.sh
# 合格後:
pjsub jobs/mountain_hike_q/v2q_score.sh
pjsub jobs/mountain_hike_q/v2q_alpha.sh
pjsub jobs/mountain_hike_q/v2q_ds.sh
pjsub jobs/mountain_hike_q/v2q_mix05_score.sh
pjsub jobs/mountain_hike_q/q_gru_gru.sh
```
