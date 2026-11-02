# try-svgd-alpha — alpha_pool encoder + alpha_lse critic + diffusion actor(score_alpha)

`score_alpha` を try-svgd の設定(K=16、L=12、warm start なし、α 学習、予測フェーズ、diffusion 方策の PPO)で走らせる。
**連続行動では α-vector 方策ではない**(2026-11-02 の Codex レビューで確認、results/reports/score_alpha_20261102.pdf):
alpha_pool encoder = 粒子位置だけを粒子ごとの MLP に通して平均(信念に線形な特徴)、alpha_lse = 16 本の線形関数の温度 1 の
logsumexp は **critic(価値)だけ**に使われ、actor は通常の diffusion 方策(MLP denoiser)。行動ごとの α-vector になるのは離散行動のとき。
スコアは encoder に渡さない(`alpha_use_scores=false`)。サンプラーは ULA と SVGD、スコアの出し方はエネルギー型(energy)と
直接出力(direct)。SMI は解釈を整理するまで保留。

粒子ごとの MLP の幅(`comparison.alpha_feedforward_dim`)は、同じサンプラー・同じスコアの出し方の transformer と推論 1 step の
FLOPs が一致するように決めた。**2026-11-02 に数え直し**: FlopCounterMode は融合 transformer カーネル(fast path)を数えないので、
以前の幅 26〜30 は transformer 本体を欠いた値に合わせていた。fast path を切った値に合わせると幅は 245〜267 になる。

| task | score | sampler | TF MFLOP (params) | alpha ff | alpha MFLOP (params) |
|---|---|---|---|---|---|
| mirror2 | energy | ULA | 1.382 (41k) | 267 | 1.382 (30k) |
| mirror2 | energy | SVGD | 1.394 (41k) | 267 | 1.394 (30k) |
| mirror2 | direct | ULA | 1.089 (41k) | 267 | 1.089 (30k) |
| mirror2 | direct | SVGD | 1.101 (41k) | 267 | 1.102 (30k) |
| tps | energy | ULA | 1.368 (39k) | 267 | 1.369 (29k) |
| tps | energy | SVGD | 1.38 (39k) | 266 | 1.38 (28k) |
| tps | direct | ULA | 1.082 (40k) | 267 | 1.082 (29k) |
| tps | direct | SVGD | 1.094 (40k) | 266 | 1.094 (29k) |
| tagr | energy | ULA | 2.297 (44k) | 246 | 2.297 (33k) |
| tagr | energy | SVGD | 2.333 (44k) | 245 | 2.333 (33k) |
| tagr | direct | ULA | 1.777 (44k) | 245 | 1.777 (33k) |
| tagr | direct | SVGD | 1.814 (44k) | 245 | 1.814 (33k) |

`ppo.sequence_microbatch_size=512`。v1 は走らせずに破棄し、v2 として math サーバーで実行。

| config | campaign | shards | updates (pretrain + PPO + predict) |
|---|---|---|---|
| trysvgdalpha_mirror2_energy-ula.json | trysvgdalpha-mirror2-energy-ula-v2 | 5 | 333 (98 + 200 + 35) |
| trysvgdalpha_mirror2_energy-svgd.json | trysvgdalpha-mirror2-energy-svgd-v2 | 5 | 333 (98 + 200 + 35) |
| trysvgdalpha_mirror2_direct-ula.json | trysvgdalpha-mirror2-direct-ula-v2 | 5 | 333 (98 + 200 + 35) |
| trysvgdalpha_mirror2_direct-svgd.json | trysvgdalpha-mirror2-direct-svgd-v2 | 5 | 333 (98 + 200 + 35) |
| trysvgdalpha_tps_energy-ula.json | trysvgdalpha-tps-energy-ula-v2 | 5 | 213 (98 + 100 + 15) |
| trysvgdalpha_tps_energy-svgd.json | trysvgdalpha-tps-energy-svgd-v2 | 5 | 213 (98 + 100 + 15) |
| trysvgdalpha_tps_direct-ula.json | trysvgdalpha-tps-direct-ula-v2 | 5 | 213 (98 + 100 + 15) |
| trysvgdalpha_tps_direct-svgd.json | trysvgdalpha-tps-direct-svgd-v2 | 5 | 213 (98 + 100 + 15) |
| trysvgdalpha_tagr_energy-ula.json | trysvgdalpha-tagr-energy-ula-v2 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdalpha_tagr_energy-svgd.json | trysvgdalpha-tagr-energy-svgd-v2 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdalpha_tagr_direct-ula.json | trysvgdalpha-tagr-direct-ula-v2 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdalpha_tagr_direct-svgd.json | trysvgdalpha-tagr-direct-svgd-v2 | 5 | 1293 (98 + 1000 + 195) |
