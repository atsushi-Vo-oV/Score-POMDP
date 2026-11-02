# try-svgd-alpha — α-vector 方策(score_alpha)の復活

9 月まで使っていた α-vector 型の方策(`score_alpha`: alpha_pool encoder = 粒子位置だけを粒子ごとの MLP に通して平均、
alpha_lse head = 行動ごとに 16 本の線形関数の温度付き logsumexp)を、try-svgd の設定(K=16、L=12、warm start なし、
α 学習、予測フェーズ、diffusion 方策の PPO)で走らせる。スコアは encoder に渡さない(`alpha_use_scores=false`)。
サンプラーは ULA と SVGD、スコアの出し方はエネルギー型(energy)と直接出力(direct)の両方。SMI は解釈を整理するまで保留。
粒子ごとの MLP の幅(`comparison.alpha_feedforward_dim`)は、同じサンプラー・同じスコアの出し方の transformer と
推論 1 step の FLOPs が一致するように決めた(信念の更新が支配的なので幅はどれも 26〜30)。

| task | score | sampler | TF MFLOP (params) | alpha ff | alpha MFLOP (params) |
|---|---|---|---|---|---|
| mirror2 | energy | ULA | 1.119 (41k) | 26 | 1.12 (21k) |
| mirror2 | energy | SVGD | 1.132 (41k) | 26 | 1.132 (21k) |
| mirror2 | direct | ULA | 0.827 (41k) | 26 | 0.827 (21k) |
| mirror2 | direct | SVGD | 0.839 (41k) | 26 | 0.839 (21k) |
| tps | energy | ULA | 1.106 (39k) | 26 | 1.106 (20k) |
| tps | energy | SVGD | 1.118 (39k) | 26 | 1.119 (20k) |
| tps | direct | ULA | 0.82 (40k) | 26 | 0.82 (20k) |
| tps | direct | SVGD | 0.832 (40k) | 26 | 0.832 (20k) |
| tagr | energy | ULA | 2.034 (44k) | 30 | 2.035 (24k) |
| tagr | energy | SVGD | 2.071 (44k) | 30 | 2.072 (24k) |
| tagr | direct | ULA | 1.515 (44k) | 30 | 1.516 (24k) |
| tagr | direct | SVGD | 1.552 (44k) | 30 | 1.553 (24k) |

| config | campaign | shards | updates (pretrain + PPO + predict) |
|---|---|---|---|
| trysvgdalpha_mirror2_energy-ula.json | trysvgdalpha-mirror2-energy-ula-v1 | 5 | 333 (98 + 200 + 35) |
| trysvgdalpha_mirror2_energy-svgd.json | trysvgdalpha-mirror2-energy-svgd-v1 | 5 | 333 (98 + 200 + 35) |
| trysvgdalpha_mirror2_direct-ula.json | trysvgdalpha-mirror2-direct-ula-v1 | 5 | 333 (98 + 200 + 35) |
| trysvgdalpha_mirror2_direct-svgd.json | trysvgdalpha-mirror2-direct-svgd-v1 | 5 | 333 (98 + 200 + 35) |
| trysvgdalpha_tps_energy-ula.json | trysvgdalpha-tps-energy-ula-v1 | 5 | 213 (98 + 100 + 15) |
| trysvgdalpha_tps_energy-svgd.json | trysvgdalpha-tps-energy-svgd-v1 | 5 | 213 (98 + 100 + 15) |
| trysvgdalpha_tps_direct-ula.json | trysvgdalpha-tps-direct-ula-v1 | 5 | 213 (98 + 100 + 15) |
| trysvgdalpha_tps_direct-svgd.json | trysvgdalpha-tps-direct-svgd-v1 | 5 | 213 (98 + 100 + 15) |
| trysvgdalpha_tagr_energy-ula.json | trysvgdalpha-tagr-energy-ula-v1 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdalpha_tagr_energy-svgd.json | trysvgdalpha-tagr-energy-svgd-v1 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdalpha_tagr_direct-ula.json | trysvgdalpha-tagr-direct-ula-v1 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdalpha_tagr_direct-svgd.json | trysvgdalpha-tagr-direct-svgd-v1 | 5 | 1293 (98 + 1000 + 195) |
