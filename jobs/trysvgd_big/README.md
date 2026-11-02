# try-svgd-big — 粒子数を増やした light 腕(表現次元合わせ)

粒子数 K × 粒子の次元 D を、同じタスクの FLOPs 合わせ済み GRU の隠れ状態の次元に合わせる(mirror2: 80 × 2 = 160、
tagr: 36 × 6 = 216)。**計算量は合わせない**(信念の更新は K に、遷移エネルギーの粒子対と transformer の注意は K² に比例)。
腕は light(直接スコア)の ULA / SVGD だけ。基準線は trysvgd-{mirror2,tagr}-base-v2、K=16 の対照は trysvgdlight-*。
それ以外の設定(L=12、α 学習、予測フェーズ、sequence_microbatch_size 512)は light と同じ。math サーバーで実行。

| task | sampler | K x D | config | campaign | shards | updates (pretrain + PPO + predict) |
|---|---|---|---|---|---|---|
| mirror2 | ULA | 80 x 2 | trysvgdbig_mirror2_tf-ula.json | trysvgdbig-mirror2-ula-v1 | 5 | 333 (98 + 200 + 35) |
| mirror2 | SVGD | 80 x 2 | trysvgdbig_mirror2_tf-svgd.json | trysvgdbig-mirror2-svgd-v1 | 5 | 333 (98 + 200 + 35) |
| tagr | ULA | 36 x 6 | trysvgdbig_tagr_tf-ula.json | trysvgdbig-tagr-ula-v1 | 5 | 1293 (98 + 1000 + 195) |
| tagr | SVGD | 36 x 6 | trysvgdbig_tagr_tf-svgd.json | trysvgdbig-tagr-svgd-v1 | 5 | 1293 (98 + 1000 + 195) |
