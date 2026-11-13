# try-svgd-light2 — light TF-SVGD の既定 v2（2026-11-14）

ユーザー指定（2026-11-14）の新しい既定。try-svgd-light の light TF-SVGD（スコアを直接出す transformer + SVGD、K = 16、L = 12、
網の幅 [8]、α 学習、予測フェーズ、diffusion 方策の PPO）から、次の 3 点だけを変える。

1. 連鎖の開始は t = 0 も t ≥ 1 も一様分布 U(−√3, √3)（分散は N(0, I) と同じ）: `langevin_proposal: "uniform"`、`langevin_initial_noise: "uniform"`
2. 観測アンカーなし（t = 0 と t ≥ 1 の粒子の尺度をそろえる）: `langevin_observation_anchor: false`
3. スコアの有界化なし: `direct_score_bound: null`。あわせて t = 0 の −x も外す（`initial_score_prior: 0.0`。2026-11-10 の指示
   「非有界なやつはついでに初期値の −x も消して」に合わせた）。

以後の新しい比較は、指定がない限りこの設定を素として作る（campaign の接頭辞 `trysvgdlight2-`）。比較相手の v1 は
`trysvgdlight-<task>-tf-svgd-v1`（tagr は `-ctl-v1`）、1 点だけ変えた既存の腕は uniform-init-u0 / noanchor / nobound。

| task | arm | config | campaign | shards | updates |
|---|---|---|---|---|---|
| mirror2 | tf-svgd | config/trysvgdlight2_mirror2_tf-svgd.json | trysvgdlight2-mirror2-tf-svgd-v1 | 5 | 333 |
| tps | tf-svgd | config/trysvgdlight2_tps_tf-svgd.json | trysvgdlight2-tps-tf-svgd-v1 | 5 | 213 |
| tagr | tf-svgd | config/trysvgdlight2_tagr_tf-svgd.json | trysvgdlight2-tagr-tf-svgd-v1 | 5 | 1293 |
| carflag | tf-svgd | config/trysvgdlight2_carflag_tf-svgd.json | trysvgdlight2-carflag-tf-svgd-v1 | 5 | 333 |

ジョブ: `trysvgdlight2_<task>_svgd_s{10..14}.sh`（shard = seed − 9）。
