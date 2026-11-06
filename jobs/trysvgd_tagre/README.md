# trysvgd_tagre: 鬼ごっこ、楕円距離観測（向き依存）、学習 2 倍、環境 64、粒子 64

`observation_mode = elliptic_ranges`, `range_forward_scale = 1.0`, `range_side_scale = 0.6`: 距離を体座標で測り、横方向の軸を 0.6 で割る。観測と整合する集合は向きに沿った楕円（半軸 1.0 r, 0.6 r）になり、体を回すと楕円が回るので、見回すことで方位が決まる（能動的センシング）。ノイズは tagr と同じ σ = 0.05 + 0.2 r。PPO 2000 update（tagr の 2 倍）、環境数 64（tagr の 4 倍、minibatch 2048、信念リプレイの microbatch 2048 = 16 環境を 1 回で再生。GPU の 1 カーネルの仕事を太らせるため）、粒子 64（tagr の 4 倍。light / energy と PF v3 に適用）。

| ジョブ | config | campaign | shard |
|---|---|---|---|
| trysvgdlight_tagre_svgd_s10-14 | trysvgdlight_tagre_tf-svgd.json（素 light、ノイズ開始） | trysvgdlight-tagre-tf-svgd-env64-k64-v1 | 1-5 |
| trysvgdlight_tagre_svgd-funcinit_s10-14 | trysvgdlight_tagre_tf-svgd-func-init.json | trysvgdlight-tagre-tf-svgd-func-init-env64-k64-v1 | 1-5 |
| trysvgdlight_tagre_svgd-paraminit_s10-14 | trysvgdlight_tagre_tf-svgd-param-init.json | trysvgdlight-tagre-tf-svgd-param-init-env64-k64-v1 | 1-5 |
| trysvgd_tagre_svgd_s10-14 | trysvgd_tagre_tf-svgd.json（energy TF-SVGD） | trysvgd-tagre-tf-svgd-env64-k64-v1 | 1-5 |
| trysvgd_tagre_gru_s10-14 | trysvgd_tagre_gru.json (methods = gru) | trysvgd-tagre-gru-env64-k64-v1 | 1-5 |
| trysvgd_tagre_pf_v3_s10-14 | trysvgd_tagre_pf_v3.json | trysvgd-tagre-pf-v3-env64-k64-v1 | 1-5 |
