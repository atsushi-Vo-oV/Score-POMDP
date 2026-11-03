# try-svgd tagr — 距離観測の鬼ごっこ

try-svgd の鬼ごっこ(config/trysvgd_tag_*)の観測を、視野内の鬼の数から **各鬼までのノイズ付き距離**
(`observation_mode=ranges`、σ = 0.05 + 0.2·d、鬼の順序は固定、視野の制限なし)に変えた版。手元の確認(GRU の網に
理想の信念を足すと oracle と同程度、観測だけの GRU より約 1.5 高い。ログは math サーバーの ~/workspace/agent/sbpomdp/tagppo)で、
この観測なら信念が効くと分かった。PPO は 1000 update(16 env × 128 step、手元の確認では 1000 update までに頭打ち)。
それ以外(予測フェーズ、warm start なし、α 学習、潜在次元 6)は try-svgd の tag と同じ。

**FLOPs(2026-11-02 に数え直し)**: FlopCounterMode は eval + no_grad の融合 transformer カーネル(fast path)を数えないので、
以前の TF の値(2.03 MFLOP)は transformer 本体を欠いていた。fast path を切って数え直した値(energy TF-ULA 2.297 MFLOP)
に基準線の幅を合わせ直した(GRU [1182, 455] / 217、
RNN [721, 856] / 309、DPFRL [377, 670])。

| task | TF ULA / SVGD / SMI (MFLOP) | GRU (params) | RNN (params) | DPFRL (params) | TF params |
|---|---|---|---|---|---|
| tagr | 2.297 / 2.333 / 2.376 | 2.297 (1030k) | 2.297 (1030k) | 2.297 (358k) | 44k |

`ppo.sequence_microbatch_size=512`(4 環境まとめて信念を再計算。予測損失を系列ごとの平均にしたので 128 と同じ目的関数、約 2.2 倍速)。
light の腕(直接スコア、-joint あり)も同じ設定。v1(trysvgd-tagr-tf-ula-v1 の shard001 だけ 3 update 走った)は破棄し、v2 として
2026-11-02 から math サーバー(~/workspace/agent/sbpomdp)で実行。

| config | campaign | shards | updates (pretrain + PPO + predict) |
|---|---|---|---|
| trysvgd_tagr_base.json | trysvgd-tagr-base-v2 | 15 | 1293 (98 + 1000 + 195) |
| trysvgd_tagr_pf_v3.json(DPFRL、観測適合度 mlp、pf_hidden [366, 651]。jobs/trysvgd/README.md の 2026-11-04 の項) | trysvgd-tagr-pf-v3 | 5 | 1293 (98 + 1000 + 195) |
| trysvgd_tagr_tf-ula.json | trysvgd-tagr-tf-ula-v2 | 5 | 1293 (98 + 1000 + 195) |
| trysvgd_tagr_tf-svgd.json | trysvgd-tagr-tf-svgd-v2 | 5 | 1293 (98 + 1000 + 195) |
| trysvgd_tagr_tf-smi.json | trysvgd-tagr-tf-smi-v2 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdlight_tagr_tf-ula.json | trysvgdlight-tagr-tf-ula-v2 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdlight_tagr_tf-svgd.json | trysvgdlight-tagr-tf-svgd-v2 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdlight_tagr_tf-smi.json | trysvgdlight-tagr-tf-smi-v2 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdlight_tagr_tf-ula-joint.json | trysvgdlight-tagr-tf-ula-joint-v2 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdlight_tagr_tf-svgd-joint.json | trysvgdlight-tagr-tf-svgd-joint-v2 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdlight_tagr_tf-smi-joint.json | trysvgdlight-tagr-tf-smi-joint-v2 | 5 | 1293 (98 + 1000 + 195) |
