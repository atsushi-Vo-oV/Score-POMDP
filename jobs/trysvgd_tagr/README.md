# try-svgd tagr — 距離観測の鬼ごっこ

try-svgd の鬼ごっこ(config/trysvgd_tag_*)の観測を、視野内の鬼の数から **各鬼までのノイズ付き距離**
(`observation_mode=ranges`、σ = 0.05 + 0.2·d、鬼の順序は固定、視野の制限なし)に変えた版。手元の確認(GRU の網に
理想の信念を足すと oracle と同程度、観測だけの GRU より約 1.5 高い)で、この観測なら信念が効くと分かった。
PPO は 1000 update(16 env × 128 step、手元の確認では 1000 update までに頭打ち)。それ以外(FLOPs 合わせ、予測フェーズ、
warm start なし、α 学習、潜在次元 6)は try-svgd の tag と同じ。基準線の幅は距離観測の TF-ULA(2.03 MFLOP)に合わせ直した
(GRU [1101, 424] / 202、RNN [671, 797] / 288、DPFRL [336, 597])。light の腕(直接スコア、-joint あり)も同じ設定。
2026-11-02 から math サーバー(~/workspace/agent/sbpomdp)で実行。

| config | campaign | shards | updates (pretrain + PPO + predict) |
|---|---|---|---|
| trysvgd_tagr_base.json | trysvgd-tagr-base-v1 | 15 | 1293 (98 + 1000 + 195) |
| trysvgd_tagr_tf-ula.json | trysvgd-tagr-tf-ula-v1 | 5 | 1293 (98 + 1000 + 195) |
| trysvgd_tagr_tf-svgd.json | trysvgd-tagr-tf-svgd-v1 | 5 | 1293 (98 + 1000 + 195) |
| trysvgd_tagr_tf-smi.json | trysvgd-tagr-tf-smi-v1 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdlight_tagr_tf-ula.json | trysvgdlight-tagr-tf-ula-v1 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdlight_tagr_tf-svgd.json | trysvgdlight-tagr-tf-svgd-v1 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdlight_tagr_tf-smi.json | trysvgdlight-tagr-tf-smi-v1 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdlight_tagr_tf-ula-joint.json | trysvgdlight-tagr-tf-ula-joint-v1 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdlight_tagr_tf-svgd-joint.json | trysvgdlight-tagr-tf-svgd-joint-v1 | 5 | 1293 (98 + 1000 + 195) |
| trysvgdlight_tagr_tf-smi-joint.json | trysvgdlight-tagr-tf-smi-joint-v1 | 5 | 1293 (98 + 1000 + 195) |
