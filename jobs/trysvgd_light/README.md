# try-svgd-light — スコアを NN で直接出力する transformer(ULA / SVGD / SMI)

try-svgd の transformer 3 腕と同じ設定(K=16、L=12、ネットワーク幅 [8]、warm start なし、α 学習、予測フェーズ、
diffusion 方策の PPO)で、エネルギーを自動微分する代わりにネットワークがスコアを直接出力する
(`model.score_parameterization: "direct"`)。t=0 は g(o, x) − x、t≥1 は f(o, x, a) + Σ_j softmax_j(k_j) v_j
((k_j, v_j) は前の粒子 x_{t-1,j} との組を読む遷移ネットワークの出力)。2 階微分が不要なぶん軽い。
基準線(GRU / RNN / DPFRL)とエネルギー版の transformer は try-svgd の結果と比べる。

| task | arm | config | campaign | shards | updates |
|---|---|---|---|---|---|
| mirror2 | tf-ula | config/trysvgdlight_mirror2_tf-ula.json | trysvgdlight-mirror2-tf-ula-v1 | 5 | 333 |
| mirror2 | tf-svgd | config/trysvgdlight_mirror2_tf-svgd.json | trysvgdlight-mirror2-tf-svgd-v1 | 5 | 333 |
| mirror2 | tf-smi | config/trysvgdlight_mirror2_tf-smi.json | trysvgdlight-mirror2-tf-smi-v1 | 5 | 333 |
| tps | tf-ula | config/trysvgdlight_tps_tf-ula.json | trysvgdlight-tps-tf-ula-v1 | 5 | 213 |
| tps | tf-svgd | config/trysvgdlight_tps_tf-svgd.json | trysvgdlight-tps-tf-svgd-v1 | 5 | 213 |
| tps | tf-smi | config/trysvgdlight_tps_tf-smi.json | trysvgdlight-tps-tf-smi-v1 | 5 | 213 |
