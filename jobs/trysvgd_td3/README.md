# try-svgd-td3 — try-svgd のモデルを PPO ではなく TD3 で学習

信念モデル(GRU、RNN、DPFRL、TF-ULA、TF-SVGD、TF-SMI)の設定、FLOPs の合わせ方(TD3 の行動経路でも同じ幅)、
予測フェーズ(20 万 step のランダム行動 → TD3 25 update ごとに予測 5 update)は try-svgd と同じ。
方策は決定的な actor μ(c)(tanh)と twin critic Q₁, Q₂(c, a)。critic の TD 損失が信念を学習し、actor は信念を
切り離した condition を読む。リプレイは各 step の雑音 seed を保存し、現在のパラメータで信念をエピソード先頭から
厳密に再計算する(burn-in は勾配なし、学習窓 32 step は勾配あり)。1 update = 2048 step 収集 + 勾配 16 step
(16 系列 × 32 step)。探索雑音 0.1、ターゲット雑音 0.2(clip 0.5)、τ 0.005、actor 更新は 2 回に 1 回、
学習率は actor / critic とも 3e-4。

| task | arm | config | campaign | shards | updates |
|---|---|---|---|---|---|
| mirror2 | base | config/trysvgdtd3_mirror2_base.json | trysvgdtd3-mirror2-base-v1 | 15 | 333 |
| mirror2 | tf-ula | config/trysvgdtd3_mirror2_tf-ula.json | trysvgdtd3-mirror2-tf-ula-v1 | 5 | 333 |
| mirror2 | tf-svgd | config/trysvgdtd3_mirror2_tf-svgd.json | trysvgdtd3-mirror2-tf-svgd-v1 | 5 | 333 |
| mirror2 | tf-smi | config/trysvgdtd3_mirror2_tf-smi.json | trysvgdtd3-mirror2-tf-smi-v1 | 5 | 333 |
| tps | base | config/trysvgdtd3_tps_base.json | trysvgdtd3-tps-base-v1 | 15 | 213 |
| tps | tf-ula | config/trysvgdtd3_tps_tf-ula.json | trysvgdtd3-tps-tf-ula-v1 | 5 | 213 |
| tps | tf-svgd | config/trysvgdtd3_tps_tf-svgd.json | trysvgdtd3-tps-tf-svgd-v1 | 5 | 213 |
| tps | tf-smi | config/trysvgdtd3_tps_tf-smi.json | trysvgdtd3-tps-tf-smi-v1 | 5 | 213 |
