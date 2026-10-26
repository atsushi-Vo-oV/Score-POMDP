# try-svgd — GRU / RNN / DPFRL と score transformer (ULA / SVGD / SMI) を多峰 2 タスクと鬼ごっこで比較

- **タスク**: mirror2(鏡像 Mountain Hike)、tps(|s| 観測 + テレポート)、tag(円盤鬼ごっこ: 鬼 3 体は視野 180°・距離 1.0
  でうろつき、見つけたら追跡、見失ったら最後の地点を探索。プレイヤーは視野 60°・距離 0.6 の中の鬼の数だけを観測。
  1 step 生存ごとに +0.1)。
- **モデル**: GRU、RNN、DPFRL(粒子フィルタ)、TF-ULA、TF-SVGD、TF-SMI。transformer は K=16、L=12、エネルギー MLP [8]、
  d_model 32、ff 64、**warm start なし**、t=0 観測アンカーあり、α_l は 0.2 から sigmoid 境界で学習(スケジュール学習率 0.0025)。
  ULA は β=0.2 固定、SVGD / SMI は雑音なし。SMI は成分の log σ を encoder に渡す。tag の潜在次元は 6。
- **FLOPs**: transformer に統一(推論 1 step、FlopCounterMode)。基準線は core4 の幅を一様に拡大。

| task | TF ULA / SVGD / SMI (MFLOP) | GRU (params) | RNN (params) | DPFRL (params) | TF params |
|---|---|---|---|---|---|
| mirror2 | 1.119 / 1.132 / 1.146 | 1.122 (435k) | 1.12 (433k) | 1.12 (114k) | 36k |
| tps | 1.106 / 1.118 / 1.132 | 1.107 (428k) | 1.106 (427k) | 1.107 (111k) | 36k |
| disk_tag | 2.021 / 2.058 / 2.101 | 2.022 (886k) | 2.023 (885k) | 2.024 (288k) | 37k |

- **予測フェーズ**: 全モデル共通の次観測予測ヘッド(policy condition + 行動 → 4 成分混合ガウス)。最初に 20 万 step を
  一様ランダム行動で集めて予測だけを学習し、その後は PPO 25 update ごとに現在の方策で集めた 5 update ぶんを予測学習。
  PPO 中の補助損失はなし。ランダム行動のエピソードは学習曲線に入れない。

| task | arm | config | campaign | shards | updates (pretrain + PPO + predict) |
|---|---|---|---|---|---|
| mirror2 | base | config/trysvgd_mirror2_base.json | trysvgd-mirror2-base-v1 | 15 | 333 (98 + 200 + 35) |
| mirror2 | tf-ula | config/trysvgd_mirror2_tf-ula.json | trysvgd-mirror2-tf-ula-v1 | 5 | 333 (98 + 200 + 35) |
| mirror2 | tf-svgd | config/trysvgd_mirror2_tf-svgd.json | trysvgd-mirror2-tf-svgd-v1 | 5 | 333 (98 + 200 + 35) |
| mirror2 | tf-smi | config/trysvgd_mirror2_tf-smi.json | trysvgd-mirror2-tf-smi-v1 | 5 | 333 (98 + 200 + 35) |
| tps | base | config/trysvgd_tps_base.json | trysvgd-tps-base-v1 | 15 | 213 (98 + 100 + 15) |
| tps | tf-ula | config/trysvgd_tps_tf-ula.json | trysvgd-tps-tf-ula-v1 | 5 | 213 (98 + 100 + 15) |
| tps | tf-svgd | config/trysvgd_tps_tf-svgd.json | trysvgd-tps-tf-svgd-v1 | 5 | 213 (98 + 100 + 15) |
| tps | tf-smi | config/trysvgd_tps_tf-smi.json | trysvgd-tps-tf-smi-v1 | 5 | 213 (98 + 100 + 15) |
| tag | base | config/trysvgd_tag_base.json | trysvgd-tag-base-v1 | 15 | 573 (98 + 400 + 75) |
| tag | tf-ula | config/trysvgd_tag_tf-ula.json | trysvgd-tag-tf-ula-v1 | 5 | 573 (98 + 400 + 75) |
| tag | tf-svgd | config/trysvgd_tag_tf-svgd.json | trysvgd-tag-tf-svgd-v1 | 5 | 573 (98 + 400 + 75) |
| tag | tf-smi | config/trysvgd_tag_tf-smi.json | trysvgd-tag-tf-smi-v1 | 5 | 573 (98 + 400 + 75) |

## 投入

```
cd ~/Belief-Score-Based-Model-trysvgd-20261026
pjsub jobs/trysvgd/debug_trysvgd.sh
( setsid ~/gate_trysvgd.sh <DEBUG の JID> > ~/gate_trysvgd.log 2>&1 < /dev/null & )
```
