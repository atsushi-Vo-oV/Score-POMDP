# try-svgd — GRU / RNN / DPFRL と score transformer (ULA / SVGD / SMI) を多峰 2 タスクと鬼ごっこで比較

- **タスク**: mirror2(鏡像 Mountain Hike)、tps(|s| 観測 + テレポート)、tag(円盤鬼ごっこ: 鬼 3 体は視野 180°・距離 1.0
  でうろつき、見つけたら追跡、見失ったら最後の地点を探索。プレイヤーは視野 60°・距離 0.6 の中の鬼の数だけを観測。
  1 step 生存ごとに +0.1)。
- **モデル**: GRU、RNN、DPFRL(粒子フィルタ)、TF-ULA、TF-SVGD、TF-SMI。transformer は K=16、L=12、エネルギー MLP [8]、
  d_model 32、ff 64、**warm start なし**、t=0 観測アンカーあり、α_l は 0.2 から sigmoid 境界で学習(スケジュール学習率は方策と同じ 0.00025。以前ここに 0.0025 と書いていたのは誤り: config に langevin_schedule_lr_multiplier はなく既定 1.0)。
  ULA は β=0.2 固定、SVGD / SMI は雑音なし。SMI は成分の log σ を encoder に渡す。tag の潜在次元は 6。
- **FLOPs**: transformer に統一(推論 1 step、FlopCounterMode)。基準線は core4 の幅を一様に拡大。

**2026-11-02 の数え直し**: FlopCounterMode は eval + no_grad の融合 transformer カーネル(fast path)を数えないため、以前の
TF の値は transformer 本体(約 0.26 MFLOP)を欠いていた。fast path を切って数え直すと TF は mirror2 1.382 / tps 1.368 /
tagr 2.297 MFLOP。v1 の基準線(下の旧表)は TF より 13〜24% 小さい予算に合わせていたことになる。基準線を合わせ直して
mirror2 / tps は **trysvgd-{mirror2,tps}-base-v2**(jobs/trysvgd/*_v2_*.sh、math サーバー)として走り直す。

| task | TF ULA / SVGD / SMI (MFLOP) | GRU (params) | RNN (params) | DPFRL (params) | TF params |
|---|---|---|---|---|---|
| mirror2 | 1.382 / 1.394 / 1.408 | 1.384 (571k) | 1.382 (569k) | 1.382 (163k) | 41k |
| tps | 1.368 / 1.38 / 1.395 | 1.368 (562k) | 1.368 (561k) | 1.368 (159k) | 39k |
| tagr | 2.297 / 2.333 / 2.376 | 2.297 (1030k) | 2.297 (1030k) | 2.297 (358k) | 44k |

**2026-11-04 DPFRL の観測適合度ヘッド(pf v3)**: v1 / v2 の DPFRL は観測適合度が [h_i, e] の連結に対する 1 層線形
(aᵀh_i + bᵀe + c)で、観測項 bᵀe + c が粒子間で共通のため粒子ごとの log-softmax で相殺され、重み更新が粒子と観測の
相互作用を表せない(Astra の再解析で判明、`baselines.py` で確認)。`comparison.pf_observation_function` を追加し
(`linear` = 従来、`mlp` = 観測を 32 次元に射影してから粒子と結合する 1 隠れ層 MLP、`bilinear` = h_iᵀ W proj(e) + aᵀh_i)、
v3(mlp)と v3b(bilinear、mirror2 のみ)として走らせる。幅は現コードで数え直した energy TF-ULA(mirror2 1.239 / tps 1.226 /
tagr 2.154 MFLOP。11/2 以降に不要な 2 回目の denoiser 計算が消えたので TF・基準線とも同じだけ減り、v2 の対応は保たれる)
に合わせた: mlp は mirror2 [212, 378]、tps [210, 374]、tagr [366, 651]、bilinear は mirror2 [221, 394]。

旧表(fast path を数えていない値。v1 の基準線はこれに合わせてあった):

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
| mirror2 | base | config/trysvgd_mirror2_base.json | trysvgd-mirror2-base-v1(旧幅)/ **-v2**(数え直した幅、sequence_microbatch_size 512) | 15 | 333 (98 + 200 + 35) |
| mirror2 | pf v3 / v3b | config/trysvgd_mirror2_pf_v3.json / _pf_v3b.json | trysvgd-mirror2-pf-v3 / -pf-v3b | 5 / 5 | 333 (98 + 200 + 35) |
| mirror2 | tf-ula | config/trysvgd_mirror2_tf-ula.json | trysvgd-mirror2-tf-ula-v1 | 5 | 333 (98 + 200 + 35) |
| mirror2 | tf-svgd | config/trysvgd_mirror2_tf-svgd.json | trysvgd-mirror2-tf-svgd-v1 | 5 | 333 (98 + 200 + 35) |
| mirror2 | tf-smi | config/trysvgd_mirror2_tf-smi.json | trysvgd-mirror2-tf-smi-v1 | 5 | 333 (98 + 200 + 35) |
| tps | base | config/trysvgd_tps_base.json | trysvgd-tps-base-v1(旧幅)/ **-v2**(数え直した幅、sequence_microbatch_size 512) | 15 | 213 (98 + 100 + 15) |
| tps | pf v3 | config/trysvgd_tps_pf_v3.json | trysvgd-tps-pf-v3 | 5 | 213 (98 + 100 + 15) |
| tps | tf-ula | config/trysvgd_tps_tf-ula.json | trysvgd-tps-tf-ula-v1 | 5 | 213 (98 + 100 + 15) |
| tps | tf-svgd | config/trysvgd_tps_tf-svgd.json | trysvgd-tps-tf-svgd-v1 | 5 | 213 (98 + 100 + 15) |
| tps | tf-smi | config/trysvgd_tps_tf-smi.json | trysvgd-tps-tf-smi-v1 | 5 | 213 (98 + 100 + 15) |
| tag | base | config/trysvgd_tag_base.json | trysvgd-tag-base-v1 | 15 | 573 (98 + 400 + 75) |
| tag | tf-ula | config/trysvgd_tag_tf-ula.json | trysvgd-tag-tf-ula-v1 | 5 | 573 (98 + 400 + 75) |
| tag | tf-svgd | config/trysvgd_tag_tf-svgd.json | trysvgd-tag-tf-svgd-v1 | 5 | 573 (98 + 400 + 75) |
| tag | tf-smi | config/trysvgd_tag_tf-smi.json | trysvgd-tag-tf-smi-v1 | 5 | 573 (98 + 400 + 75) |

## 投入

DEBUG と gate は mirror2 と tps だけを対象にする(鬼ごっこは手元で学習できる設定を探してから)。

```
cd ~/Belief-Score-Based-Model-trysvgd-20261026
pjsub jobs/trysvgd/debug_trysvgd.sh
( setsid ~/gate_trysvgd.sh <DEBUG の JID> > ~/gate_trysvgd.log 2>&1 < /dev/null & )
```
