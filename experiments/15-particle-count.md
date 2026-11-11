# 15 粒子数を増やせば良くなるか（K = 16 / 64 / それ以上）

- **問い**: light TF-SVGD の粒子数 K を 16 から 64、さらに上へ増やすと信念と方策は良くなるか（ユーザー 2026-11-10「粒子数を増やした実験もしてみて。K = 64 で。もっと増やせるならもっと増やしたやつも」）。[09](09-representation-size.md) は K × D を GRU の隠れ次元に合わせた比較（mirror2 K80 × D2、tagr K36 × D6）で、ここは潜在次元を変えず K だけを増やす。
- **腕（campaign）**: `trysvgdlight-<task>-tf-svgd-k{64,256,1024}-v1`（N(0, I) 開始）vs 素の light SVGD（K = 16: `trysvgdlight-<task>-tf-svgd-v1` / `-ctl-v1`）。一様開始版 `trysvgdlight-<task>-tf-svgd-k{64,256,1024}-uniform-init-v1`（`langevin_proposal=uniform`、ユーザー 2026-11-10「K を増やした版に一様の方も追加」。対照は [04](04-chain-start.md) の K16 uniform-init）。
- **課題**: mirror2（333 update）/ tps（213）/ tagr（1293）/ Car-Flag（333）× 5 seed。計算量は合わせない（遷移の粒子対と注意は K² に比例）。
- **設定とジョブ**: `../config/trysvgdlight_<task>_tf-svgd-k<K>.json`（`model.num_particles` だけ変更）、`../jobs/trysvgd_light/trysvgdlight_<task>_svgd-k<K>_s*.sh`。生成は scratchpad gen_k.py。
- **速度（math、他ジョブと同居した GPU 3 で mirror2 light を 3 update、2026-11-10）**: 1 update（ロールアウト + PPO 更新）は K = 64 で 30〜37 s、K = 128 で 45〜48 s、K = 256 で 86〜103 s（同条件の K = 16 は 35〜49 s）。GPU ピークメモリ 0.3 / 0.4 / 0.7 GB。K = 256 でも K = 16 の約 2.5 倍で済む。K = 1024 は 1 update 840〜870 s（PPO 更新 670〜800 s）、ピーク 7.0 GB で、ここから粒子対の K² 項が支配的（mirror2 1 run 約 80 h、tagr 約 310 h）。
- **状態**: ユーザー決定（2026-11-10 05:00）「tagr の K = 1024 以外投入 OK。全 GPU が空き次第、全 GPU を使って tagr K = 1024 を実行」。K = 64 / 256（40 本）と K = 1024 の mirror2 / tps / Car-Flag（15 本）を math の待ち行列へ（seed 10–12 が先、13–14 は最後尾）。tagr K = 1024（5 seed、1 run 約 310 h）は待ち行列と GPU が空になってから 1 GPU 1 本で 4 本同時、5 本目は空いた GPU で（run1102/tagr_k1024_watch.sh）。一様開始版 55 本も投入（ユーザー「256 / 1024 は後ろで OK。64 の 3 seed だけ優先に追加」）: K64 uniform の seed 10–12 は 11-10 ブロックの末尾、K64 s13–14・K256 全 seed は最後尾。K = 1024 は N(0, I) 版・一様版とも全 seed を待ち行列の最後尾へ（ユーザー「K1024 は全部後ろで OK」）。1 run は 1 GPU しか使えない（コードにマルチ GPU の仕組みはない）。

## 結果の解説

- 参考（K80 × D2 の mirror2、K36 × D6 の tagr）: [09-representation-size.md](09-representation-size.md)、[results_summary_1109.md](../results/reports/summary_2026-11-09/results_summary_1109.md)
- 訓練中のエピソード報酬分布（探索の確認、2026-11-10）: light tagr は最初の 40 万 step から約 10 % のエピソードで報酬 6 超（最大 20）を見ているが分布が動かない。GRU s10 / PF v3 は同じ分布から p90 を 6 → 8〜9 に動かした。scratchpad explore_check.py（math run1102）。

## 結論メモ

- （未）
- **K = 64 / 256（2026-11-11、seed 10–12、mean_chain）**: mirror2 K64 −429 / −421 / −348（平均 −399）、K256 −416 / −391 / −414（−407）、K64 一様 −427 / −416 / −376（−406）。tps K64 −288、K256 −281、K64 一様 −285。tagr K64 2.2 / 2.5 / 2.6（s12 途中）、K64 一様 2.7 / 2.4 / 2.8、K256 2.8 / 2.2 / 3.5（途中、update 375〜750）。**K16 の素（mirror2 −369、tps −290、tagr 2.2〜2.6）と同じ帯**。粒子数は 16 → 256 の範囲で結果を変えない。計算時間は K256 で 1 update 170〜190 s（混雑時、K16 の 4〜5 倍）。
