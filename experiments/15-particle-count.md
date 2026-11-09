# 15 粒子数を増やせば良くなるか（K = 16 / 64 / それ以上）

- **問い**: light TF-SVGD の粒子数 K を 16 から 64、さらに上へ増やすと信念と方策は良くなるか（ユーザー 2026-11-10「粒子数を増やした実験もしてみて。K = 64 で。もっと増やせるならもっと増やしたやつも」）。[09](09-representation-size.md) は K × D を GRU の隠れ次元に合わせた比較（mirror2 K80 × D2、tagr K36 × D6）で、ここは潜在次元を変えず K だけを増やす。
- **腕（campaign）**: `trysvgdlight-<task>-tf-svgd-k64-v1`（K = 64）、`trysvgdlight-<task>-tf-svgd-k<K>-v1`（K > 64 は速度計測後に決定）vs 素の light SVGD（K = 16: `trysvgdlight-<task>-tf-svgd-v1` / `-ctl-v1`）。
- **課題**: mirror2（333 update）/ tps（213）/ tagr（1293）/ Car-Flag（333）× 5 seed。計算量は合わせない（遷移の粒子対と注意は K² に比例）。
- **設定とジョブ**: `../config/trysvgdlight_<task>_tf-svgd-k<K>.json`（`model.num_particles` だけ変更）、`../jobs/trysvgd_light/trysvgdlight_<task>_svgd-k<K>_s*.sh`。生成は scratchpad gen_k.py。
- **速度（math、他ジョブと同居した GPU 3 で mirror2 light を 3 update、2026-11-10）**: 1 update（ロールアウト + PPO 更新）は K = 64 で 30〜37 s、K = 128 で 45〜48 s、K = 256 で 86〜103 s（同条件の K = 16 は 35〜49 s）。GPU ピークメモリ 0.3 / 0.4 / 0.7 GB。K = 256 でも K = 16 の約 2.5 倍で済む。K = 1024 は 1 update 840〜870 s（PPO 更新 670〜800 s）、ピーク 7.0 GB で、ここから粒子対の K² 項が支配的（mirror2 1 run 約 80 h、tagr 約 310 h）。
- **状態**: K = 64 と K = 256 を生成・CPU スモーク済み、投入 OK 待ち。K = 1024 は速度計測のみ（投入するなら mirror2 / tps に限る）。

## 結果の解説

- 参考（K80 × D2 の mirror2、K36 × D6 の tagr）: [09-representation-size.md](09-representation-size.md)、[results_summary_1109.md](../results/reports/summary_2026-11-09/results_summary_1109.md)
- 訓練中のエピソード報酬分布（探索の確認、2026-11-10）: light tagr は最初の 40 万 step から約 10 % のエピソードで報酬 6 超（最大 20）を見ているが分布が動かない。GRU s10 / PF v3 は同じ分布から p90 を 6 → 8〜9 に動かした。scratchpad explore_check.py（math run1102）。

## 結論メモ

- （未）
