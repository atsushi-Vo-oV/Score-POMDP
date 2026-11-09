# 16 粒子の潜在次元をそろえると（PF を 2 次元に、light を 16 次元に）

- **問い**: light TF-SVGD の潜在は状態次元（mirror2 / Car-Flag で 2）に縛られ、PF（DPFRL）の粒子は学習潜在 16 次元。この差が結果を分けているか。PF を 2 次元にしても勝ち続ければ問題は Score 腕の学習信号側、PF も light の水準に落ちれば「2 次元の潜在」が共通の制約。鏡像として light の潜在を 16 次元にして改善するかも見る（ユーザー 2026-11-10「pf の dim=2 と、light-TF-SVGD の dim=16 の投下 OK」）。
- **腕（campaign）**: `trysvgd-<task>-pf-v3-d2-v1`（`comparison.pf_particle_dim` 16 → 2、幅はその課題の PF v3 と同じ FLOPs に再一致: mirror2 [463, 823]（1.24 MFLOP）、Car-Flag [419, 744]（1.07 MFLOP））と `trysvgdlight-<task>-tf-svgd-d16-v1`（`model.belief_latent_dim` 2 → 16、計算量は合わせない: 3.4 MFLOP、素の 0.96〜1.07 の約 3.4 倍）。対照は PF v3（16 次元）と素の light（2 次元）。
- **課題**: mirror2 / Car-Flag × 5 seed × 2 腕 = 20 本（各 333 update）。
- **設定とジョブ**: `../config/trysvgd_<task>_pf_v3-d2.json`、`../config/trysvgdlight_<task>_tf-svgd-d16.json`、`../jobs/trysvgd/trysvgd_<task>_pf_v3-d2_s*.sh`、`../jobs/trysvgd_light/trysvgdlight_<task>_svgd-d16_s*.sh`。幅合わせは scratchpad flops_d2.py、生成は gen_d.py。
- **状態**: 生成・CPU スモーク済み。math へ投入（2026-11-10、seed 10–12 が先、13–14 は最後尾）。

## 結果の解説

- 関連: [09-representation-size.md](09-representation-size.md)（K × D を GRU の隠れ次元に合わせた比較）、[15-particle-count.md](15-particle-count.md)（K だけ増やす比較）。

## 結論メモ

- （未）
