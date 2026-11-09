# 12 現在観測を方策に直結すると差は埋まるか（leak 2 × 3）

- **問い**: 信念の読み出しが現在観測を落としていることがボトルネックか、記憶の差か。全腕に同じオプションを入れて切り分ける。
- **仕組み**: `model.policy_observation_input = true` で actor / critic の入力が [c_t, o_t]。予測ヘッドは c_t のまま。FLOPs の増加 0.009 MFLOP（0.5 %）。
- **腕（campaign）**: light / GRU / PF v3 × {対照 `*-ctl-v1`, leak `*-obs-v1`}（`trysvgdlight-<task>-tf-svgd-{ctl,obs}-v1`、`trysvgd-<task>-gru-{ctl,obs}-v1`、`trysvgd-<task>-pf-v3-{ctl,obs}-v1`）。対照腕は [13](13-prediction-only.md) の対照でもある。
- **課題**: tagr（1293 update）、mirror2（333）。seed 10〜14。
- **設定とジョブ**: `../jobs/trysvgd_obs/`（README に表）、`../config/*-obs.json`、`../config/trysvgd_<task>_gru.json`。木 v16（commit 96e0213）。
- **状態**: seed 10〜11 の 24 本は完走（2026-11-08）。seed 12〜14 は従来の待ち行列の後ろ。

## 結果の解説

- 動機（GRU は観測を積分し、スコア腕は観測より少ない。mirror2 では現在観測から読める m を粒子が捨てている）: [MDN_REPORT.md](../results/reports/mdn_2026-11-07/MDN_REPORT.md)
- 改善案としての位置づけ（Codex 案 1）: [WHY_REPORT.md §4](../results/reports/codex_why_2026-11-07/WHY_REPORT.md)
- 結果（seed 10 / 11、決定的評価）: [results_summary_1109.md](../results/reports/summary_2026-11-09/results_summary_1109.md)

## 結果メモ（seed 10 / 11、2 seed なので傾向のみ）

| 課題 | 腕 | 対照 ctl | leak obs |
|---|---|---|---|
| tagr | light TF-SVGD | 2.40 / 2.20 | 2.41 / 2.40 |
| tagr | GRU | 5.29 / 2.40 | 4.15 / 5.12 |
| tagr | PF v3 | 3.39 / 4.49 | 2.40 / 2.86 |
| mirror2 | light TF-SVGD | −397 / −342 | −347 / −481 |
| mirror2 | GRU | −839 / −1947 | −1068 / −180 |
| mirror2 | PF v3 | −1277 / −1168 | −1029 / −307 |

## 結論メモ

- light は leak で変わらない（tagr 2.4 → 2.4、mirror2 ±100）。「読み出しが現在観測を落としているのがボトルネック」という説明は、2 seed の範囲では支持されない。light の tagr は何をしても 2.2〜2.6 で、方策が低い局所解にいる。
- GRU / PF v3 は leak で seed により大きく動く（mirror2 の GRU s11 −1947 → −180、PF s11 −1168 → −307、tagr の PF は悪化）。基準線の分散が大きく、2 seed では方向を決められない。

## 読み方

- light だけ大きく伸びる → 読み出しのボトルネック。GRU も同じだけ伸びる → 課題が直接観測を好むだけで差は記憶側。両方 leak しても残る差 → 記憶・表現の差。
