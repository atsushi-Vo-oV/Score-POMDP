# 12 現在観測を方策に直結すると差は埋まるか（leak 2 × 3）

- **問い**: 信念の読み出しが現在観測を落としていることがボトルネックか、記憶の差か。全腕に同じオプションを入れて切り分ける。
- **仕組み**: `model.policy_observation_input = true` で actor / critic の入力が [c_t, o_t]。予測ヘッドは c_t のまま。FLOPs の増加 0.009 MFLOP（0.5 %）。
- **腕（campaign）**: light / GRU / PF v3 × {対照 `*-ctl-v1`, leak `*-obs-v1`}（`trysvgdlight-<task>-tf-svgd-{ctl,obs}-v1`、`trysvgd-<task>-gru-{ctl,obs}-v1`、`trysvgd-<task>-pf-v3-{ctl,obs}-v1`）。対照腕は [13](13-prediction-only.md) の対照でもある。
- **課題**: tagr（1293 update）、mirror2（333）。seed 10〜14。
- **設定とジョブ**: `../jobs/trysvgd_obs/`（README に表）、`../config/*-obs.json`、`../config/trysvgd_<task>_gru.json`。木 v16（commit 96e0213）。
- **状態**: 60 本投入済み。seed 10〜11 を先頭、seed 12〜14 は予測専用の seed 10〜11 の後。

## 結果の解説

- 動機（GRU は観測を積分し、スコア腕は観測より少ない。mirror2 では現在観測から読める m を粒子が捨てている）: [MDN_REPORT.md](../results/reports/mdn_2026-11-07/MDN_REPORT.md)
- 改善案としての位置づけ（Codex 案 1）: [WHY_REPORT.md §4](../results/reports/codex_why_2026-11-07/WHY_REPORT.md)
- 結果: 完走後に追記。

## 読み方

- light だけ大きく伸びる → 読み出しのボトルネック。GRU も同じだけ伸びる → 課題が直接観測を好むだけで差は記憶側。両方 leak しても残る差 → 記憶・表現の差。
