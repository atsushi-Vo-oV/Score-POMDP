# 11 観測直結（leak）比較: actor / critic に o_t を直接入れる

- **目的**: 信念の読み出しが現在観測を落としていることがボトルネックか、記憶の差かを切り分ける。全腕（light / GRU / PF v3）に同じオプションを入れ、対照も同じ木・同じ seed で走らせる 2 × 3。
- **仕組み**: `model.policy_observation_input = true` で actor / critic の入力が [c_t, o_t]。予測ヘッドは c_t のまま。FLOPs の増加は 0.009 MFLOP（0.5 %）。
- **課題**: tagr（1293 update）、mirror2（333 update）。
- **設定とジョブ**: `../jobs/trysvgd_obs/`（README に表）、`../config/*-obs.json`、`../config/trysvgd_<task>_gru.json`。campaign `*-ctl-v1` / `*-obs-v1`。木 v16（commit 96e0213）。
- **状態（2026-11-07）**: 60 本を math のキュー先頭に投入（seed-major、mirror2 → tagr）。枠が空き次第開始。

## 結果の解説

- 動機（MDN: GRU は観測を積分し、スコア腕は観測より少ない。mirror2 では現在観測から読める m を粒子が捨てている）: [MDN_REPORT.md](../results/reports/mdn_2026-11-07/MDN_REPORT.md)
- 改善案としての位置づけ（Codex 案 1）: [WHY_REPORT.md §4](../results/reports/codex_why_2026-11-07/WHY_REPORT.md)
- 結果: 完走後に `results_summary` を更新してここに追記。

## 読み方

- light だけ大きく伸びる → 読み出しのボトルネック。GRU も同じだけ伸びる → 課題が直接観測を好むだけで差は記憶側。両方 leak しても残る差 → 記憶・表現の差。
