# 05 t = 0 の尺度不一致を直せば良くなるか（prior / noanchor / flat）

- **問い**: t = 0 だけ別の初期網と観測 anchor を使う既定が尺度不一致を生む。anchor を外す（noanchor）、t = 0 を学習事前粒子からの再帰 step にする（prior、`model.initial_belief=prior`）、前粒子を平坦に連結する（flat、`direct_flat`）で良くなるか。
- **腕（campaign）**: `trysvgdlight-<task>-tf-svgd-{prior,noanchor,flat}-v1` vs 素の light SVGD。
- **課題**: mirror2 / tps / tagr × 5 seed。
- **設定とジョブ**: `../config/trysvgdlight_<task>_tf-svgd-{prior,noanchor,flat}.json`、`../jobs/trysvgd_light/*-{prior,noanchor,flat}_s*.sh`。実装の説明は commit 8a81405。
- **状態**: 完走（tagr flat s13 は走行中）。

## 結果の解説

- 評価値一覧: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)
- anchor の log-scale 上限張り付きの指摘: [IMPROVE_REPORT.final.md](../results/reports/steps_2026-11-04/IMPROVE_REPORT.final.md)
- 復号（MDN、prior / noanchor / flat は未収録。素と変種の表を参照）: [MDN_REPORT.md](../results/reports/mdn_2026-11-07/MDN_REPORT.md)

## 結論メモ

- tagr: prior 3.04、noanchor 4.19（s13 の 9.39 が引き上げ、他は 2.4〜3.9）、flat 2.86。mirror2: prior −632、noanchor −746、flat −886（崩壊 seed あり）。tps は 3 腕とも素と同等。
