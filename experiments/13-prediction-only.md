# 13 表現を予測だけで学び PPO から切ると（prediction_only）

- **問い**: PPO の勾配を表現から外し、信念（または再帰状態）を次観測予測だけで学ぶと、Dreamer 型の「表現学習と制御の分離」は同じ課題で効くか。学習が「観測 → 終端スコア → condition」の短い経路に偏り、PPO と予測がその経路を共有している構造（[analyses.md](analyses.md) の勾配帰属）への対照。
- **仕組み**: `model.belief_training = prediction_only`。予測損失は予測フェーズに加えて PPO の全ロールアウトにもかける。PPO は actor / critic ヘッドのみ（条件ベクトルに stop-gradient）。GRU / PF v3 にも同じ扱い（AIS 型）。
- **腕（campaign）**: `trysvgdlight-<task>-tf-svgd-predonly-v1`、`trysvgd-<task>-gru-predonly-v1`、`trysvgd-<task>-pf-v3-predonly-v1` vs [12](12-obs-leak.md) の対照 `*-ctl-v1`（joint）。
- **課題**: tagr（1293 update）、mirror2（333）。seed 10〜14。
- **設定とジョブ**: `../jobs/trysvgd_predonly/`（README に表）、`../config/*-predonly.json`。木 v17（commit 3f75bc4）。
- **状態**: seed 10〜11 の 12 本は完走（2026-11-08）。seed 12〜14 は従来の待ち行列の後ろ。

## 結果の解説

- 動機: [GRAD_REPORT.md](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)
- Dreamer との対比（表現を誰が形づくるか）: [14-dreamer.md](14-dreamer.md)
- 結果（seed 10 / 11、決定的評価）: [results_summary_1109.md](../results/reports/summary_2026-11-09/results_summary_1109.md)

## 結果メモ（seed 10 / 11）

| 課題 | 腕 | 対照 ctl（joint） | prediction_only |
|---|---|---|---|
| tagr | light TF-SVGD | 2.40 / 2.20 | 2.60 / 2.26 |
| tagr | GRU | 5.29 / 2.40 | 2.61 / 2.20 |
| tagr | PF v3 | 3.39 / 4.49 | 2.10 / 2.05 |
| mirror2 | light TF-SVGD | −397 / −342 | −379 / −501 |
| mirror2 | GRU | −839 / −1947 | −833 / −1468 |
| mirror2 | PF v3 | −1277 / −1168 | −1717 / −349 |

## 結論メモ

- 表現を 1 step 予測だけで学ぶと、tagr の GRU（5.3 / 2.4 → 2.6 / 2.2）と PF v3（3.4 / 4.5 → 2.1 / 2.0）は light と同じ 2 台まで落ちる。1 step 先の観測予測は制御に要る情報（特に鬼から逃げる向き）を表現に残さない。
- light は変わらない（元から 2 台）。PPO の勾配が信念を壊していたという説明は支持されない。
- mirror2 は seed で割れる（PF s11 は −1168 → −349 と改善、s10 は悪化）。2 seed では方向を決められない。

## 読み方

- light が改善 → PPO の勾配が信念を壊していた（共適応・ドリフト）側。悪化 → 1 step 予測だけでは制御に要る情報（記憶）が入らない。
