# 13 表現を予測だけで学び PPO から切ると（prediction_only）

- **問い**: PPO の勾配を表現から外し、信念（または再帰状態）を次観測予測だけで学ぶと、Dreamer 型の「表現学習と制御の分離」は同じ課題で効くか。学習が「観測 → 終端スコア → condition」の短い経路に偏り、PPO と予測がその経路を共有している構造（[analyses.md](analyses.md) の勾配帰属）への対照。
- **仕組み**: `model.belief_training = prediction_only`。予測損失は予測フェーズに加えて PPO の全ロールアウトにもかける。PPO は actor / critic ヘッドのみ（条件ベクトルに stop-gradient）。GRU / PF v3 にも同じ扱い（AIS 型）。
- **腕（campaign）**: `trysvgdlight-<task>-tf-svgd-predonly-v1`、`trysvgd-<task>-gru-predonly-v1`、`trysvgd-<task>-pf-v3-predonly-v1` vs [12](12-obs-leak.md) の対照 `*-ctl-v1`（joint）。
- **課題**: tagr（1293 update）、mirror2（333）。seed 10〜14。
- **設定とジョブ**: `../jobs/trysvgd_predonly/`（README に表）、`../config/*-predonly.json`。木 v17（commit 3f75bc4）。
- **状態**: 30 本投入済み。seed 10〜11 は leak の seed 10〜11 の直後、seed 12〜14 は leak の seed 12〜14 の後。

## 結果の解説

- 動機: [GRAD_REPORT.md](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)
- Dreamer との対比（表現を誰が形づくるか）: [14-dreamer.md](14-dreamer.md)
- 結果: 完走後に追記。

## 読み方

- light が改善 → PPO の勾配が信念を壊していた（共適応・ドリフト）側。悪化 → 1 step 予測だけでは制御に要る情報（記憶）が入らない。
