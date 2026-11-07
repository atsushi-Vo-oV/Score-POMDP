# 12 予測だけで表現を学ぶ（actor–critic からは止める）

- **目的**: PPO の勾配を表現から外し、信念（または再帰状態）を次観測予測だけで学ぶ。Dreamer 型の「表現学習と制御の分離」が同じ課題でどう出るか。短い経路に PPO と予測が同時に乗る構造（10 の勾配帰属）への対照でもある。
- **仕組み**: `model.belief_training = prediction_only`。予測損失は予測フェーズに加えて PPO の全ロールアウトにもかける。PPO は actor / critic ヘッドのみ（条件ベクトルに stop-gradient）。GRU / PF v3 にも同じ扱い（AIS 型）。
- **課題**: tagr（1293 update）、mirror2（333 update）。
- **設定とジョブ**: `../jobs/trysvgd_predonly/`（README に表）、`../config/*-predonly.json`。campaign `*-predonly-v1`。対照は 11 の `*-ctl-v1`。木 v17（commit 3f75bc4）。
- **状態（2026-11-07）**: 30 本を math に配置済み、投入待ち。

## 結果の解説

- 動機（学習が観測 → 終端スコア → condition の短い経路に偏る。PPO と予測が同じ経路を使う）: [GRAD_REPORT.md](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)
- Dreamer との対比（表現を誰が形づくるか）: [13-dreamer.md](13-dreamer.md)
- 結果: 完走後に追記。

## 読み方

- light が改善 → PPO の勾配が信念を壊していた（共適応・ドリフト）側。悪化 → 1 step 予測だけでは制御に要る情報（記憶）が入らない。
