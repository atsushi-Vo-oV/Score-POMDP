# 08 最適化器を TD3 にすると（PPO vs TD3）

- **問い**: オンポリシーの PPO をオフポリシーの TD3 に替えると、スコア信念と基準線の順位は変わるか。
- **腕（campaign）**: `trysvgdtd3-<task>-tf-{ula,svgd,smi}-v1`、基準線 `trysvgdtd3-<task>-{gru,rnn,pf}-v1` vs PPO 版（[01](01-sampler-energy.md)）。
- **課題**: mirror2 / tps。
- **設定とジョブ**: `ppo.algorithm=td3`（`../src/sb_pomdp/td3.py`）、`../config/trysvgdtd3_*.json`、`../jobs/trysvgd_td3/`。prediction_head=condition のみ対応。
- **状態**: 完走（2026-11-09、mirror2 / tps × TF-ULA / SVGD / SMI / GRU / RNN / PF × 5 seed）。

## 結果の解説

- 評価値一覧: [results_summary_1109.md](../results/reports/summary_2026-11-09/results_summary_1109.md)（campaign `trysvgdtd3-*`）

## 結論メモ

- mirror2 では TD3 は全腕で崩壊に近い（TF 系 −1430〜−1890、GRU 平均 −1330、PF −900、RNN −1800）。PPO 版（light −368、GRU v2 −701）より大きく悪い。
- tps では全腕 −290〜−390 で PPO 版と同等。TD3 への置換は mirror2 の学習を壊すだけで、腕間の順位の情報は得られない。
