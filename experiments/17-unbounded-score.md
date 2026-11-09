# 17 スコアの有界化と t = 0 の −x は要るか（nobound / logbound）

- **問い**: light（直接スコア）は出力を 10·tanh(x/10) で有界化し、t = 0 の初期スコアに −x（N(0, I) 事前）を固定で足している。どちらも学習の制約で、必須かは未検証。有界化を外し、t = 0 の −x も外した light は、素の light と比べてどうか（ユーザー 2026-11-10「有界化は必須なの？非有界なモデルは比較できる？」「非有界なやつはついでに初期値の −x も消して」「もし発散するなら x/log(e + x²) の活性化も入れて OK」）。
- **腕（campaign）**: `trysvgdlight-<task>-tf-svgd-nobound-v1`（`model.direct_score_bound = null`、`model.initial_score_prior = 0`）vs 素の light（有界 10、prior 1）。発散した場合の控え: `trysvgdlight-<task>-tf-svgd-logbound-v1`（`direct_score_bound = "log"` = x / log(e + x²)、傾き 1 で始まり大きい |x| では x / (2 ln|x|) で伸びる非有界の圧縮、prior 0）。
- **課題**: mirror2 / tagr / Car-Flag × 5 seed（nobound 15 本、logbound 15 本は控え）。
- **設定とジョブ**: `../config/trysvgdlight_<task>_tf-svgd-{nobound,logbound}.json`、`../jobs/trysvgd_light/trysvgdlight_<task>_svgd-{nobound,logbound}_s*.sh`。実装: `model.direct_score_bound` に null / "log" を追加、`model.initial_score_prior`（既定 1.0、energy 版は −p|x|²/2）を追加。テスト `src/tests/test_score_bound_prior.py`。
- **事前診断（2026-11-10、学習済み checkpoint で有界化だけを外した再評価、32 話）**: tagr と Car-Flag は返り値・スコアとも不変（スコアの最大値 5〜9 で上限 10 に届かない）。mirror2 は s10 で −406 → −416、**s11 で −329 → −843、スコア最大 9,600、1 step の粒子移動 790**（tanh が飽和していた成分 4 % が解放されて発散）。学習済みモデルの mirror2 では有界化が効いている。非有界で最初から学習した場合は別問題（この実験）。
- **状態**: 生成・CPU スモーク済み。nobound を投入（seed 10–12 が先、13–14 は最後尾）。logbound は未投入（nobound が発散したら差し替え）。

## 結果の解説

- 事前診断のスクリプト: `C:\Users\Atsushi\AppData\Local\Temp\cxs\out\bound_check.py`、結果 `bound_check.csv`（results/reports/codex_stall_2026-11-10/ にコピー）。
- 関連: [05-t0-scale.md](05-t0-scale.md)（anchor と t = 0 の尺度）、[04-chain-start.md](04-chain-start.md)。

## 結論メモ

- （未）
