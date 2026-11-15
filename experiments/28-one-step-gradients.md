# 28 信念の勾配を 1 step で打ち切ると（保存した粒子から全行を同時に再生する版を含む）

- **問い**: (1) 信念の時刻間の勾配を 1 step で打ち切っても成績は保てるか。(2) 打ち切るなら、時刻 t の更新に必要な入力は「前時刻の粒子（定数）・観測・行動・ノイズ」だけなので、ロールアウト収集時に保存した前時刻の粒子を使えば全 (時刻, 環境) の行を 1 つのバッチで再生できる（A 方式）。これで速くなり、成績は保てるか。(3) 方策学習だけ 1 step にして、観測予測の学習には時刻間の勾配を残すとどうか（ユーザー 2026-11-15「方策学習時 1step-bptt かつ、観測復元学習時の時刻間の勾配を計算する実験ってしたっけ？」「1-step-bptt なら並列化できるようになる？」「v2 ベースに A 方式で実装してみて！あと、観測復元の時も同時に 1step-bptt にするのも気になる」）。
- **1 step 打ち切りの意味**: 時刻 t の損失（その時刻の方策・価値、または予測）の勾配は、時刻 t の信念の更新（その時刻の輸送 12 段とスコア網）にだけ流れ、前時刻の粒子より前へは戻らない。網は全時刻で共有なので全時刻のデータから学ぶが、「後の時刻で役に立つように、いま何を粒子に残すか」は教えられない。報酬の時間方向の伝播（GAE: Generalized Advantage Estimation）はそのまま。
- **新しい選択肢**（どちらも既定は従来と bit 一致）:
  - `model.belief_replay`: `sequence`（既定、時刻順）/ `stored`（A 方式。`belief_gradient_mode: tbptt_1` が必要）。PPO と critic の更新に効く。
  - `model.prediction_replay`: `same`（既定、PPO と同じ）/ `full` / `tbptt_1` / `stored`。予測フェーズ（事前学習と交互学習）だけ別にできる。
  - `stored` は、更新の最初の勾配 step では時刻順の `tbptt_1` と値も勾配も一致する（テストで確認）。その後は前時刻の粒子が収集時のパラメータのもの（作り直さない）になる。curl 罰則・seed バッチ経路・集団探索とは併用不可。
- **腕（campaign）**: v2 の素・v1 の素のそれぞれに 3 腕。

  | 腕 | campaign | PPO の更新 | 予測フェーズ |
  |---|---|---|---|
  | 素（既存） | `trysvgdlight{,2}-<task>-tf-svgd-v1` | 時刻間の勾配あり | 時刻間の勾配あり |
  | bptt1 | `…-tf-svgd-bptt1-v1` | 1 step、時刻順 | 1 step、時刻順 |
  | stored | `…-tf-svgd-stored-v1` | 1 step、A 方式 | 1 step、A 方式 |
  | stored-predfull | `…-tf-svgd-stored-predfull-v1` | 1 step、A 方式 | 時刻間の勾配あり |

  bptt1 と stored の差が「前時刻の粒子を作り直さないこと」の影響、素と bptt1 の差が「時刻間の勾配」の寄与、stored と stored-predfull の差が「予測側の時刻間の勾配」の寄与。
- **課題**: mirror2（333 update）/ tps（213）/ Car-Flag（333）/ tagrs（1293）× seed 10–14。2 系統 × 3 腕 × 4 課題 × 5 seed = 120 本。
- **注意**: 1 step の腕は結果のディレクトリが `…/tbptt_1/score_transformer/…` になる（従来は `…/full/…`）。math の評価・ビューア用スクリプトは両方を拾うよう直す。
- **設定とジョブ**: `../config/trysvgdlight{,2}_<task>_tf-svgd-{bptt1,stored,stored-predfull}.json`、`../jobs/bptt1/*_s{10..14}.sh`、生成 `scripts/bptt1_generate.py`、テスト `src/tests/test_stored_replay.py`。
- **状態**: 2026-11-15、実装・設定 24 個・ジョブ 120 本・テストを準備。未投入。

## 結果の解説

未取得。

## 結論メモ

- （結果待ち）
