# 27 観測アンカーを毎 step の開始に入れると（と、その輸送 1 段版）

- **問い**: (1) 観測アンカー（観測から出す平行移動 μ(o) とスケール σ(o)）を t = 0 だけでなく毎 step の連鎖の開始に入れると、t = 0 と t ≥ 1 の尺度がそろい、アンカーなし（v2）で起きた学習の遅れも消えるか。(2) そのとき、アンカーだけで学習していないか。輸送を 12 段から 1 段に減らしても成績が変わらなければ、スコアによる輸送はほとんど働いていない（ユーザー 2026-11-15「毎ステップの開始にアンカーを入れる案を試して」「観測アンカーが強すぎてアンカーだけで学習している可能性があるのかは気になったので、拡散 1 step の観測アンカーモデルも実験してほしい」）。
- **新しい選択肢**: `model.langevin_observation_anchor_steps`（`"initial"` = 既定・これまでの全 run と bit 一致、`"all"` = 毎 step）。`"all"` では step t の連鎖が x_start = μ(o_t) + σ(o_t)·ξ から始まる（ξ は t ≥ 1 の素のノイズ: N(0, I)、一様開始なら U(−√3, √3)）。アンカーの網は t = 0 と共通の 1 個（観測だけが入力、最終層 0 初期化なので学習開始時は素の開始と同じ）。学習する開始分布（affine / network）と warm start とは併用不可。seed バッチ経路は未対応（拒否する）。
- **func-init（[04](04-chain-start.md)）との違い**: func-init は前時刻の粒子を入力に取り、開始点が時間方向に再帰する。こちらは現在の観測だけなので再帰しない。
- **腕（campaign）**: v1 の素・v2 の素のそれぞれに 2 腕、計 4 腕。

  | 腕 | campaign | 素との差（`model.`） |
  |---|---|---|
  | v1 + 毎 step アンカー | `trysvgdlight-<task>-tf-svgd-anchor-all-v1` | `langevin_observation_anchor_steps: all` |
  | v1 + 毎 step アンカー、輸送 1 段 | `trysvgdlight-<task>-tf-svgd-anchor-all-l1-v1` | 上に加えて `langevin_steps: 1` |
  | v2 + 毎 step アンカー | `trysvgdlight2-<task>-tf-svgd-anchor-all-v1` | `langevin_observation_anchor: true`、`langevin_observation_anchor_steps: all`（一様開始・非有界・−x なしは v2 のまま） |
  | v2 + 毎 step アンカー、輸送 1 段 | `trysvgdlight2-<task>-tf-svgd-anchor-all-l1-v1` | 上に加えて `langevin_steps: 1` |

  対照は既存の v1 の素（`trysvgdlight-<task>-tf-svgd-v1`）と v2 の素（`trysvgdlight2-<task>-tf-svgd-v1`）、基準線は GRU / PF v3。
- **課題**: mirror2（333 update）/ tps（213）/ Car-Flag（333）/ tagrs（1293、[24](24-slow-tag.md)）× seed 10–14 = 80 本。鬼ごっこは腕の差が出る tagrs にした（tagr はどの腕も 3.3〜3.7 に固まるため。私の判断）。
- **注意**: 輸送 1 段の腕は行動 1 step の計算量が小さい（スコアの評価が 12 回 → 1 回）。FLOPs をそろえた比較ではなく、「輸送が働いているか」の切り分け用。
- **設定とジョブ**: `../config/trysvgdlight{,2}_<task>_tf-svgd-anchor-all{,-l1}.json`、`../jobs/anchor_all/*_s{10..14}.sh`、生成 `scripts/anchor_all_generate.py`、テスト `src/tests/test_anchor_all.py`（既定が bit 一致、毎 step の開始点が式どおり、アンカーへの勾配、併用不可の組、各腕が素と上の表のキーだけ違うこと）。
- **状態**: 2026-11-15、実装・設定 16 個・ジョブ 80 本・テストを準備。math の GPU で autograd と本番設定（解析スコア + CUDA Graphs）を 4 update 照合し、12 段・1 段とも損失と収益が表示桁まで一致（`timing_aall/`）。同日 13:03 に math へ 80 本を投入（木 v34、ユーザー「投入はおｋ」）: seed 10–12 の 48 本は待ちの先頭（tps → mirror2 → Car-Flag → tagrs、各課題の中は 1 段の腕が先）、seed 13–14 の 32 本は最後尾。結果待ち。

## 結果の解説

未取得。読み方: 「毎 step アンカー 12 段」が素より良く「1 段」が同じなら、アンカー（観測 → 粒子の直通）だけで足りていてスコアの輸送は寄与が小さい。「1 段」が大きく落ちるなら輸送が働いている。

## 結論メモ

- （結果待ち）
