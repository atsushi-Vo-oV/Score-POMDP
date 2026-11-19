# 30 遷移の網の初期値を「前の雲へ戻る場」にすると（v2-initialized / v1-initialized）

- **問い**: light TF-SVGD は、推論の式を変えず、遷移の網の重みの初期値だけを変えれば、長い記憶を学べるようになるか。記憶が要らない元のベンチマーク課題（多峰の mirror2 / tps、鬼ごっこ）で成績を落とさないか（ユーザー 2026-11-18「この形を v2-initialized として、基準とできる？ あと、元々ベンチとしてたタスクを追加して」）。
- **背景**: 小さな教師あり試験（N = 10 の保持、最後の時刻だけの教師、CPU）で、素の light v2 は 800 更新で未達、この初期値では 100 / 125 更新で解けた（[analyses.md](analyses.md) の「持ち越しなしの改善案の追加確認」(8)、[COMPARISON.md](../results/reports/nocarry_2026-11-18/COMPARISON.md) の 2d (6)）。
- **新しい選択肢**: `model.transition_persistence_init`（`null` = 既定・これまでの全 run と bit 一致、正の数 p = 初期値を置き換える）。置き換えると、遷移の網は組（x, x'_j, 行動）ごとに重み 0（16 個で一様）と向き `p·(x'_j − x)` を出す状態から学習を始める。16 個を足すと再帰 step の遷移の項は `p·(前の雲の平均 − x)` になる。スコアには何も足さない。重みは学習で自由に動く。行動への依存は初期値では 0。乱数は消費しないので、他の網の初期値は素と同じ。
  - `model.transition_persistence_units: "pair"`（既定）: 座標ごとに隠れユニット 2 個（`SiLU(z) − SiLU(−z) = z`）。厳密。隠れ幅 ≥ 潜在次元 × 2 が必要。
  - `"single"`: 座標ごとに隠れユニット 1 個を SiLU の変曲点（2 階微分が 0 の点、u0 = 2.399）に置く近似。隠れ幅 ≥ 潜在次元で置ける。z = x' − x が 0 の近くでは式どおり、z < −2.4 で引き戻しが頭打ちになり、大きな正の z では約 9 % 弱い。
  - `score_parameterization: direct`・`energy_network_kind: mlp`・隠れ層 1 つ、が条件。seed バッチ経路は未対応（拒否する）。
- **近い前粒子へ引く形（2026-11-19 追加、ユーザー「v2-initialized-（混合正規）って名前でそれによる初期化も実装できる？」）**: `model.transition_persistence_kernel`（`null` = 既定・上の一様な重み、正の数 c）。c を置くと、重みの元（logit）も `−c·Σ_d z_d·tanh(z_d / 2)`（z = x'_j − x）から始まる。同じ 2 個組のユニットの和 `SiLU(z) + SiLU(−z) = z·tanh(z / 2)` を使うので、追加のユニットは要らない。問い合わせの粒子は、雲全体の平均ではなく近くの前粒子へ引かれる。**これは一般には何かの分布のスコアではない**（Astra の指摘。logit が厳密に二次なら核密度の p / c 乗のスコアになるが、この形は 0 から離れると二次からずれ、場に回転成分が残る）。当初「混合正規のスコア」と説明したのは誤り。`units: pair` が条件なので、この大きさの tagr には置けない（[31](31-enlarged-representation-matched.md) の大きさでは置ける）。
- **t = 0 の雲を狭く始める形（2026-11-19 追加、Astra の案）**: `model.initial_score_projection_precision`（`null` = 既定、正の数 q）。初期の網が `q·(B o − x)`（観測を線形に写した点のまわりの、分散 1 / q の正規分布のスコア）から始まる。B は網が元々持っていた第一層の重みから取るので乱数は消費しない。座標ごとに隠れユニット 2 個（隠れ幅 ≥ 潜在次元 × 2）。v2 の設定（上限なし、t = 0 に −x なし）でだけ使える。t = 0 で多峰の雲を作る助けにはならない。
- **腕（campaign）**: 強さ p = 4（小さな試験で 4 と 11 は同じ速さ、4 のほうが雲の広がりが残る）。近い前粒子へ引く形は c = 4、t = 0 を狭く始める形は q = 4。

  | 腕 | campaign | 素との差（`model.`） |
  |---|---|---|
  | v2-initialized | `trysvgdlight2-<task>-tf-svgd-tinit-v1` | `transition_persistence_init: 4.0`（tagr だけ `transition_persistence_units: single` も） |
  | v1-initialized | `trysvgdlight-<task>-tf-svgd-tinit-v1` | 同上 |
  | v2-initialized-mix（近い前粒子へ引く形） | `trysvgdlight2-<task>-tf-svgd-tinitmix-v1` | `transition_persistence_init: 4.0`、`transition_persistence_kernel: 4.0`（tagr は無し） |
  | v1-initialized-mix | `trysvgdlight-<task>-tf-svgd-tinitmix-v1` | 同上 |
  | v2-initialized-t0（t = 0 を狭く始める） | `trysvgdlight2-<task>-tf-svgd-tinitq-v1` | `transition_persistence_init: 4.0`、`initial_score_projection_precision: 4.0`（v2 のみ、tagr は無し） |

  対照は既存の v2 の素（`trysvgdlight2-<task>-tf-svgd-v1`）と v1 の素（`trysvgdlight-<task>-tf-svgd-v1`）、基準線は GRU / PF v3（どれも走り終えた run を使う）。
- **課題**: 元のベンチマーク mirror2（333 update）/ tps（213）/ tagr（1293）と、記憶課題 ml10 / ml30 / cpv（各 333）/ pdv（1293、[29](29-literature-memory-tasks.md)）× seed 10–14 = 70 本。v2-initialized-mix / v1-initialized-mix は tagr を除く 6 課題 × seed 10–14 = 60 本、v2-initialized-t0 は同じ 6 課題 × seed 10–14 = 30 本。計 160 本。
- **小さな試験と小さな PPO（2026-11-19、CPU、2 seed）**: [COMPARISON.md](../results/reports/modefree_2026-11-19/COMPARISON.md)。記憶の鎖・位置を隠した移動・鏡像の移動（多峰）の小さな PPO で、3 課題とも安定していたのは v2-initialized-t0（記憶の鎖は収益 0.8 超えが 13 / 21 update、v2-initialized は 34 / 40、素と v2-initialized-mix は 100 update で未達）。v2-initialized-mix は単峰の記憶を学べない。前の雲の峰の数に依らない固定の c は無い（Astra）。p = 16・c = 4・q = 4（Astra の第一候補）は記憶の鎖 11 / 13 update・位置を隠した移動 0.90 / 0.95 だが、鏡像の移動は 2 seed とも学べなかった（腕としては未生成、設定の上書きで走らせられる）。
- **注意**:
  - 潜在次元は mirror2 / tps / ml10 / ml30 / pdv が 2、cpv が 4、tagr が 6。隠れ幅は 8 なので、`pair` は 4 次元まで（cpv は 8 個すべてを使う）。tagr は `single`。小さな試験（N = 10 の保持、light v2、seed 10 / 11）では、`single` は 2 次元で 100 / 125 更新（`pair` と同じ）、6 次元で 100 / 75 更新（6 次元の素は 400 更新で未達）。学習後の粒子の大きさは `pair` 1.2〜1.5、`single` 1.5〜1.8（[COMPARISON.md](../results/reports/nocarry_2026-11-18/COMPARISON.md) の 2d (8)）。`single` の入力の係数を 0.25 にすると初期値は式に近いが、学習で雲が外へ流れた（粒子の大きさ 5〜17）ので 1 にした。
  - 初期値の時点では、運ぶのは前の雲の平均だけ。前の雲が 2 群に分かれていると、中点の 1 群にまとめる（[COMPARISON.md](../results/reports/nocarry_2026-11-18/COMPARISON.md) の 2d (7)）。多峰の課題（mirror2 / tps）で悪くなる可能性があり、それを見るのもこの実験の目的。
- **設定とジョブ**: `../config/trysvgdlight{,2}_<task>_tf-svgd-tinit.json`、`../jobs/tinit/*_s{10..14}.sh`、生成 `scripts/tinit_generate.py`、テスト `src/tests/test_transition_persistence_init.py`（既定が bit 一致、遷移の網だけが変わり乱数を消費しない、初期値の場が式どおり、再帰 step のスコアが引き戻しの形で始まる、重みが学習で動く、無効な組、各腕が素と上の表のキーだけ違うこと、全腕のモデルが作れること）。
- **状態**: 2026-11-18、実装・設定 14 個・ジョブ 70 本・テストを準備。2026-11-19、近い前粒子へ引く形と t = 0 を狭く始める形の実装・設定 18 個・ジョブ 90 本・テストを追加。どれも未投入（投入 OK 待ち）。

## 結果の解説

未取得。読み方: ml10 / ml30 で記憶の切除（前時刻の粒子の差し替え）による落ち幅が出れば、初期値で長い記憶が学べている。mirror2 / tps で素より下がれば、平均へ寄せる初期値が多峰の信念を損ねている。

## 結論メモ

- （結果待ち）
