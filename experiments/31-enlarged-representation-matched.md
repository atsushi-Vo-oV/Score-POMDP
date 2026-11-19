# 31 全体を大きくし、表現の次元と計算量をそろえると（256 次元、約 12 MFLOP）

- **問い**: light TF-SVGD の潜在（鬼ごっこで 6 次元、Hike で 2 次元）とスコアの網（隠れ幅 8）は小さい。全体を大きくし、時刻をまたいで運ぶ表現の次元と行動 1 step の計算量を全腕でそろえたとき、light（素と、遷移の網の初期値を置いた版）は GRU / PF v3 と比べてどうか（ユーザー 2026-11-18「潜在 6 ってものすごく小さい気がしたので、全体的なアーキテクチャで大きくした版で統一して比較した実験」「表現次元統一の実験も」「サイズは大きくしてもいいよ！可能なら大体の計算時間は一緒くらいになるように」）。
- **表現の次元の数え方**: GRU は隠れ状態の次元、粒子の腕は 粒子の状態次元 × 粒子数。全腕 256 にそろえる。

  | 腕 | 運ぶ表現 | 大きくした所 | 行動 1 step の計算量 | パラメータ数 |
  |---|---|---|---|---|
  | light（v2 / v1、素 / 初期値） | 16 粒子 × 16 次元 = 256 | `belief_latent_dim` 2〜6 → 16、`energy_hidden` [8] → [32] | 11.7〜12.0 MFLOP（今の 7〜12 倍） | 4.3〜5.0 万 |
  | PF v3 | 16 粒子 × 16 次元 = 256（今と同じ） | `pf_hidden` を計算量に合わせて拡大（例 mirror2 [1347, 2394]） | light に一致 | 約 350〜360 万 |
  | GRU | 隠れ状態 256（今は 132〜217） | `gru_hidden_dim` 256、`gru_encoder_hidden` を計算量に合わせて拡大（例 mirror2 [3470, 1335]） | light に一致 | 約 580〜590 万 |

  粒子数 16・輸送 12 段・読み出し（Transformer、d_model 32）・方策の頭は今のまま。幅合わせは scratchpad の `flops_rep256.py`（これまでの FLOPs 合わせと同じ測り方、的は大きくした light v2）。
- **腕（campaign）**: 6 腕 + 2026-11-19 に追加した 3 腕（[30](30-transition-initialisation.md) の近い前粒子へ引く形と t = 0 を狭く始める形）。

  | 腕 | campaign |
  |---|---|
  | light v2 | `trysvgdlight2-<task>-tf-svgd-r256-v1` |
  | v2-initialized（[30](30-transition-initialisation.md) の初期値、強さ 4） | `trysvgdlight2-<task>-tf-svgd-tinit-r256-v1` |
  | light v1 | `trysvgdlight-<task>-tf-svgd-r256-v1` |
  | v1-initialized | `trysvgdlight-<task>-tf-svgd-tinit-r256-v1` |
  | v2-initialized-mix（c = 4） | `trysvgdlight2-<task>-tf-svgd-tinitmix-r256-v1` |
  | v1-initialized-mix（c = 4） | `trysvgdlight-<task>-tf-svgd-tinitmix-r256-v1` |
  | v2-initialized-t0（q = 4） | `trysvgdlight2-<task>-tf-svgd-tinitq-r256-v1` |
  | GRU | `trysvgd-<task>-gru-r256-v1` |
  | PF v3 | `trysvgd-<task>-pf-v3-r256-v1` |

- **課題**: mirror2（333 update）/ tps（213）/ tagr（1293）/ ml10 / ml30 / cpv（各 333）/ pdv（1293）× seed 10–14 = 210 本。追加の 3 腕は 7 課題 × seed 10–14 = 105 本（計 315 本）。
- **注意**:
  - この大きさでは隠れ幅 32 = 潜在 16 × 2 なので、遷移の網の初期値は全課題で厳密な形（`pair`）が入る。鬼ごっこだけ近似（`single`）にする必要がない。
  - light の粒子は交換可能な集合なので、256 個の数が自由に使えるわけではない。次元の数はそろえたが、情報量をそろえたわけではない。light の読み出しは粒子に加えてスコア（同じ 256 個）も読む。PF は粒子ごとの重み 16 個も運ぶ。
  - これまでに一部だけ大きくした実験は改善しなかった: 潜在だけ 16 次元（[16](16-latent-dimension.md)、隠れ幅 8 のまま、v1、mirror2）、粒子数 × 潜在を GRU の隠れ次元に合わせる（[09](09-representation-size.md)）。どちらも素の light。
  - GPU の所要時間は未計測（計算量は 7〜12 倍だが、light は GPU の命令の発行が律速なので比例はしない見込み。私の推察）。
- **小さな試験（CPU、N = 10 の保持、最後の時刻だけの教師、light v2、seed 10 / 11）**: 大きくしても素は学べず、初期値を置くと学べる。

  | 大きさ | 素 | 初期値（`pair`、強さ 4） |
  |---|---|---|
  | 潜在 2、隠れ幅 8（今） | 800 更新で未達 | 100 / 125 更新 |
  | 潜在 8、隠れ幅 16 | 400 更新で未達（正解率 0.47 / 0.46） | 100 / 100 更新 |
  | 潜在 16、隠れ幅 32 | 400 更新で未達（正解率 0.53 / 0.45） | 100 / 75 更新 |

  記録は `results/reports/nocarry_2026-11-18/spot_more/`（`run_large.sh`、`m1/v2L*`）。
- **設定とジョブ**: `../config/trysvgdlight{,2}_<task>_tf-svgd{,-tinit}-r256.json`、`../config/trysvgd_<task>_{gru,pf_v3}-r256.json`、`../jobs/rep256/*_s{10..14}.sh`、生成 `scripts/rep256_generate.py`、テスト `src/tests/test_rep256.py`（各腕が元の設定と表のキーだけ違うこと、全腕が 256 個を運ぶこと、初期値が厳密な形で入ること）。
- **状態**: 2026-11-18、設定 42 個・ジョブ 210 本・テストを準備。2026-11-19、追加の 3 腕の設定 21 個・ジョブ 105 本を準備。未投入（math の再起動後に GPU で所要時間を測り、投入 OK をもらってから）。

## 結果の解説

未取得。

## 結論メモ

- （結果待ち）
