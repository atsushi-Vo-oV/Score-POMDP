# 22 集団（多エージェント）による探索と、最適化専用エージェント

- **問い**: light TF-SVGD（Transformer 読み出し + Stein Variational Gradient Descent）の頭打ちは方策側の台地（[analyses.md](analyses.md) の 2026-11-13 の検証: 行動を 5〜20 step 続けて変えて初めて得になる）だった。信念を共有し actor だけを集団にして探索させると抜けられるか。どの部品が効き、どれが効かないか。
- **部品（すべて切り替え可能、既定 off）**:
  - A1 重み空間の集団（16 個体、初期摂動 std 0.01）／A2 時間相関のある行動ノイズ（OU: Ornstein–Uhlenbeck、std 0.3、時定数 20 step）
  - B 集団の更新則（5 PPO update ごと、排他）: CEM（Cross-Entropy Method）／ES（Evolution Strategies）／PBT（Population Based Training）／QD（Quality-Diversity、MAP-Elites + CEM）
  - C 探索個体ごとの PPO（Proximal Policy Optimization）
  - D 内発報酬 RND（Random Network Distillation、係数 0.01）
  - E 推論に使う actor（mean / best / optimizer）
  - F 最適化専用エージェント 1 個体（index 0。摂動・選抜・OU・内発報酬の対象外、外部報酬だけの PPO、専用の価値ヘッド。探索個体が対応差の窓 5 点・margin 0.1・下側信頼限界 > 0 で上回ったときだけ拡散ノイズ予測の模倣（BC: Behavior Cloning）で取り込む）
- **腕（campaign `popx-<task>-<arm>-v1`）**: 全部入り `full` と、そこから 1 つ外した `no_f` / `no_a2` / `no_c` / `no_d` / `no_qd`（QD→CEM）、単独の `cem` / `es` / `pbt` / `a2` / `d`、対照 `ppo`。12 腕。
- **課題**: Car-Flag（333 update）、mirror2（333）、tagr（1293）。信念も最初から学習（予測 pretrain 98 update を含む通常の学習）。seed 10–14。3 課題 × 12 腕 × 5 seed = 180 本。
- **対照の注意**: 全腕 64 環境 × 128 step（集団腕は 16 個体 × 4 環境）。update 数を既存の 16 環境の実験と揃えたので、環境 step は既存の 4 倍。既存の `trysvgdlight-<task>-tf-svgd-ctl-v1` とは直接比べず、同じ 64 環境の `ppo` 腕と比べる。F 自身の PPO データは 4 環境分（512 step / update）だけ。
- **設定とジョブ**: `../config/popx_scratch_<task>_<arm>.json`、`../jobs/popx/popx_scratch_<task>_<arm>_r{10..14}.sh`、対応表 [jobs/popx/manifest.json](../jobs/popx/manifest.json)、説明 [jobs/popx/README.md](../jobs/popx/README.md)。実装 `src/sb_pomdp/popexplore/`（`python -m sb_pomdp.popexplore`。本体の train / compare とは別経路で、本体のソースは変更していない）、テスト `src/tests/test_popexplore.py` / `test_popx_optimizer.py`（全 off が元の PPO と bit 一致することを含む）。解析 `scripts/popx_analyze.py`、速度計測 `scripts/popx_bench.sh`。
- **実行上の制約**: popexplore は解析スコア・CUDA Graphs を使わない（autograd）。木の `.sb_overrides` は読まない。
- **状態**: 2026-11-13、実装・CPU 検証（18 本の短縮実行は配線確認のみ）・設定 36 個・ジョブ 180 本を準備。未投入。

## 結果の解説

未取得。完了後、`scripts/popx_analyze.py 'results/campaigns/popx-*'` の出力（除去比較 = `no_x − full`、単独比較 = `x − ppo`、同じ seed の対応差と bootstrap 区間、F の採用履歴、失敗規則の判定）をここに書く。

## 結論メモ

- （結果待ち）
