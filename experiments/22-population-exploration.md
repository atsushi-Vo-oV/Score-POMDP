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
- **実行**: 信念の計算は本体の関数をそのまま呼ぶので、解析スコア + CUDA Graphs（実行時の切替、`SB_POMDP_POPX_FAST=1`）が使える。木の `.sb_overrides` は読まない。math（Ada 4 枚を 13 本で共有）での実測は 1 update あたり、autograd で tagr `ppo` 94〜148 秒 / tagr `full` 127〜185 秒 / mirror2 `full` 195〜247 秒、解析スコア + CUDA Graphs と評価の同時実行（全話を 1 batch で進める）で 44〜72 / 50〜77 / 51〜67 秒。同じ seed の 6 update で、両者の損失・収益・評価値は表示桁まで一致（mirror2 は 4 桁目に浮動小数の差）。最適化専用エージェント自身の前後評価は、取り込みが起きた update とその次の PPO update だけ行う。
- **状態**: 2026-11-13、実装・CPU 検証（短縮実行は配線確認のみ）・math での速度計測・設定 36 + 6 個・ジョブ 180 + 30 本（GRU は下の節）を準備。同日 19:36 に math へ 210 本を投入（木 v27、ユーザー「3seedのは全部優先で、全部投下OK」「早く終わるタスクから優先で！」）。seed 10–12 の 126 本は優先ブロック（SIL の tagr・Car-Flag 長時間版・KAN の seed 10–12）の直後、seed 13–14 の 84 本は最後尾。どちらも課題順は 1 本が短い順の mirror2 → Car-Flag → tagr、その中は seed 順、腕は light 12 腕 → `gru_ppo` → `gru_full`。math 側の投入は `run1102/add_popx.py`、並べ替えは `run1102/reorder_popx.py`。

## 結果の解説

未取得。完了後、`scripts/popx_analyze.py 'results/campaigns/popx-*'` の出力（除去比較 = `no_x − full`、単独比較 = `x − ppo`、同じ seed の対応差と bootstrap 区間、F の採用履歴、失敗規則の判定）をここに書く。

## 結論メモ

- （結果待ち）

## GRU の追加比較

同じ64環境の `gru_ppo`（全off）と `gru_full`（A1/A2/QD/C/D/F=BC、推論optimizer）を追加。
観測encoder・GRU・condition射影・価値／予測ヘッドは共有、拡散actorだけ16個体にする。
幅は `trysvgd_<task>_gru.json` を使用。3課題×2腕×5seed=30本、6config。
予測pretrain98updateを含む333/333/1293updateで、lightと環境step予算を揃える。
投入順は別ファイル [gru.queue](../jobs/popx/gru.queue)、既存180本とscratch.queueは維持。
解析のGRU節で `gru_full−gru_ppo`、`full−gru_full`、`ppo−gru_ppo` を対応seedで比較する。
CPUのbit一致・分割再開・strict load・短縮実行は機構検証であり、性能比較の結論は本番結果待ち。
行動時の FLOPs（行列演算のみの計数、再帰 step）は GRU / light で Car-Flag 0.97 倍、mirror2 1.25 倍、tagr 1.26 倍。幅は既存の GRU 基準線のまま（揃え直していない）。actor の構造とパラメータ数は light と同じ。
実装の報告: `results/reports/popx_2026-11-13/POPX_GRU.md`（light は `POPX_REPORT.md` / `POPX_FOLLOWUP.md`）。
