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
- **v2 への作り直し（2026-11-14、ユーザー「最優先で探索エージェントの実験を入れれる？長時間かかりそうなのは後回しにしても OK」「探索エージェントのやつから v2 にして」）**: light の 12 腕を light TF-SVGD の既定 v2（[23](23-light-default-v2.md): 全時刻一様開始・観測アンカーなし・非有界）の上に作り直した。campaign `popx2-<task>-<arm>-v1`、`../config/popx2_scratch_<task>_<arm>.json`、`../jobs/popx/popx2_scratch_<task>_<arm>_r{10..14}.sh`、生成 `scripts/popx2_generate.py`、対応表 `jobs/popx/manifest_v2.json`。v1 との差は v2 の 5 キーだけ。GRU の 2 腕は信念を持たないので `popx-<task>-gru-{ppo,full}-v1` をそのまま使う。v1 の light 180 本（`popx-<task>-<arm>-v1`）は 2026-11-14 05:32 にキューの最後尾へ移した（ユーザー「v1 は全部の最後尾に置いといて」）。2026-11-14 05:06 に math のキューを差し替え（木 v30）: 待ちの先頭は既定 v2 の短い 9 本（tps / mirror2 / Car-Flag、seed 10–12）、その直後に集団探索の mirror2 と Car-Flag の seed 10–12（84 本）、既定 v2 の tagr 3 本、以降は従来の順（集団探索の tagr はその中、seed 13–14 は後ろ）。math 側は `run1102/popx_top.py`。解析は `popx2-*` と `popx-*-gru-*` を合わせて読む。
- **状態**: 2026-11-13、実装・CPU 検証（短縮実行は配線確認のみ）・math での速度計測・設定 36 + 6 個・ジョブ 180 + 30 本（GRU は下の節）を準備。同日 19:36 に math へ 210 本を投入（木 v27、ユーザー「3seedのは全部優先で、全部投下OK」「早く終わるタスクから優先で！」）。seed 10–12 の 126 本は優先ブロック（SIL の tagr・Car-Flag 長時間版・KAN の seed 10–12）の直後、seed 13–14 の 84 本は最後尾。どちらも課題順は 1 本が短い順の mirror2 → Car-Flag → tagr、その中は seed 順、腕は light 12 腕 → `gru_ppo` → `gru_full`。math 側の投入は `run1102/add_popx.py`、並べ替えは `run1102/reorder_popx.py`。

## 結果の解説

2026-11-15 時点。mirror2 は light v2 の 12 腕 × seed 10–12 と GRU 2 腕が全部完走、Car-Flag は seed 10–11 が完走で seed 12 の一部が走行中（暫定）、tagr は未着手。数値は学習器内の決定的評価（最終 update、推論に使う actor）で、全 run 共通の検証 100 話ではない。解析は `scripts/popx_analyze.py`（math の `run1102/popx_analysis/popx2.md`、84 run）。

| 腕 | mirror2（seed 10 / 11 / 12） | 平均 | `ppo` との対応差 [95 % 区間] | Car-Flag（暫定） |
|---|---|---:|---|---:|
| `ppo`（対照、全 off） | −329 / −309 / −305 | **−314** | — | −0.35 |
| `a2`（OU ノイズだけ） | −333 / −304 / −322 | −320 | −5 [−17, +5] | −0.49 |
| `d`（RND だけ） | −329 / −312 / −381 | −341 | −26 [−76, 0] | −0.36 |
| `pbt` | −859 / −383 / −308 | −516 | −202 [−530, −3] | −0.44 |
| `full`（全部入り） | −1185 / −359 / −391 | −645 | −330 [−855, −50] | −0.58 |
| `no_f` | −839 / −814 / −341 | −665 | （full 比 −20 [−455, +346]） | −0.54 |
| `no_qd`（QD → CEM） | −1490 / −589 / −823 | −967 | （full 比 −322） | −0.43 |
| `no_a2` | −1208 / −1119 / −578 | −968 | （full 比 −323） | −0.50 |
| `no_d` | −1249 / −1470 / −431 | −1050 | （full 比 −405） | −0.43 |
| `no_c` | −1675 / −1840 / −542 | −1353 | （full 比 −708） | −0.42 |
| `es` | −1718 / −1887 / −1781 | −1795 | −1481 | −0.43 |
| `cem` | −1909 / −1815 / −1780 | −1834 | −1520 | −0.95 |
| GRU `ppo` | −782 / −145 / −156 | −361 | — | −0.44 |
| GRU `full` | −542 / −542 / −362 | −482 | GRU `ppo` 比 −121 [−397, +239] | −0.46 |

- **対照を上回った部品は無い**。時間相関のある行動ノイズ（`a2`）と内発報酬（`d`）は単独では対照と同じ帯、集団の更新則（PBT / CEM / ES / QD）を入れると悪くなり、CEM と ES は方策を壊した（−1800 は崩壊した run と同じ水準）。
- **全部入りの中では各部品が「効いて」見える**（1 つ外すと 320〜710 悪化）が、全部入り自体が対照より 330 悪いので、これは「悪い仕組みの中での相対的な寄与」であって部品の有用性ではない。最適化専用エージェント F だけは外しても変わらない（−20 [−455, +346]）。
- **失敗の主因は選抜が雑音で決まっていること**。集団を持つ 54 run のうち 53 run で「個体の評価の雑音 ≥ 個体間の差」（mirror2: 雑音 115〜289 に対して個体間の標準偏差 4〜165。同じ個体を 2 回評価したときの順位相関は 0.0〜0.2）。ES は個体間の差が 4〜6 しかなく、初期摂動 0.01 では個体が区別できていない。
- **F の取り込み条件は厳しすぎる**。F を持つ 30 run すべてで「探索個体のほうが良いのに取り込まれない」update があり（28 run で 60 回以上、最大 182 回）、取り込み回数は 0〜23 回。取り込んだ直後に PPO が元へ戻した疑いは 6 run。
- 内発報酬が外部報酬を上回る（RND が支配する）失敗と、選抜・再標本化で収益が落ちる失敗は 0 run。
- Car-Flag（暫定）でも対照が最良（−0.35）。QD を CEM に替えると全部入りより +0.16 [+0.03, +0.31] 良く、QD はここでは害。light v2 の対照は GRU の対照より +0.11 [−0.00, +0.32]。
- **副産物**: 64 環境で学習した v2 の素（`ppo` 腕）は mirror2 で 3 seed とも崩れず −314。16 環境の本体では同じ v2 が −1651 / −789 / −312（[23](23-light-default-v2.md)）。

## 結論メモ

- mirror2（seed 10–12 完走）と Car-Flag（暫定）では、どの部品も 64 環境の素の PPO を上回らなかった: `ppo` −314、`a2` −320、`d` −341、`pbt` −516、`full` −645、`es` / `cem` −1800（学習器内の評価）。
- 主因は選抜の雑音: 集団を持つ 54 run 中 53 run で個体の評価の雑音が個体間の差以上、再評価の順位相関 0.0〜0.2。F の取り込みは条件が厳しすぎて 0〜23 回、F を外しても結果は同じ。
- この実験の動機（行動を 5〜20 step 続けて変えないと得にならない台地）は tagr のもので、tagr の 36 本はまだ走っていない。mirror2 / Car-Flag の結果からは、個体の評価を共通の環境 seed・多めの話数に直さない限り、tagr でも選抜は雑音で決まる見込み。

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
