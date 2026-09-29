**検証結果：表の数値集計は正しい。ただし、現状の実験から導いている結論には修正が必要。特にLight-Dark 2Dの補助損失と参照方策が問題になる。**

対象は [results-comparison-2026-09.tex](../docs/results-comparison-2026-09.tex)。最新の外部向け改稿 `bbb6a14` も確認した。本文のDesign A/B/C/Dは設定・Markdownのv1/v2/v3/v4に対応する。

Genkaiから2026-09-14 **08:39:10–08:40:54 JST** に取得した、評価CSV・学習指標・resolved config・manifest・実行ソースを用いた。取得中も一部runは進行している。[監査対象TeXのコピー](../results/audit-20260914/reviewed-results-comparison.tex)、[元snapshot](../results/audit-20260914/snapshot.json)、[数値照合結果](../results/audit-20260914/table_checks.json)を保存した。学習コード、元TeX、実行中のジョブはこの監査では変更していない。

**確認できたこと。**

- 表のseed別数値 **286個** は、元CSVの該当評価または連続4評価の平均を整数に丸めた値とすべて一致した。進行中を示す `*` の数値も、そのrunの履歴に存在する。
- 平均・SD **48組** は、文書が完了扱いしているseedだけを使うとすべて一致した。SDは学習seed間の標本標準偏差（`ddof=1`）で、エピソード間SDや信頼区間ではない。
- 対象となる実験は重複を除き **193 runs**。取得時点のmanifestは完了157、running35、failed1だった。文書作成後にLD2Dのparameter-matched Bの3 seedsが完了しているため、文書の `*` の一部は現在より古い。
- 対象CSVに同一update/modeの重複はなく、評価stepは `update × 2,048` と一致。完了runには200-updateで両評価モードが存在した。16環境、128 steps、409,600 steps、10評価episodes、symlog価値ターゲット、通常PPO、truncation bootstrapなしという基本条件を確認した。
- 8実行treeすべてで取得ソースのSHA-256が、そのtreeのrun metadataに記録された実行ソースhashと一致した。全treeが同一バージョンという意味ではない。LD2Dの実行ソースは現在のローカル学習ソースとも一致する。
- 現在のコードの全既存テストは **446 passed / 1 failed / 4 skipped**。失敗は [test_genkai_submission.py:150](../src/tests/test_genkai_submission.py#L150) の旧「shard 20以下だけresume可能」という文字列検査。コミット `40f76f7` がその制限を削除したのに、テストが追従していない。4件のskipはWindowsでbashがないため。これを学習計算の失敗とは解釈しないが、全テスト成功とも報告できない。

**1. 重要：Light-Darkの報酬予測補助損失が、位置に関する主要報酬をすべて除外している。**

[ppo.py:103](../src/sb_pomdp/ppo.py#L103) の `reward_prediction_loss` は、行動を次時刻の `previous_actions` から取得し、`valid = ~dones[:-1]` で終端stepを除外する。一方、この実験のLight-Dark設定は `state_cost=0`, `action_cost=0.001`, `terminal_cost=10`。状態に直接依存するコストは、30歩目の `-10 ||x_30||²` だけである。[envs.py:811](../src/sb_pomdp/envs.py#L811)

したがってB/Cの報酬予測器が見るターゲットは、基本的に **`-0.001 ||a_t||²`だけ**。行動を条件として与えている以上、この値を予測するために位置・位置の不確実性を粒子へ記憶する必要がない。次観測予測の補助損失まで無効という意味ではない。

実際にこの環境で31 stepsのrolloutを作り、30歩目の報酬 **−139.22349を−1,000,000へ置き換えても、報酬予測損失と予測器への勾配が完全に同じ**ことを確認した。[再現コード](../results/audit-20260914/local_checks.py)、[出力](../results/audit-20260914/local_checks.json)

これは補助損失ノートの既存マスク定義には従っている。しかし「タスクの報酬予測によってbeliefへ有用な状態情報を学ばせる」というLight-Darkの実験意図を満たしていない。報酬は終端stepでも観測済みなので、次エピソードの観測を除外するマスクをそのまま報酬へ適用する必要はない。

修正対照では、各stepで実際に実行した行動を使って終端報酬も学習対象に含める。次観測予測用のマスクとは分ける。学習目的が変わるため、B/Cは新しい実験条件として再実行し、現在の結果と混ぜない。

**2. 重要：Light-Darkでは、観測を使わない簡単な参照方策が掲載手法より良い。**

初期位置の平均は `(2,2)`。そこで「最初の2歩は `(-1,-1)`、残り28歩は `(0,0)`」という、観測・真の状態を一切参照しない方策を評価した。初期分布と力学の既知定数を使うスクリプトであり、学習手法との同一情報条件を主張するものではない。課題の達成水準と、観測利用の有用性を判断する参照になる。

| 同じ評価seed集合での成績 | 平均return |
|---|---:|
| 観測を使わない2歩移動方策 | **−16.46** |
| DeepSets A | −60.18 |
| DeepSets B | −60.71 |
| GRU | −68.35 |
| RNN | −68.13 |
| PF（成功した4 seedsのみ） | −59.18 |

2歩移動方策のseed別平均は `−18.397, −15.168, −15.791, −15.440, −17.508`。別の2,000独立環境seedでも平均 **−19.82**（episode SD 19.68）だった。微小なgoal半径への途中到達を無視すれば、期待returnは解析的にも

`−10 × 2 × (1 + 30 × 0.01²) − 0.001 × 2 × 2 = −20.064`

となる。[検証出力](../results/audit-20260914/local_checks.json)

したがって「DeepSetsはGRU/PFと同程度」という記述的比較は成立しても、これだけでbelief推定が役立つことや、課題を十分に解けていることは示せない。この参照値を追加し、学習方策が観測をどう使っているかを評価する必要がある。これを根拠に情報収集が原理的に不要とも断定しない。

**3. 重要：本文のseed効果に関する数値・因果解釈が成立していない。**

最新版TeXの `Seed effects` 段落は「各seedの手法間平均の差は両課題で10未満なので、列間差は評価難易度ではなく学習変動」とする。しかし、未完了・legacy・欠測PFを除き、すべてのseedに存在する同じ14手法で計算すると次になる。

| 表 | s10 | s11 | s12 | s13 | s14 | 最大−最小 |
|---|---:|---:|---:|---:|---:|---:|
| Mountain Hike @200 | −273.87 | −273.10 | −290.76 | −326.30 | −271.48 | **54.82** |
| Light-Dark @200 | −59.62 | −202.71 | −156.11 | −115.92 | −142.99 | **143.09** |
| Light-Dark 末尾4評価 | −182.61 | −159.95 | −126.89 | −182.67 | −201.45 | **74.55** |

さらに、学習seedと評価集合を同時に切り替えているため、手法間平均が近かったとしても、学習変動と評価難易度を分離した根拠にはならない。固定した複数の学習済み方策を、共通の複数評価集合で交差評価する必要がある。

評価環境seedは `training_seed + 1,000,000 + episode`。10〜14の5学習seed、各10 episodesでも、**異なる環境seedは14個しかなく、隣接seedで9/10を共有**する。学習seedが5つある事実とは区別すべきで、50個の独立評価ケースとみなせない。固定テスト集合に条件付けた学習変動と、新しい環境episodeに対する不確実性を分けて報告する。

**4. 「変動は主にsampling artefact」という断定を支持する検証がない。**

[train.py:842](../src/sb_pomdp/train.py#L842) は毎評価で専用generatorを同じseedに初期化する。環境seedも毎評価同じ。このLight-Darkの保存評価は30 stepsなので、通常、各評価で新しい独立な乱数集合へ交換されるわけではない。

例えばAlpha A seed 10のmean-chain評価は更新125/150/175/200で `−305.45 / −95.32 / −503.71 / −157.99`。ここではcheckpointのパラメータも変わっている。有限10 episodesによる推定誤差、パラメータ変化への感度、学習の不安定さを、この時系列だけで分離できない。

保存された同runの200-update checkpointをCPUへ読み込み、同じ評価seedで繰り返すと **−159.87865が2回一致し、全episode returnも一致**した。CPUとCUDAでは乱数列が異なるため、これはGPUログの数値を完全再現したという主張ではない。固定checkpointに対する再評価の結果とコードの乱数規約を確認したもの。[再評価コード](../results/audit-20260914/reevaluate.py)、[結果](../results/audit-20260914/reevaluation.json)

同じcheckpointを新しい200環境episodesで再評価すると **−147.95**（episode SD **77.25**）だった。これは当該checkpointの評価サンプルによる変動の一例であり、5学習seed全体の信頼区間でも、更新間変動の原因分解でもない。

末尾4評価の平均は4種類の方策の成績を平均した指標で、固定された最終方策を40 episodesで測ったものではない。本文は「評価値はcheckpoint間で不安定。原因は未分離」とするのが妥当。

**5. 同一学習量でない数値が、同じ末尾4評価表に入っている。**

Table 3のcaptionは「updates 125–200」と説明するが、parameter-matchedの `*` は別の窓である。元CSVと一致する窓は以下の通り。

- Aの5 seeds：75/100/125/150。
- Bのs10/s12/s14：100/125/150/175。s11/s13だけ125/150/175/200。
- Cのs10/s11：75/100/125/150。s12/s13/s14：50/75/100/125。
- Dの5 seeds：100/125/150/175。

数値の計算自体は合っているが、captionは正確でない。各セルにupdate窓を付けるか、指定窓の揃ったrunだけで表を作る。途中結果から「parameter matchingは役に立たない」と確定することも避ける。

**6. 「同程度」は行動の決定方法に強く依存する。**

元CSVにはmean-chainとstochasticの両方がある。文書がmean-chainだけを掲載すること自体は、事前に定めた評価対象なら可能。ただし、これは各逆拡散stepの平均を通した行動の評価であり、学習した確率方策から行動をsampleした成績とは異なる。

| Light-Dark @200 | mean-chain：平均 ± seed SD | stochastic：平均 ± seed SD |
|---|---:|---:|
| DeepSets A | −60.18 ± 23.45 | −131.61 ± 37.35 |
| DeepSets B | −60.71 ± 17.82 | −241.24 ± 93.38 |
| GRU | −68.35 ± 26.87 | −76.56 ± 30.79 |
| RNN | −68.13 ± 16.09 | −87.80 ± 31.36 |
| PF（成功した4 seeds） | −59.18 ± 8.74 | −103.22 ± 45.20 |

「DeepSets Bがbaselineと同程度」を、確率方策一般へ広げることはできない。また、mean-chainでもbelief粒子の初期化・ULAの乱数は残る。baselineのDPFRL型PFも内部に乱数を持つ。[全手法の両モード再集計](../results/audit-20260914/recomputed-tables.md)

PFの平均は成功4/5条件付きであり、失敗率20%を併記する。任意の失敗returnを勝手に補完すべきではない。現在の平均±SDは記述統計として正しいが、手法の同等性や優位性を示す検定にはなっていない。評価対象を先に固定し、学習seed・評価episodeの両方の変動を扱う区間推定が必要になる。[Agarwal et al., NeurIPS 2021](https://arxiv.org/abs/2108.13264)

**7. 再現性のために直すべき記述。**

- 記載パラメータ数は主にAの値。TransformerのA/B/C/Dは **75,286 / 76,768 / 76,316 / 75,738**、通常Alphaは **35,525 / 37,007 / 36,555 / 35,977**、matched Alphaは **75,269 / 76,751 / 76,299 / 75,721**、DeepSetsは **67,222 / 68,704 / 68,252 / 67,674**。追加予測器と初期proposalを含む総数をdesign別に記載する。matched Alphaと同design Transformerの近い容量は確認できる。
- DeepSetsの実装はsum poolingではなく **mean pooling**。[networks.py:146](../src/sb_pomdp/networks.py#L146)
- 温度1のAlpha headはlog-sum-expによる滑らかな凸関数で、厳密な区分線形関数ではない。温度を0へ近づけたmaxの極限と区別する。[networks.py:765](../src/sb_pomdp/networks.py#L765)
- Light-Darkには `process_noise_std=0.01` があるため、記載の `x'=x+u` だけでは設定が再現できない。両座標の初期平均2、標準偏差1、行動範囲±1、観測分散 `5(5−x_1)²+0.01²`、行動コスト0.001、終端コストの二乗ノルム、goal半径10^-6も明記する。
- Mountain Hike本文のAlpha平均「−258〜−281」はBの **−229.25** を漏らしている。
- 有限horizonで終端コストを持つこの設定について、`bootstrap_on_truncation=false` という事実だけを実装バグとは判定しない。報酬・終端規約と併せて明記する。

**この結果で支持できる範囲。**

Mountain Hikeでは、掲載された完了runのmean-chain平均において、通常のscore各designがGRU/RNN/DPFRL型PFより劣る、という記述は支持される。Light-Darkでは、mean-chain最終値に限ればDeepSets A/Bがbaselineに近いという観察は支持される。真のBayesian beliefの正しさ、情報収集の有効性、手法の同等性、parameter matchingの最終的な効果までは立証されていない。

優先順位は、**終端報酬を含む補助損失の修正対照、観測を使わない参照の追加、固定checkpointを十分な共通テストepisodeで両評価モード再評価、本文のseed解釈・評価窓・sampling artefactの訂正**。全学習193 runsの再実行はこの監査では行っていない。

再現用コマンド（リポジトリルート、既存Python環境）:

```powershell
.venv\Scripts\python.exe results\audit-20260914\check_tables.py
.venv\Scripts\python.exe results\audit-20260914\local_checks.py
.venv\Scripts\python.exe results\audit-20260914\reevaluate.py
.venv\Scripts\python.exe -m pytest -q
```
