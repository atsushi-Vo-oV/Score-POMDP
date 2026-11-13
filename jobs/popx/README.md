# POPX: scratch 12 腕と最適化専用個体

`python -m sb_pomdp.popexplore` は通常の train/compare と独立した学習経路です。
全ジョブは生成済みで、投入はしていません。凍結群のconfig/job/queueは削除しました。

## 本番比較

|腕|集団|OU|B|探索側PPO|RND|最適化専用F|評価|
|---|---:|---|---|---|---|---|---|
|full|16|on|qd|on|on|bc|optimizer|
|no_f|16|on|qd|on|on|off|mean|
|no_a2|16|off|qd|on|on|bc|optimizer|
|no_c|16|on|qd|off|on|bc|optimizer|
|no_d|16|on|qd|on|off|bc|optimizer|
|no_qd|16|on|cem|on|on|bc|optimizer|
|cem|16|off|cem|off|off|off|mean|
|es|16|off|es|off|off|off|mean|
|pbt|16|off|pbt|on|off|off|mean|
|a2|1|on|none|通常PPO|off|off|mean|
|d|1|off|none|通常PPO|on|off|mean|
|ppo|1|off|none|通常PPO|off|off|mean|

3課題 × 12腕 × seed10–14 = 180ジョブ、36config。
全腕64環境・rollout128。集団は16個体×4環境。F onではindex0が最適化専用、探索15個体。
信念も最初から学習します。98 pretrain + 1000/200 PPO + 195/35 predict = tagr1293、mirror2/carflag333 update。
元の16環境設定のupdate数を維持するため、環境step数は元設定の4倍です。

- config: `config/popx_scratch_<task>_<arm>.json`
- job: `jobs/popx/popx_scratch_<task>_<arm>_r<seed>.sh`
- campaign: `popx-<task>-<arm>-v1`、shard001–005
- 完全対応表: [manifest.json](manifest.json)
- `scratch.queue`: seed-major、各seedでcarflag全腕→mirror2全腕→tagr全腕。各腕の順は上表。

再生成: `PYTHONPATH=src python scripts/popx_generate.py`。既存の凍結生成物もこのコマンドで削除します。

## mathでの起動

```bash
export SB_POMDP_PYTHON=python3.11
bash jobs/popx/run_popx.sh /path/to/popx-tree jobs/popx/popx_scratch_carflag_full_r10.sh 0
```

runnerはTREE/JOB/GPU、stdout統合、学習exit codeを維持します。入力不備2、flock競合75。
`SB_POMDP_SEGMENT_UPDATES=10`で分割再開できます。partial成功もexit0なので、完了まで同じ行を再投入してください。
`launch.sh`は`.sb_overrides`を読みません。既定はautograd/graphs=false、`SB_POMDP_POPX_FAST=1`でanalytic/graphs=true、どちらもcompile=offを末尾で明示します。
専用の`SB_POMDP_POPX_OVERRIDES`で1行`dotted.key=JSON`のoverrideを渡せます。

mathのscheduler2実物はこのtreeにありません。固定runnerで引数を解釈した後、旧compare引数の構築前に次の分岐を入れる案です。

```bash
if grep -q '^export SB_POMDP_ENTRY=popexplore$' "$tree/$job"; then
  exec bash "$tree/jobs/popx/run_popx.sh" "$tree" "$job" "$gpu"
fi
```

## Fの設定と学習

`popexplore.optimizer_agent`は閉じたobjectです。既定はoff。`enabled=true`かつE未指定なら`inference_actor=optimizer`。

|キー|既定|意味|
|---|---:|---|
|enabled|false|index0を最適化専用にする|
|adopt|bc|copy / bc / none|
|window|5|直近M PPO updateの対応差。欠測・候補の置換で履歴を消す|
|margin|0.1|外部episode収益の平均差の最小幅（課題依存で調整可能）|
|confidence|0.95|片側Student t、候補数N−1のBonferroni補正|
|bc_steps|10|直近の候補rolloutのcondition/実行行動でnoise予測損失をK回学習|
|bc_learning_rate|0.0003|一時的なBC Adamの学習率|
|pull_to_optimizer|0|選抜後のCEM/ES/QD平均をFへ寄せる係数。次世代以降に反映|

Fには初期重み摂動、選抜置換、OU、内発報酬を掛けません。
pretrainでもFの行動は通常の拡散サンプルで、探索側だけを一様ランダム行動にします。
PPOフェーズではCのfraction/scaleに関係なく自分の4環境だけで外部報酬PPOを実施。
予測pretrain/交互predictフェーズは予測学習を維持し、actor PPOは実施しません。
共有のbelief/readoutは全個体から学びます。F用に`optimizer_value_head`を追加し、外部専用のGAE/returnを学習します。
探索側の既存valueは外部＋内発を学習。advantage正規化もF/探索に分離。
終端打切りbootstrapにもF専用valueを用います。copy/BC時にはcriticをコピーしません。

各PPO updateで現在の全候補をOUなし・外部報酬だけで評価し、同じupdateの探索i−Fの差を記録します。
M個揃い、平均差>marginかつ下側限界>0なら適格候補のうち平均差最大の個体を採用。
通常のBは5 PPOごとなので、既定M=5は置換前に判定を1回行える最小限の平滑化です。
M>selection_intervalでは毎回置換されるCEM/ES/QD候補は窓が揃いません。
PPOで動く同一系統を比較するため、相関・非定常性・反復検定に対する有意性保証ではありません。
M=5、margin=.1は初期設定で、5 seedの本番結果で妥当性を確認してください。

copyは候補actorを複製、BCはFのactorだけを更新。どちらもFのPPO Adam moment/stepを初期化。
BC入力は保存済みconditionと実行行動をdetachし、共有信念を変更しません。
取り込みは現在の探索個体のみ。平均・archiveを取り込み候補にする拡張はありません。
学習actorはepisode開始snapshotを終了まで保持するため、採用の行動への反映は各環境の次のresetからです。
推論・評価・latest exportには即座にFの現在重みが使われます。

## fitness費用

`popexplore.fitness_from_rollout`を実験用に追加しましたが、既定はfalseです。
4環境×128 stepでもtagr horizon200/carflag160はepisodeがupdateを跨ぎ、OU付き行動と古いactor snapshot、学習中のbeliefの収益が混ざります。
これを現在のOUなし候補fitnessと見なすと、Fとの比較条件も異なります。
trueでは完了episodeの外部収益を対応差・選抜に使用し、追加fitness実行は一切行いません。欠測は0扱いせず、Bは全探索個体の値が揃わなければskip、Fは対応差の欠測で履歴reset。
population.csvに`rollout_behavior`と記録。QD archiveの再評価も省略するため保存scoreは古いbeliefでの値を含みます。
採用先の現actorと収益を生んだsnapshotの差も含む近似です。採用直前直後・PPO直前直後の成績は未測定のNAとします。
定期評価は通常どおり行い、安価な機構のON/OFF比較用としてのみ使ってください。
既定falseではBは従来の固定方策評価、Fは毎PPO更新の対応評価を使用。archiveも現在のbeliefで再評価するため費用は大きくなります。
FのPPO前/後、採用前/後は単一actor評価（全話同時のbatch評価）の同一seed比較で、採用するupdateとその次のPPO updateだけ記録します（それ以外はNaN。毎updateのFの成績は`optimizer_fitness_return`）。

## 出力と再開

通常のmetrics/evaluation/episodesに、population.csv、population_evaluation.csv、timing.csvを追加します。
metrics/populationの`optimizer_adoptions/adopted/adoption_update/candidate/difference/lower_bound`で回数・時点・差を追えます。
metricsはpair_count、BC損失、F fitness、探索最良、採用前後、PPO前後差も持ちます。
population各行のFイベント列は同じupdateのイベントを反復記録するので、行方向に合算しないでください。

`latest.pt`は通常loaderでstrict loadできる推論export。E=optimizerではF用valueを通常value_headへ置きます。
`population.pt`が再開正本。追加value、全actor/optimizer、OU/RND、分布/archive、採用履歴・回数、episode snapshot、環境履歴、乱数を保存。
ソース・config・task・seedの一致を確認し、未commit CSV行は再開updateまで切り戻します。
`timing.csv`はCUDA同期付きでrollout開始から学習・fitness・選抜・定期評価・checkpoint保存終了までを計測します。
構築/再開ロードと最終summary出力は含みません。旧`steps_per_second`を総時間の代用にしないでください。

## 解析と速度計測

```bash
PYTHONPATH=src python scripts/popx_analyze.py 'results/campaigns/popx-*' --out analysis/popx.md
bash scripts/popx_bench.sh --output-root results/popx-bench-stress --warmup 5 --repeats 3
bash scripts/popx_bench.sh --output-root results/popx-bench-production --cadence production --warmup 5 --repeats 25
```

解析は5除去腕−full、単独腕/full−ppo、対応seed bootstrap、課題別・課題横断の要約、F時系列、明示的な切り分け規則を出力します。
途中runも解析可能、評価未到達は欠測。各ペアの最新共通updateで比較します。
横断差は課題ごとΔ/max(1,abs(参照平均))に規格化し、共通seedを課題ベクトルごとbootstrap。
探索個体最大には上方選択バイアスがあります。区間やヒューリスティックだけで原因・有効性を断定しません。

benchはGPU必須。tagr/mirror2 × ppo/cem/fullを連続実行し、warmup後の各update秒を`benchmark.json`へ出力します。
既定stressは選抜・定期評価・checkpointを毎updateに増やした負荷測定です。本番平均秒ではありません。
productionは元の間隔を保ち、25以上のPPO updateを測って周期評価を含めます。
どちらも初期pretrainを省略し、beliefを更新するPPO区間を測定します。交互predict時間はtiming.csvに残りますがPPO平均に含めません。通常trainerの終了時評価・checkpointも最後の試料に含まれます。
stressでは選抜間隔1<M=5のためBC採用は発生せず、その費用を実測したと見なしません。
本番総時間はpretrain/PPO/predictをそれぞれ測定し、回数を掛けて計画してください。
CUDA同期、モデル・ハード情報、各試料、解決済みconfigをJSONに保存。出力先は毎回新しく指定。
`--cpu-small`は配線確認専用でGPU速度予測には使えません。
`--fitness-from-rollout`で追加fitnessなしの比較計測もできます（上記の近似と欠測によるskipを含む）。

CPU短縮例: `PYTHONPATH=src python scripts/popx_followup_smoke.py --output-root /new/output`。
full/ppo/no_f、3課題、2対応seed、1pretrain+8PPO+3predict、縮小モデルで信念も学習します。

対応範囲はdirect/autograd score、SVGD、transformer(dropout0)、continuous diffusion、state/shared encoder、condition prediction。
compile、activation checkpoint、SIL/curl、別value encoder等は従来どおり明示拒否です。
analytic score と CUDA Graphs は共有beliefの内部だけで働く実行時の切替なので許可しています（`SB_POMDP_POPX_FAST=1`でlaunch.shがanalytic + graphsを指定、既定はautograd）。
