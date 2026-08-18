# Score-Belief POMDP comparison experiments

[`docs/sb-pomdp-retry.tex`](docs/sb-pomdp-retry.tex) の提案法と、同じ POMDP・PPO 条件で比較する
baseline を実装しています。観測 energy と遷移 energy から belief score を作り、ULA
（Langevin dynamics）で得た粒子集合を方策へ渡します。比較実験の共通入口は
`sb-pomdp-compare` です。

## タスク

外部の Gym には依存せず、力学は `src/sb_pomdp/envs.py` に実装しています。通常の方策が
受け取るのは部分観測だけです。真の状態は評価軌跡へ診断用に保存し、`oracle_state` だけが
比較用入力として利用します。

| タスク | 隠れ状態 | 部分観測 | 行動 |
|---|---|---|---|
| `masked_cartpole` | 位置・速度・角度・角速度 | 正規化した位置と角度 | `Discrete(2)` |
| `masked_pendulum` | 角度・角速度 | `cos(theta), sin(theta)` | `Box(1)`, `[-2, 2]` |
| `masked_mountain_car_continuous` | 位置・速度 | 正規化した位置 | `Box(1)`, `[-1, 1]` |
| `light_dark` | (N) 次元位置 | 第1座標の明るさに応じた雑音付き位置 | `Box(N)`, `[-1, 1]^N` |

`light_dark` は `dimension` で次元を指定します。原典の2次元問題を
(x_{t+1}=x_t+u_t) のまま (N) 次元へ拡張し、観測標準偏差を
`sqrt(0.5 * (5 - x[0])^2 + observation_noise_std^2)` とします。local は3次元、
production比較は5次元です。

`config/production.json` の `observation_noise_std` は、`masked_mountain_car_continuous` だけ
`0.01`、他3タスクは `0.05` です。**これは意図的なタスク別設定です。** MountainCarの観測は正規化位置
1次元だけで、隠れ状態の速度は連続する観測の差分からしか推定できません。正規化スケールでの1 step変位は
最大でも `0.078`（`|v| <= 0.07`、正規化係数 `2/1.8`）です。`0.05` では連続2観測の差分の雑音標準偏差が
`0.05 * sqrt(2) = 0.071` と最大変位に匹敵し、典型的な変位はそれよりはるかに小さいため、速度の推定が
実質的に成立しません。`0.01` なら差分の雑音標準偏差は `0.014` で信号が残ります。この値を他タスクへ
揃えるとタスクの難易度定義自体が変わるため、MountainCarの全runを再実行するまで既存結果と
混在させられません。

### `light_dark` の報酬

コストは遷移**前**の状態と行動に課金し、goal bonusは終了stepにだけ1回加算します。

```
r_t = -(state_cost * ||x_t||^2 + action_cost * ||u_t||^2)
      + goal_bonus * 1{ ||x_{t+1}|| <= goal_radius }
```

`||x_{t+1}|| <= goal_radius` でepisodeをterminateします（horizon到達はtruncation扱いで
bonusは付きません）。次の7キーは `environment.tasks` の `light_dark` エントリに書ける
任意キーで、既定値は環境実装のハードコード値と同一です。

| キー | 既定値 | 意味 |
|---|---|---|
| `state_cost` | `0.5` | 遷移前状態の二次コスト係数 |
| `action_cost` | `0.5` | 行動の二次コスト係数 |
| `goal_radius` | `0.25` | 終了判定の半径 |
| `goal_bonus` | `0.0` | 終了stepに加算する報酬（新規） |
| `light_position` | `5.0` | 明るい超平面 `x[0]` の位置 |
| `initial_mean` | `2.0` | 初期状態の平均（全座標） |
| `initial_std` | `0.5` | 初期状態の標準偏差 |

キーを書かない設定ファイルはproduction campaignと完全に同一の報酬になります。
`resolved_config.json` のlegacy正規化も同じ既定値を補完するので、旧成果物の
resume・集計の一致判定はこれらのキーを追加しても壊れません。

既定の `goal_bonus = 0.0` では報酬がコスト打ち切りだけになるため、production条件
（`N=5`、light at `x[0]=5`、初期 `N(2*1, 0.5^2 I)`、horizon 30）では光へ寄る情報収集が
厳密劣後します。実測（真のenv・Kalman filter方策・3,000 episodes）では、dead reckoningが
平均return `-31.8`・goal到達率 `0.03` に対し、光で自己位置同定してから原点へ戻る方策は
`-87.9`・到達率 `0.90` でした。監査時の独立実測（情報収集方策の変種が異なる）でも
`-31.7` 対 `-77.0` と大小関係は同じで、この結論は方策変種に依存しません。
機構としては、力学が決定論的な `x' = x + u` なので事後平均は観測を使わずとも行動履歴の和だけで
追跡でき、観測の価値は不確実性の縮小＝goal到達確率にしか現れません。既定の `goal_bonus = 0.0`
はその到達確率に報酬を与えないため、belief品質がreturnへほとんど反映されません。
したがって既定条件の `light_dark` はbelief品質をほとんど弁別せず、
**既定パラメータで得たLight-Darkの結果を、belief表現の品質に関する証拠として読んではいけません**
（実行経路・数値安定性・連続5次元行動の確認としては有効です）。
情報収集を最適にするには、損益分岐の `65`〜`85` より十分大きい `goal_bonus`
（目安 `100` 以上、`100` で情報収集方策が `+1.8` 対 dead reckoning `-29.0`）を設定します。
ただしこれは報酬定義そのものの変更なので、既定値のまま実行した既存campaignの結果とは
混在させられません。

## 実装済み6手法と現在の2手法比較

| method | 入力・記憶 | 行動方策 |
|---|---|---|
| `score_transformer` | score 粒子 + 位置 encoding のない Transformer。提案法 | CartPole は categorical、連続3タスクは diffusion |
| `score_deepsets` | 同じ score 粒子 + Deep Sets。encoder ablation | categorical / diffusion |
| `score_gaussian` | score 粒子 + Transformer | tanh-squashed Gaussian。連続3タスクのみ |
| `observation_mlp` | 現在の部分観測だけを MLP へ入力 | categorical / tanh-squashed Gaussian |
| `gru` | 部分観測と正規化した直前行動。episode prefixを再生するGRU | categorical / diffusion（active比較では提案法と同一head） |
| `oracle_state` | simulator の正規化した完全状態を MLP へ入力 | categorical / tanh-squashed Gaussian |

6手法の実装は残していますが、現在のproduction比較は `score_transformer` と `gru` の2手法に
絞っています。4タスク、belief勾配の `full` / `tbptt_1`、各5 seedsを実行するため、行列は
`2 methods x 4 tasks x 2 modes x 5 seeds = 80 runs`です。GRUにも時間方向の勾配modeを実装し、
`full`では未終了episodeのprefixをrollout境界越しに再生し、`tbptt_1`では各環境stepの入力hiddenを
detachします。したがって両modeは同じforward値を持ちますが、同一設定の反復ではなく時間方向の
gradient範囲を比較する条件です。`score_gaussian` は離散CartPoleでは提案法との差が
消えるため、将来6手法比較を再有効化する場合も連続3タスクだけが対象です。

## ローカル比較

Python 3.11 と [uv](https://docs.astral.sh/uv/) を使用します。

```powershell
uv sync --extra dev
uv run pytest
uv run sb-pomdp-compare --config config/local.json
```

`config/local.json` は、2勾配条件 x 23組を seed 0、各1 PPO updateだけ動かす CPU smoke testです。コードと
成果物の検証用であり、学習性能の比較には使えません。production 行列を GPU なしで検証するには
次を実行します。

```powershell
uv run sb-pomdp-compare --config config/production.json --validate-only
```

手法・タスク・seedを絞る例です。

```powershell
uv run sb-pomdp-compare `
  --config config/local.json `
  --gradient-modes full tbptt_1 `
  --methods score_transformer gru `
  --tasks masked_cartpole `
  --seeds 0 `
  --max-updates 1 `
  --label trial
```

`--methods` / `--tasks` と同様に `--seeds` も config の `experiment.seeds` と照合し、
宣言されていないseedは有効なseed一覧を添えて実行前に拒否します。campaignのcoverage行列に
含まれないseedで検証したい場合は、`--override "experiment.seeds=[0]"` のように意図を明示します
（比較driverは選択seedで `experiment.seeds` を書き直すため、`resolved_config.json` は変わりません）。

設定は閉じた schema で検証します。未知のキー、不正な device、非有限値、重複 seed、rollout・
minibatch・sequence microbatch の不整合などは学習前に失敗します。
`ppo.minibatch_size` は1回の optimizer stepに対応する有効PPO minibatch、
`ppo.sequence_microbatch_size` はその内部で一度にbelief sequenceを再生してbackwardする大きさです。
各microbatchの勾配を蓄積してから1回だけgradient clippingとoptimizer stepを行うため、有効minibatchは
変わりません。どちらも `ppo.rollout_steps` の倍数とし、有効minibatchはsequence microbatchで
割り切れる必要があります。`sb-pomdp-run` は提案法だけを実行する低水準入口として残していますが、
比較結果を作るときは `sb-pomdp-compare` を使います。

## 公平性と評価規約

- 全手法でタスク、train/evaluation seed、environment-step budget、PPO の GAE・clip・optimizer、
  終了処理、評価 episode 数を共有します。
- 比較条件ではentropy bonusを無効化し、`ppo.entropy_coef=0.0`とします。実装上の
  `L_policy := -L_clip`というloss記法では、`L_total = L_policy + c_V L_value`であり、
  方策familyに依存するentropy項は比較へ入りません。
- active比較の連続3タスクでは、`score_transformer`と`gru`の両方が同じ`DiffusionPolicy`、
  reverse-step数・noise schedule・分散下限・denoising advantage重み、および各reverse Gaussian遷移を
  個別にclipするPPO目的を使います。離散CartPoleも共通のcategorical headです。したがってactive比較は
  主にbelief表現とrecurrent architectureの差を測り、action familyの差を含みません。
- **GAEのepisode終了規約**: 既定では真の終端とhorizon打ち切り（truncation）を区別せず、どちらも
  `done=1`として次状態のbootstrapを止めます（`buffer.py`の`generalised_advantage_estimate`）。
  この規約は両手法・両modeで完全に共通なので内部比較の公平性は保たれますが、絶対returnと価値学習は
  truncation-bootstrap実装と直接比較できません。投入済みcampaignはこの既定で走っています。
  opt-inの`ppo.bootstrap_on_truncation=true`を設定すると、truncationしたstepに限り
  `gamma * V(s_T)`をGAEのdelta項へ加えます（Pardoらの方式）。`s_T`はauto-reset wrapperが捨てる
  打ち切り直後の真の観測で、`SyncEnvironmentBatch`が`final_observation`として保持します。score系は
  その観測と実行した行動でbelief contextのcloneを1step進めて`V`を評価し、GRU baselineは
  cloneしたhiddenを1step進めます（どちらも`no_grad`、online状態とreplay履歴は不変）。`done`自体は
  変わらないためlambda-returnはepisode境界で切れたままで、bootstrap項だけが増えます。
  この設定はconfigの一部としてresume時にも照合されます。既定値`false`を省略したconfigは
  従来どおり動き、旧`resolved_config.json`とも等価と判定されます
  （`config.py`の`normalize_legacy_resolved_config`）。有効時の`truncation_bootstrap_value`平均は
  `metrics.csv`のclosed schemaを壊さないよう`run.log`にのみ記録します。この設定はPPOのvalue target
  そのものを変えるため、**値の異なるrunを同一の`comparison.csv`・aggregateへ混在させてはいけません**。
- **打ち切りbiasの負担が両表現で等しい保証はない**: 上の既定規約が両手法・両modeで共通であることは
  手続き上の公平性（同じ目的関数を最適化していること）を保証するだけで、「打ち切りをepisode終端と
  みなすbiasの負担が両表現で等しい」ことまでは意味しません。GRUのhiddenは制約のないrecurrent
  vectorなので、観測と直前行動のprefixから経過stepを暗黙に数え、horizon依存のvalueをfitできます。
  一方score beliefの条件は状態上の事後粒子とそのscoreの集合であり、時間channelを持ちません
  （どのタスクの部分観測にも時刻は含まれません）。したがって既定条件のvalue学習・絶対returnを
  比較する際は、この非対称性を織り込んで解釈してください。手法間の差の解釈をこの規約へ
  依存させたくない場合は、両手法とも`ppo.bootstrap_on_truncation=true`で全runを再実行します。
- GRUの`full`は観測・直前行動からなる未終了episode prefixを保存し、current parametersでepisode開始から
  再生してrollout境界越しにBPTTします。`tbptt_1`は各環境stepへ入るhiddenだけをdetachし、そのstep内の
  actor/value計算は微分します。PPO minibatchは時間を固定長chunkへ切らず、環境軸だけを分割します。
- deterministic action readout の評価名は、離散方策が `greedy`、diffusion が `mean_chain`、
  Gaussian が `mean_action` です。`mean_chain`は初期noiseを0とし、各reverse Gaussianの条件付き平均を
  順に通す規約、`mean_action`はsquashed Gaussianのpre-tanh平均を行動へ写す規約です。これらは
  return等の実験指標ではなく、評価時のaction readout名です。score系ではこの場合もbelief推論の初期粒子とULA noiseをsampleするため、
  agent全体が決定論的という意味ではありません。best checkpointはこのreadout側の平均returnで選びます。
- 評価中の乱択は、両手法とも評価seedで初期化した評価専用 `torch.Generator` から引きます。score系の
  belief粒子（初期noiseとULA noise）、diffusion / Gaussian方策のsample、離散方策のaction sampleがこれに
  含まれ、GRU baselineも同じ規約です。学習rolloutは従来どおりglobal RNGを使い、評価は前後でglobal RNG
  stateを保存・復元するため、評価の実行有無は学習の乱数列に影響しません。
- 全手法で `stochastic` も同じ held-out seedにより評価します。`comparison.csv/json` は
  belief-gradient-mode/method/taskごとに、
  deterministic readoutとstochasticの双方についてseed平均・標準偏差・提案法との同一seed paired差を
  別々の列へ保存します。行動familyをまたぐ共通比較ではstochastic列を主に使い、deterministic列には
  `greedy` / `mean_chain` / `mean_action` というreadout規約の差も含まれるものとして解釈してください。
  事前に一つのprimary metricを定めるなら、held-out episodeにおけるstochastic policyの最終returnを
  seed間で集約した値を推奨します。active連続比較では両手法のdeterministic readoutも`mean_chain`です。
- **評価episodeはseed間で重複する**: 評価環境のseedは training seed から
  `seed + 1_000_000 + episode` で決めます（`train.py` / `train_baselines.py`、`evaluate` は
  `make_env` と `reset` の両方へこの値を渡します）。productionの `eval_episodes = 10` では
  隣接するtraining seed（例: 10と11）が10 episode中9 episodeで同一の評価環境
  （同じ初期状態と同じ環境RNG stream）を共有し、seeds 10--14の全体でも異なる評価環境は
  1,000,010--1,000,023の14個しかありません。そのため `*_return_mean` / `*_return_std` などの
  seed間統計は独立5標本ではなく相関しており、seed数から素朴に計算した標準誤差は過小評価になります。
  同一seedのpaired差（`*_paired_difference_*`）では両手法が同じ評価環境seedを使うため、同一seed内では
  common random numbersとして環境noiseの一部を相殺できます。ただしtraining seedをまたぐpaired差どうしも
  評価episodeを共有して相関するため、paired差のseed間標準偏差や標準誤差を独立5標本として解釈することは
  できません。
  seed間の重複をなくすには評価seedの導出規則そのものを変える必要があり、それは全runの
  再実行を伴います。
- **paired差の符号規約と分散**: `*_paired_difference_mean` は
  同一 belief gradient mode・task・seedにおける **`score_transformer` − その行のmethod** です。
  正なら提案法が行のmethodを上回ったことを意味し、`score_transformer` 行は定義上0です。
  同じ paired差のseed間標準偏差（`ddof=1`）を `*_paired_difference_std`、その標本数を
  `*_paired_difference_n_seeds` として必ず併記します。paired統計へ入るのは、同じmode・taskの
  `score_transformer` runも `completed` であるseedだけです。
- **n<2 の標準偏差は空欄**: `deterministic_return_std` / `stochastic_return_std` /
  `*_paired_difference_std` は、対象seedが2未満のとき `0.0` ではなく空欄（JSONでは `null`）です。
  `seeds_completed` と `*_paired_difference_n_seeds` を併せて読めば、空欄が「分散0」ではなく
  「標本不足」であることが区別できます。
- 同ファイルにはenvironment steps、wall-clock、parameter数も保存します。productionのGRUは
  encoder MLP幅`[348,185]`、hidden幅`176`とし、共通のpolicy/value headを含むparameter数を
  `score_transformer`へ合わせています。

| production task | `score_transformer` | `gru` | 相対差 |
|---|---:|---:|---:|
| CartPole | 347,462 | 347,060 | -0.116% |
| Pendulum | 348,357 | 348,759 | +0.115% |
| MountainCarContinuous | 348,229 | 348,411 | +0.052% |
| Light-Dark (N=5) | 351,817 | 352,223 | +0.115% |

最大絶対差は0.116%です。local設定は実装確認用の小型networkであり、この厳密なcapacity比較の対象外です。

## 数式から補完した定義

- 初期 potential は専用MLP `g_0(o_0, s)` と標準正規priorの和です。`-||s||^2/2` は重みの
  正則化ではなく、初期密度の裾とlatent scaleを固定するGaussian基準密度です。`g_0`は通常時の
  観測energy `f(o_t, s, a_{t-1})` と重みを共有しません。無制限の関数族なら二次項を`g_0`へ
  吸収できますが、有限幅MLPが非有界latent空間全体で同じ二次tailを表す保証はないため明示的に残します。
- 再帰 potential は観測 energy と、前時刻粒子に対する遷移 energy の `logmeanexp` の和です。
- ULA の drift と noise は、どちらも各 step の `alpha_l` を使います。
- Transformer token は `(s_tk, score_t(s_tk))` です。位置 encoding は使わず、mean poolingで
  粒子順序に不変としています。Deep Sets ablationでも同じ tokenを使います。
- 連続 diffusion 方策は `x^N -> ... -> x^0` の全 chainを保存し、noisy側から
  `gamma_den^(N-1), ..., 1` の advantage重みを使います。
- `f` と `u` に別の score-matching教師は与えません。rollout時のULA初期noiseと全step noiseを保存し、
  PPO更新時にcurrent parametersから粒子生成を再parameterizeして、ULA全stepへ勾配を流します。
- `full` は未終了episodeのreplay入力をrollout境界越しに保持し、episode開始から内部belief再帰を全BPTTします。
  `tbptt_1` は環境時刻の境界だけdetachし、各時刻内のULA全stepは同じく微分します。両条件のforward値は同一で、
  差は時間方向のgradient範囲だけです。
- GRUでも同じ時間範囲を比較します。`full`はepisode開始から観測・直前行動列をcurrent parametersで再生し、
  `tbptt_1`は各環境stepの直前hiddenをdetachします。optimizer step後は、未終了episode prefixを更新後parametersで
  再生してonline hiddenを再構築します。
- 観測、実現済み行動、報酬はPPO trajectory dataとして固定します。外部環境や未知の遷移・観測モデルを
  微分するものではなく、完全にmodel-freeのままです。PPO更新後は保存履歴からongoing belief contextを再構築します。
- `full` の長いautograd graphにはnon-reentrant activation checkpointを使います。これはbackward時に同じ演算を
  再計算するmemory最適化で、gradientをtruncateしません。
- 文書中の「無限次元行動」は有限次元の連続 `Box` 行動として実装しています。関数値そのものを
  行動とする問題は対象外です。

連続行動の目的関数と近似は
[`docs/continuous-action-ppo.tex`](docs/continuous-action-ppo.tex)、関連研究と比較条件は
[`docs/related-work.tex`](docs/related-work.tex) に整理しています。同じ場所に PDF もあります。

## 成果物

比較実行ごとに `results/YYYYmmddTHHMMSS_label/` を新規作成し、既存結果は上書きしません。
Genkai productionだけは取得単位をまとめるため、この構造を
`results/campaigns/<campaign-id>/` の直下に最大80個作ります。

```text
results/<timestamp>_<label>/
  manifest.json
  metadata.json
  resolved_config.json
  summary.csv
  summary.json
  comparison.csv
  comparison.json
  <belief-gradient-mode>/<method>/<task>/seed_<seed>/
    run.log
    metrics.csv
    episodes.csv
    evaluation.csv
    arrays.npz
    evaluation_{greedy|mean_chain|mean_action|stochastic}_trajectories_*.npz
    resolved_config.json
    metadata.json
    checkpoints/{best,latest,step_*}.pt
```

NPZ は pickle objectを含まず、`allow_pickle=False` で読み込めます。evaluation CSVには方式、
固定評価 seed、return、action saturationを保存します。checkpointとJSONは一時ファイルから
atomicに置換します。score trainerの新しいcheckpointはmodel・optimizerに加え、環境 RNG、現在の観測、
belief context、belief replay履歴、進行中 episode、全乱数状態を保存し、update境界から厳密に再開できます。
score trainerの現在のcheckpoint formatはversion 4です。version 3以前や`trainer_state`を持たない
旧checkpointは、新しいmodel/config/log schemaと互換でないため、近似的に再開せず明示的に拒否します。
baseline trainerも同じ範囲（環境RNG、現在の観測、未終了episodeのreplay prefix、進行中episode、
全乱数状態）を保存しますが、`format_version` は専用の識別子 `baseline-v1` です。online hiddenだけは
保存せず、中断のないrunと同じ経路でprefixとcommit済みparametersから再構築します。互いのdialectは
復元前に拒否するので、score checkpointとbaseline checkpointを取り違えて再開することはできません。
modelまたはoptimizer stateにnon-finite値があるcheckpointも復元前に拒否し、同じrunの古い
`step_*.pt` へ戻す候補をエラーに表示します。最終updateの`latest.pt`保存後にprocessが停止しても、
次のsegmentは学習stepを重複実行せず、checkpointのtrainer stateから未完了のroot成果物を再構築します。

`summary.csv/json` の `evaluation_return_mean` は、最終評価のうち **deterministic readout側**
（`greedy` / `mean_chain` / `mean_action`）の平均returnであり、**primary metricではありません**。
名前が実体を表していないため、同じ値を持つ明示的な別名 `deterministic_readout_return_mean` を
併記します。旧名は投入済みcampaignが書いた全成果物との互換のために残してあり、両列は常に同一値です。
本READMEが推奨するprimary metricは `comparison.csv/json` のstochastic列
（`stochastic_return_mean` とその paired差）であって、この readout値ではありません。
別名を持たない旧`summary.json` / `manifest.json`も、集約時に旧名から別名を補完して同一視します
（`train.py`の`normalize_legacy_summary`）。

`metrics.csv` の `approximate_kl` と `clip_fraction` は、設定した全epoch・全minibatchの平均です。
これらは安定性を観測するための診断値であり、KL閾値によるearly stoppingやminibatch拒否には
使用しません。PPO log-ratioも事前clampしません。標準ULAを変えるscore/particle clampはmain設定で
`0.0`（無効）です。pilotで不安定性が出た場合は、まずlearning rateとLangevin step sizeを明示的に
調整し、条件間で同じ値を使います。

同じ`metrics.csv`には、初期energy $g_0$・観測energy $f$・遷移energy $u$それぞれの
pre-clip gradient norm、初期時刻/再帰時刻を分けたscore L2 normとparticle L2 norm、各区分の
particle数、およびscore/particleのnon-finite数も保存します。これらは数値安定性とgradient flowを
確認する診断値であり、optimizer更新を拒否する規則やprimary performance metricではありません。

独立して実行・取得した shard は、重複と必須成果物を検証してから一つの軽量レポートに集約できます。
`--config` は比較全体の master設定として必須です。各 sourceの `resolved_config.json` は、shard selector
（belief-gradient-mode、task、seed、method、label、output先）を除く全設定が masterと完全に一致する必要があります。さらに、
各 completed runの `resolved_config.json` も、その mode/method/task/seedから導出した設定と完全一致すること、
各 source内と全 source合算の mode/method/task/seed coverage、および実行したpackage sourceのSHA-256を
厳密に照合します。軽量レポートには集計CSV、
JSON、coverage metadataのみを保存し、大きい checkpointや軌跡はコピーしません。元 runの絶対 pathは
metadataへ記録されるため、元成果物も保持してください。

```powershell
uv run sb-pomdp-aggregate `
  --campaign-dir results/campaigns/<campaign-id> `
  --campaign-id <campaign-id> `
  --config config/production.json `
  --output-dir results `
  --label comparison-production
```

## Genkai

コードや本リポジトリ内のスクリプトがジョブを自動投入することはありません。
通常の短時間DEBUGは `b-batch` の1 node（4 full GPU）を30分だけ要求します。従来の精密DEBUG、
stability、productionは1 full GPUを要求します。いずれもGenkaiの Python 3.11 / PyTorch 2.3.1
module stackを読み込み、CUDAを認識できなければ即座に失敗します。各スクリプトは同じ
`config/production.json` を基準とし、有効minibatch 512を共有します。
score beliefはsequence microbatch 128へ分けてgradientを蓄積し、GRUは時間軸を切らず環境軸だけを
minibatch分割します。どちらもgradient clippingとoptimizer stepは論理minibatchごとに1回です。
短時間DEBUGだけは `num_envs=4`、`epochs=1`、1 update、評価1 episodeに縮小しますが、モデル、
粒子32、Langevin 8 step、rollout長128、minibatch 512、sequence microbatch 128は変えません。
以前のMIG debugではscore系がPyTorch/NVML allocator assertionで失敗したため、MIGは使いません。

productionは `masked_cartpole`、`masked_pendulum`、
`masked_mountain_car_continuous`、`light_dark` の全4タスクを含みます。
`light_dark` のproduction次元は `N=5` です。1 seedは
`2 methods x 2 gradient modes x 4 tasks = 16条件`、seeds 10--14の完全な行列は80条件です。

通常は、activeな全16条件を4 GPUの動的キューで確認します。内部watchdogは28分、PJM上限は
30分です。

```bash
pjsub -x "SB_POMDP_CAMPAIGN_ID=debug-short-$(date -u +%Y%m%dT%H%M%SZ)" jobs/genkai_debug_short.sh
```

`node=1`、GPU数の明示なし、`elapse=00:30:00` なので、B系の短時間ジョブ向け空きノードも
実行候補になります。実行時間は30分で強制終了しますが、キュー開始時刻と全16条件の正常完了を
保証するものではありません。1 node x 30分の要求上限は60 ptで、課金は実使用時間に比例します。

### 短時間DEBUGの実験上のメリット・デメリット

- メリット: productionと同じnetwork、方策head、粒子数、Langevin設定、128-stepのfull/TBPTT経路、
  minibatch/microbatch構造を実GPUで通します。全task・method・gradient modeのCUDA/OOM、非有限値、
  gradient、optimizer、評価、成果物生成の基本的な破綻を1 jobで検出できます。条件間では勾配を共有しません。
- デメリット: 1 updateのrolloutは2,048から512 transitionへ、PPO optimizer stepは16から1へ減ります。
  seed 0・1 update・評価1 episodeだけなので、性能比較、収束、分散、複数epoch後のKL、長期数値安定性、
  rollout境界を越えるfull replay、resume、本番の時間・point・peak memoryは判定できません。DEBUGのreturnを
  本番結果へ混ぜることもできません。

従来の2時間DEBUGは、短時間DEBUGで問題が見つかった場合の精密診断・本番時間校正用として残しています。
最重量条件だけを確認するコマンドは次です。

```bash
pjsub -x "SB_POMDP_CAMPAIGN_ID=debug-heavy-$(date -u +%Y%m%dT%H%M%SZ)" jobs/genkai_debug_heavy.sh
```

productionでは科学設定を変えず、実行方法だけを条件によって分けます。1 seedのうち
`full/score_transformer` の4タスクは、それぞれ「10 updates x 25本」のPJM step chainで同じshardを
厳密にresumeします。残る12条件（`full/gru`、`tbptt_1/score_transformer`、
`tbptt_1/gru` の各4タスク）は独立した通常jobです。異なるchain/jobはスケジューラが許す範囲で
並列実行されますが、1 chain内の25 segmentは逐次実行されます。

投入形態のこの区別は運用上のものだけで、resume能力の差ではありません。baseline
（`gru` / `observation_mlp` / `oracle_state`）もscoreと同一の`--segment-updates` + `--run-dir`
+ `--resume`経路で厳密にresumeします。baseline checkpointは専用のformat識別子を持つため、
score checkpointとの取り違えは復元前に拒否されます。独立jobがGPU時間上限で落ちた場合も、
同じrun rootへ同じsegmentコマンドを再投入すれば`checkpoints/latest.pt`から継続でき、
ゼロからの再実行は不要です。

seed 10の16条件について、投入内容だけを表示してから実投入するコマンドは次です。同じ明示的な
campaign IDをdry-runと実投入の両方へ渡してください。

```bash
campaign_id="prod-two-method-$(date -u +%Y%m%dT%H%M%SZ)"
bash jobs/submit_production_campaign.sh --dry-run --seed-offset 0 "$campaign_id"
bash jobs/submit_production_campaign.sh --seed-offset 0 "$campaign_id"
```

`--seed-offset 0..4` はproduction seeds 10--14に対応します。pilot確認後に残りも実行する場合は、
同じcampaign IDへoffset 1--4を投入します。

```bash
for seed_offset in 1 2 3 4; do
  bash jobs/submit_production_campaign.sh \
    --seed-offset "$seed_offset" \
    "$campaign_id"
done
```

まず1 shardだけで10 updatesの所要時間、GPU memory、数値安定性を確認する場合は、実測上重要な
seed-10 `full/score_transformer/MountainCar`（index 11）を使います。

```bash
pilot_id="prod-pilot-mountaincar-$(date -u +%Y%m%dT%H%M%SZ)"
pjsub -x "SB_POMDP_SHARD_INDEX=11,SB_POMDP_CAMPAIGN_ID=${pilot_id},SB_POMDP_SEGMENT_UPDATES=10" \
  jobs/genkai_production.sh
```

このsegmentが正常終了した後、同じ `pilot_id`、shard index、segment updatesで同じコマンドを再投入すると、
固定result rootの `checkpoints/latest.pt` から次の最大10 updatesを再開します。segment間で
`config/production.json`、source、campaign ID、shard indexを変更しないでください。同じshardの同時実行は
lockで拒否されます。resume可否は `metadata.json` の `source_sha256`（`src/sb_pomdp/*.py` 全体の
hash）で判定するため、**進行中のcampaignがある間は `src/sb_pomdp/` 配下を1バイトも変更しないでください**。
変更した場合、既存rootへのsegmentは「resumable run source code differs」で恒久的に拒否されます。
さらに厳密なresumeのため、Python・PyTorch・NumPy・device・CUDA・GPUのruntime identityも照合します。
同じsourceでもmodule stackやGPU種別を変更したsegmentは拒否されるため、同一chainでは実行環境も
固定してください。
sourceを更新した後にcampaignを再開・新規投入するときは、まず上記の短時間DEBUGを新しいtreeで
1回流してから投入してください。

初回updateのcheckpointが1つもcommitされる前にprocess/nodeが停止した場合、次回の同一segmentは
残ったseed treeを `.uncommitted-attempts/` 以下へ削除せず退避し、identity照合後にupdate 0から
再実行します。逆に `latest.pt` または `step_*.pt` が1つでもあれば、commit済み状態を勝手に捨てません。
`latest.pt`だけが欠けていれば最新のreadableな`step_*.pt`（必要なら`best.pt`）を自動選択します。`latest.pt` が
non-finite/corruptとして拒否された場合は、エラーに列挙される古い `step_*.pt` を確認し、
同じshardが停止中であることを確認してから、現在の`latest.pt`を別名で保存し、選んだstep checkpointを
`latest.pt`へコピーして同じsegmentを再投入してください。launcherは任意のstep pathを直接指定しないため、
このrollbackだけは手動です。

step chainの親jobだけでなく各subjobの状態を見るには、expanded表示を使います。

```bash
pjstat -E
# chain全体がactive一覧から消えた後の履歴
pjstat -E -H
```

成果物の `partial` は、segmentが正常終了してresume stateをatomicに保存したものの、設定された250 updatesは
まだ完了していない状態です。これは失敗ではありませんが、最終比較結果としては扱えません。
`completed` は全250 updatesと最終評価が完了した状態です。segmentが非0で終了した場合は実際の失敗であり、
campaign launcherの `sd=ec!=0:all` により、そのchainの残りsegmentは削除されます。
この規約はscoreとbaselineで同一です。configured updatesに到達していないbaseline runも
`partial` を報告するため、aggregateの「completedのみ集計」フィルタは両手法で同じ意味を持ちます。

step分割は1 jobあたり168時間という制限を回避するためのもので、総GPU時間やpointを削減しません。
各segmentは最大1 GPU x 168時間を要求し、異なる条件の並列化も終了までの暦時間を短くするだけです。
90,000 pt上限では、16条件一括投入より前に上記の1 shard・10-update pilotを実測し、その結果から
続行範囲を決めてください。

campaign全体または単独rootを上書きせず取得し、manifest・必須ファイル・NPZを検証するhelperもあります。

```powershell
uv run sb-pomdp-fetch `
  --host <alias-in-user-ssh-config> `
  --remote-results-dir /absolute/project/path/results/campaigns `
  --campaign-id <campaign-id> `
  --expected-shards 80
```

上記の1 shard pilotだけを取得する場合は、最後を `--expected-shard-indices 11` に置き換えます。
5 seeds・80条件がすべて `completed` になったcampaignだけが `--expected-shards 80` の対象です。
詳細なindexと運用上の注意は `jobs/README.md` を参照してください。

既定では manifest status が `completed`（と、診断成果物を含む `failed`）のresult rootだけを受け付けます。
上記の10-update pilotのように、まだ250 updatesへ到達していない `partial` shardを途中取得する場合は
`--allow-partial` を明示します。

```powershell
uv run sb-pomdp-fetch `
  --host <alias-in-user-ssh-config> `
  --remote-results-dir /absolute/project/path/results/campaigns `
  --campaign-id <pilot-id> `
  --expected-shard-indices 11 `
  --allow-partial
```

`partial` rootもmanifest・必須ファイル・NPZを同じ厳格さで検証します。唯一の緩和は各 seed runの
`evaluation.csv` で、最初の評価updateへ到達する前のsegmentでは存在しないため必須にしません。
取得した `partial` root名は標準エラーへ `accepted N partial result root(s): ...` と明示するので、
レポートが未完了shardを黙って含むことはありません。`--allow-partial` なしで `partial` を取得しようと
した場合は、そのフラグを案内するメッセージで失敗します。`partial` の成果物は途中経過の確認用であり、
最終比較結果としては使えません（`sb-pomdp-aggregate` は `completed` でないsource rootを拒否します）。

この形式はcampaign directoryを1回のrecursive SCPで取得し、指定したrootすべてのmanifest、campaign ID、
bulk index、必須ファイル、NPZを検証してから `results/campaigns/<campaign-id>/` へ配置します。
単独runを取得する場合は従来どおり `--run-name <timestamp_label>` を使えます。

接続先 alias `kyushu-super-agent` とRSA鍵 `agent-rsa` は `~/.ssh/config` にあり、repository直下の
`ssh-config` は鍵のpassphraseを保持するローカルファイルです。後者は OpenSSH configではないため、
`--ssh-config` へ渡しません。また `.gitignore` の対象です。秘密値を表示しない一時askpassを介した
接続、Genkaiへの配置、debug成果物の取得が成功しています。

## 研究上の制約

- PPO rewardだけでは `f, u` を真の likelihood/transitionとして同定できません。学習されるのは
  calibrated Bayesian beliefではなく、task-oriented latent score beliefです。
- 標準的なDPPOは事前学習済み diffusion policyのfine-tuningを主用途とします。本実装は全networkを
  ランダム初期化する DPPO-style from-scratchで、sample efficiencyが低い可能性があります。
- 共通 hyperparameterによる比較は environment-step budgetを揃え、active比較のparameter数も最大0.116%差へ
  合わせますが、各手法を個別に十分tuningした結果ではありません。5 seedsの標準偏差とpaired差を必ず併記してください
  （`comparison.csv` の `*_return_std`、`*_paired_difference_mean`、`*_paired_difference_std`、
  `*_paired_difference_n_seeds` がそのまま対応します）。
- Light-Darkで5次元連続行動を追加しましたが、関数値を直接扱う無限次元行動や、多峰な最適行動分布まで
  網羅するものではありません。diffusion方策の一般的な優位性を、この比較だけで主張はできません。
- `oracle_state` は完全観測参照ですが、有限学習予算と最適化誤差があるため理論的な性能上限ではありません。
- 実装した recurrent baselineはGRUだけで、LSTM、DPFRL、DVRL、FORBESは比較行列に含みません。
