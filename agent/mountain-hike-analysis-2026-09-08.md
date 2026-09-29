**Mountain Hike 全実験の考察 — 2026-09-08**

GenkaiへSSH接続し、7個の実行用treeにある39 campaign・105 seed runsの設定、manifest、評価、学習指標、実行ソースを読み取った。snapshotは2026-09-08 19:00:39 JSTからの取得で、取得中も学習は継続している。追加でスケジューラ履歴、10個の評価軌跡、4個の学習済みcheckpointを確認した。以下の数値はこのsnapshotに固定する。

主な結論は、**元のcold-start Score粒子には過去の情報を次時刻へ伝える経路が非常に弱く、初期の方策が盤面外へ進むと、一定報酬の領域から回復しにくい**ということ。warm startを含むv2/v2gでは改善する。ただし、改善版の完了結果は主にseed 10だけであり、手法の安定した優位性はまだ示せていない。

[学習曲線・条件比較・記憶経路の診断図](../results/analysis-20260908-mountain-hike/overview.png) ／ [全105 runの結果一覧](../results/analysis-20260908-mountain-hike/run-index.md)

**比較の読み方**

主指標はevaluation.csvのstochastic mean_return。高い値、つまり負の値なら0に近い方がよい。1評価につき10 episodes。以下の「±」は、明記したものを除き3個の**学習seed間の標本標準偏差**であり、10 episodes間の標準偏差や信頼区間ではない。最良checkpointではなく指定updateの評価を用いた。1 updateは2,048環境steps、200 updatesは409,600 steps。評価行のupdate/mode重複はなかった。完了63 runsについて最終評価stepsが設定budgetに一致することも検証した。

**通常版：3 seedsの結果**

| 手法 | seed 10 | seed 11 | seed 12 | 平均 ± SD | GRUとの差 ± SD（同じseed） | パラメータ数 |
|---|---|---|---|---|---|---|
| Transformer | -442.0 | -444.1 | -171.6 | -352.6 ± 156.8 | -208.2 ± 185.3 | 75286 |
| Alpha | -217.0 | -236.9 | -235.5 | -229.8 ± 11.1 | -85.4 ± 24.1 | 35525 |
| DeepSets | -413.1 | -254.8 | -438.5 | -368.8 ± 99.6 | -224.4 ± 90.1 | 67222 |
| GRU | -122.8 | -133.0 | -177.2 | -144.4 ± 28.9 | 0.0 ± 0.0 | 75674 |
| RNN | -140.1 | -156.4 | -138.3 | -144.9 ± 10.0 | -0.6 ± 34.4 | 75970 |
| PF (deterministic) | -446.2 | -170.3 | -197.1 | -271.2 ± 152.2 | -126.8 ± 170.4 | 75952 |

Score-Transformerはseed 10/11で失敗し、seed 12は−171.6まで改善する。したがって「全seedで学習不能」ではなく、失敗する学習経路へ入りやすい。DeepSetsも3 seeds中2つが悪い。Alphaは平均−229.8と劣るものの、seed間変動は小さい。Alphaはscore入力の有無・集約構造・head・パラメータ数も異なるため、その差をTransformerだけの優劣には帰属できない。

GRUとRNNの平均はほぼ同じ。元の決定論的PFも1 seedが−446.2まで崩れるので、問題の一部はScore固有ではなく、この環境とfrom-scratch方策学習に共通する。後述のDPFRL型PFは別実装として区別する。

**直接確認した失敗の機構**

1. **失敗した方策は盤面外へ進み続ける。** seed 10/11のScore-Transformerでは、保存状態の93.6%/93.5%が箱[-10,10]²の外だった。環境は箱外でほぼ−6/stepを返し、位置を箱内に戻すclampや箱外での終了は行わない。これが約−443という成績の実体。遠くへ出ると、大部分の行動が同じ報酬を受けるため、帰還方向を学ぶ信号が弱くなる。成功したseed 12、GRU seed 10、v2/v2g seed 10の保存状態では箱外率は0%。保存statesは行動前なので、箱外報酬の発生率と1 step分ずれる。
2. **cold-startの粒子は毎時刻ほぼノイズから再構成される。** v1はK=16、L=4、α=0.02、τ=1で、連鎖初期値は毎時刻新しい標準正規乱数。失敗seed 10の最後25 updatesで、recursive score norm平均は0.262、particle norm平均は1.343。1 ULA stepのdrift normの目安は0.02×0.262≃0.0052に対し、2次元Gaussian incrementのRMS normは√(2ατd)≃0.283。これは最終scoreでの近似診断であり、全内側stepの厳密なdrift比ではないが、雑音に比べてscoreによる移動が小さい。実際、driftを無視したcold粒子の期待normは約1.350で、観測された1.343に近い。
3. **記憶経路をcheckpointで検証しても、coldとwarmで大差がある。** 同じGRU評価の10 episodesの観測・行動列を4個の固定checkpointへ再生。乱数を固定し、t=5,10,…,70で前時刻の全粒子の両座標に+0.01を加え、次粒子の変化量÷前粒子の変化量を測った。表は14箇所の中央値。

| 固定checkpoint | 粒子の変化が次時刻へ伝わる比率 |
|---|---|
| v1_score_transformer_10.pt | 0.001143 |
| v1_score_transformer_12.pt | 0.003124 |
| v2-v1_score_transformer_10.pt | 0.918439 |
| v2g-v1_score_transformer_10.pt | 0.939844 |

これは一つの摂動方向・一つの共通入力集合での局所感度であり、相互情報量、全方向のJacobian、beliefの校正度を測ったものではない。ただし、v1の粒子更新が前粒子の変化を大幅に減衰させ、warm版に直接的な伝達経路ができていることは数値で確認できる。v1の成功seed 12にも同じ傾向があるので、「returnが改善したこと」だけを「粒子が正しいbeliefを表した証拠」とは読めない。エンコーダがscore値を現在観測の特徴として利用している可能性がある。

4. **entropy係数の増加が探索を増やしていない。** DiffusionPolicyは逆遷移の標準偏差を固定bufferから作り、PPOが足すentropyは各Gaussian逆遷移のentropy平均である。平均パラメータに依存せず、固定checkpointでもentropy.requires_grad=Falseだった。v2gのentropy_coef=0.01とent03=0.03は、最初100 updatesの全学習指標（速度・GPUメモリを除く）と8評価行が完全一致した。したがってこの比較から「探索を増やしても無効」とは言えない。実際には探索に関する変更が効いていない。これはGaussian方策一般のentropyや最終actionの周辺entropyについての主張ではない。

5. **長い微分経路と表現の目標の弱さも残る。** PPOはscoreの微分、ULAの反復、環境時刻方向を通して更新する。v1の遷移energy勾配はゼロではないが、単に勾配が存在することと、役立つ記憶が学べることは別。次観測・報酬の1-step予測損失は潜在表現を制約するが、長期予測の十分統計や真の状態事後分布を保証しない。高ノイズ下では予測器側の分散で誤差を吸収することも可能なので、損失値と予測分散の診断が必要。

**v1〜v4：同じ学習量で比較**

seed 10、75 updates=153,600 stepsに揃えた結果。v3は最新値も75-update評価で、実行は途中終了している。

| 条件 | Transformer | Alpha | DeepSets |
|---|---|---|---|
| v1: cold、補助なし | -442.3 | -200.1 | -419.7 |
| v2: warm＋観測anchor＋補助 | -216.0 | -195.6 | -223.2 |
| v3: cold＋補助 | -443.0 | -262.3 | -443.9 |
| v4: warm＋観測anchor、補助なし | -217.2 | -195.1 | -434.5 |

Transformerではv3の補助損失だけでは盤面外の失敗を解消せず、v4のwarm＋anchorだけでも大きく改善する。DeepSetsではv4が悪く、v2で大きく改善するため、補助損失と記憶の持ち越しが組み合わさる効果が示唆される。Alphaはcoldでも比較的学習しており、エンコーダへの依存が大きい。

ただしwarm startと観測anchorはこの比較では同時に切り替わる。ネットワーク追加は乱数消費と初期化にも影響し、seed 10単独なので、warmだけの因果効果を確定する比較ではない。独立したwarm-only/aux-only対照は、CPU途中実行に評価がなく、GPU再実行は待機中。

200-update完了結果は以下。GRUのseed 10は−122.8。

| 条件 | Transformer | Alpha | DeepSets |
|---|---|---|---|
| v1 | -442.0 | -217.0 | -413.1 |
| v2 | -164.3 | -187.1 | -180.1 |
| v2g: v2＋遷移proposal | -146.6 | -220.4 | -179.7 |

v2gの遷移proposalはTransformerの最終成績を改善する一方、Alphaでは悪化、DeepSetsではほぼ同程度。さらに100 updatesではTransformerもv2の−198.1がv2gの−207.9より良く、最終時点で順位が逆転する。したがって「gを入れれば常に改善」「100 updatesでの順位が最終順位」とは読めない。v4 Transformerの最新125-update評価は−201.6で、同時点のv2は−188.2。v2/v3のseed 11/12は進行中または評価前で、改善のseed間再現性は未確定。

**粒子数・内側更新・学習率・KAN・PFの比較**

| 実験群 | 確認した成績・状況 | 解釈 |
|---|---|---|
| K=32、100 updates | Transformer −443.4、Alpha −434.3、DeepSets −425.5、PF −444.4 | 単純な粒子数増加では解消しない。Alphaは幅704も使うので元の小さいAlphaとの比較は交絡する。 |
| K=64 | 本番campaignなし。DEBUG結果のみ | 性能については未評価。 |
| L=8, α=.05、50 updates | Transformer −443.5、Alpha −415.4 | 同時にL/αを変更。50 updates時点の改善根拠なし。 |
| L=16 / L=16, α=.1 | 4 runsに性能評価なし、full GPUの4 jobsはQUE | 過去の実行はGPU identity不一致でfailed。学習性能の失敗に数えない。 |
| 学習率 .001 / .00006、200 updates | Transformerは双方−442.6。PFは−140.0 / −209.5 | Scoreの失敗は学習率変更だけでは解消せず、PFには大きな影響。単一seed。 |
| Alphaのパラメータ数一致、200 updates | 75,269 params、−217.8。元のseed 10は−217.0 | 同seedの最終値では容量増加の改善なし。途中の学習曲線は異なる。 |
| KAN head/trunk、200 updates | Transformer −191.2、Alpha −199.6、DeepSets −218.7 | この群はenergy_network_kind=mlpのまま。Score energyをKAN化した効果とは言えない。Alphaの幅も異なる。 |
| KAN baseline、200 updates | GRU −225.4、RNN −337.7、PF −205.6 | KANが一律に良い結果ではない。 |
| DPFRL型PF、200 updates | seeds 10/11/12: −174.3 / −210.0 / −134.1、平均−172.8 ± 38.0 | 元の決定論的PFより安定。粒子表現一般の否定はできない。GRUとの差の平均は−28.4。 |
| DPFRL型PF、128次元・MGFあり | seed 10: −136.5、211,536 params | 性能は良いが元のScoreの約2.8倍の容量。原論文全体の再現成績や同容量比較とは区別する。 |

**v2gのハイパーパラメータ比較**

すべてseed 10、100 updatesのTransformer。軽量版はモデル幅・K・Lも同時に変わる。

| 変更 | return@100 |
|---|---|
| 基準 | -207.9 |
| α=.1 | -198.9 |
| α=.3 | -318.2 |
| τ=.3 | -195.8 |
| τ=.1 | -331.0 |
| 学習率=.001 | -218.2 |
| entropy係数=.03 | -207.9 |
| 両補助係数=.1 | -211.6 |
| 両補助係数=10 | -210.5 |
| score入力なし | -212.7 |
| 軽量ネット＋K32/L8 | -286.5 |

α=.1やτ=.3はこのseedの100-update値では小改善にとどまり、α=.3やτ=.1は大きく悪化。単純に移動幅を大きくする／雑音を小さくする方向ではない。最後25 updatesの平均particle normはα=.3で23.1（v2g最終期10.0）、最大絶対座標の平均は65.5まで増える。ただし粒子座標は教師なし潜在座標なので、物理座標の範囲超過や推定誤差そのものとは解釈しない。score入力除去は−212.7で基準−207.9と近く、v2gにおけるscore入力が必ず有害とは言えない。

軽量K32のAlphaは−204.4。軽量L16はまだ25-update評価のみで、Transformer −309.3 / Alpha −283.8。長いULAの有効性を結論づける段階ではない。補助係数=.1/10は観測と報酬の両方を同時に変えており、それぞれの必要性は分離できない。

**Qクリティック版**

| 条件（seed 10） | return@75 | return@100 | 最新評価 | 状況 |
|---|---|---|---|---|
| 通常GRU | -210.5 | -167.2 | -122.8 @200 | 完了 |
| Q-GRU | -275.8 | -300.2 | -442.6 @200 | 完了 |
| Q-Transformer | -279.7 | -294.4 | -300.8 @150 | 途中 |
| Q-Alpha | -160.3 | -274.1 | -273.3 @125 | 途中 |
| Q-DeepSets | -272.8 | -275.6 | -311.1 @125 | 途中 |
| Q/GAE 50:50 | -185.8 | -252.8 | -256.4 @125 | 途中 |

Q-GRUも最終−442.6へ崩れるため、Q版の問題をScore表現だけへ帰属できない。Q-Alphaは75 updatesの−160.3から125 updatesの−273.3へ後退する。Q/GAE混合も50 updatesの−191.3から125 updatesの−256.4へ後退するので、best checkpointだけなら不安定さを隠す。Q-Transformerの最後25学習updatesでKL平均0.0486、clip fraction 0.255、gradient norm 61.9。v2gの最終25 updatesではそれぞれ0.0040、0.0424、5.39だった。この窓は同updateではないため記述的比較だが、Q版で更新が荒くなっていることを示す。

**Q版には解釈前に直したい数式上の不整合もある。** Q headはsymlog(return)を学習するが、expected_action_valueは変換後のQ出力を行動方向に平均し、その後でsymexpを適用する。これは平均したraw Qと等しくない。例えば二つのraw値−1,−9の平均は−5だが、symlogを平均して戻すと約−3.472。標榜するraw単位のEπQにするには、各行動の値をrawへ戻してから平均する必要がある。この差はbootstrapにも入る。なお、この不整合が今回の崩壊をどれだけ説明するかは修正対照が必要で、唯一の原因とは断定しない。Q損失自体が小さくても、行動間の価値差の正確さは保証されない。

**テレポート版（TP）・対称観測版（TPS）**

TPの100-update完了結果はGRU −219.0、RNN −238.0、DPFRL型PF −332.4（すべてseed 10）。TPのScore v1〜v4は本番結果を確認できなかったため、Scoreとの比較は未成立。通常版からのreturn差は環境が変わるので、純粋なfiltering性能差ではない。

TPSについて共通に存在する25-update評価を示す。観測は|位置|＋雑音、teleport確率=.03。

| 条件 | Transformer | Alpha | DeepSets |
|---|---|---|---|
| v1 | -335.7 | -269.9 | -320.6 |
| v2 | -348.6 | -292.0 | -261.6 |
| v3 | -354.8 | -297.5 | -361.3 |
| v4 | -294.1 | -269.5 | -350.5 |

TPS baselineの25-update値はGRU −286.2、RNN −325.8、PF −339.4。100-update完了値はGRU −271.9、RNN −262.7、PF −281.9。Scoreの最新評価はv1@50、v2@75、v3@25、v4@50で、まだ最終順位はつけられない。warmの有利不利はエンコーダにも依存しており、「teleportならwarmが必ず不利」という仮説はこの途中結果から支持されない。

**TPSの実験意図と入力設計にずれがある。** 観測energyと遷移energyは観測・前行動・前粒子を受け取るが、実際に受け取った直前報酬は入力しない。baselineも報酬履歴をオンライン入力にしていない。reward_prediction_coef>0は報酬を教師信号にするだけで、当該episodeで観測した報酬によってbeliefを更新する仕組みではない。したがって「非対称な報酬を見て4象限の候補を絞れるか」の検証になっていない。行動とその後の|位置|の変化、開始位置の事前分布から符号が分かる場合はあるため、符号識別が原理的に不可能という意味ではない。

**理論上の解釈と追加実験の優先順位**

学習しているのは報酬と補助損失で調整される潜在的なscore表現であり、真の状態事後分布の校正を保証しない。recursive potentialにはinitial分岐の−||x||²/2に相当する裾の制約がなく、学習されたexp(potential)の正規化可能性も自動的には保証されない。短いLangevin連鎖が便利な生成写像として学習されることと、平衡分布からサンプリングできることは区別する必要がある。[短い非収束MCMCを生成器として扱う研究](https://arxiv.org/abs/1904.09770)もこの区別を示している。これは今回の性能劣化の直接証拠ではなく、「score粒子＝Bayesian belief」と主張する際の制約である。

1. v2/v3/v4を同じ予算・seedsで比較し、warm＋anchorと補助損失の相互作用を確かめる。特にv2のseed 12は25 updatesで−442.5であり、改善が頑健とはまだ言えない。warm-only対照が揃えばanchorとの切り分けも進む。
2. 大規模なK/L掃引より先に、固定checkpointで履歴を消す／shuffleする対照、score入力を消す対照、共通観測列でのstate decoderや複数stepの予測を評価する。decoderは学習用episodeと評価用episodeを分ける。潜在粒子座標を物理座標と直接引き算するだけのRMSEは避ける。
3. 箱外への流出率・最初の流出時刻・復帰率を記録し、探索に実際に勾配が出る変更を、GRUにも同条件で比較する。entropy係数だけの再掃引は今のdiffusion設定では有効な対照にならない。環境の終了条件・箱外報酬を変える場合は別タスクとして比較する。
4. Q版はraw EπQの集約順序を整え、通常GRUにも同じ修正を適用して対照する。Q-GRUまで崩れている以上、Score側の改変を先に増やす根拠は弱い。
5. TPSで報酬による候補の絞り込みを測りたいなら、直前報酬を全手法のオンライン入力に含めた別条件を設ける。現条件は「報酬入力なしの対称観測」であると明記する。
6. 補助損失の値、予測分散、損失別のbelief勾配、粒子多様性・履歴感度を追加診断する。現metricsでは観測／報酬予測損失が個別に記録されておらず、どちらの補助項が機能したかは未確定。

DPFRLは識別的に粒子表現を方策と学習する枠組みであり、粒子が厳密な物理状態posteriorでなくても制御に役立ち得る。[DPFRL原論文](https://arxiv.org/abs/2002.09884)と[生成モデルを併用するDVRL原論文](https://proceedings.mlr.press/v80/igl18a.html)を踏まえても、今回の結果は「Score方式全般の否定」より「この短いcold連鎖と学習信号の組合せの弱点」として読むのが妥当。

**実行状態と証拠の所在**

manifest集計ではcompleted 63、running 38、failed 4。ただしrunningのうち5 runsはスケジューラと食い違う。v3 seed 10の3 jobs（6720533/34/35）は履歴でEXT、CPU対照の2 jobs（6720872/73）はCCLだった。v3の最新学習updateは83/83/81、評価は75。CPU対照は13/16 updatesで評価なし。EXTの原因はこの取得で終了詳細まで確定していない。L16系4 jobs、v4追加seeds6 jobs、warm-only/aux-onlyのGPU2 jobsはQUE。QUEの予定開始時刻は確定時刻として扱わない。

記録上の学習wall timeは元のScore-Transformerが各seed約52〜54時間、GRU約0.18〜0.19時間。計算量の差も大きい。ただし逐次resumeを含む実装固有のwall-time指標であり、課金GPU時間そのものやハードウェア一般の比較値ではない。性能比較と計算費用は別々に報告する必要がある。

元データは[取得snapshot](../results/analysis-20260908-mountain-hike/snapshot.json)、[スケジューラ一覧](../results/analysis-20260908-mountain-hike/queue.txt)、[履歴](../results/analysis-20260908-mountain-hike/queue-history.txt)、[checkpoint診断結果](../results/analysis-20260908-mountain-hike/probe.json)に保存した。snapshot内には各remote path、resolved_config、source hash、実行ソース、全評価行、学習指標が含まれる。再現用probe.py、評価軌跡、4個のcheckpointも同ディレクトリに保存した。認証情報は含まない。

主な実装箇所は、belief.pyのparticles_from_noise / _recursive_log_potential、networks.pyのBeliefSetEncoder / expected_action_value、policies.pyのDiffusionPolicy.transition_distribution、ppo.pyのaction_value_advantages / reward_prediction_loss、envs.pyのMountainHikeEnv.step。実行treeごとのソースをremote-sources以下に保存しており、Q/TPSの指摘は実際のremote版でも確認した。
