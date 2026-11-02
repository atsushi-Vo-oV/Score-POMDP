# Codex レビュー 2026-11-02(GPT-6-Astra、推論 high、read-only、コミット 4e80ac9 + sequence_microbatch_size 512 の案)

依頼文は末尾。対応: 第 1 部 1 → 予測損失を系列ごとの平均に修正(ppo.py `_sequence_mean`)。第 2 部 1 → FLOPs を fast path 込みで数え直し、基準線と alpha の幅を合わせ直した(v2)。第 2 部 2 → results/reports/score_alpha_20261102.pdf。第 2 部 7 → README 訂正。その他は報告に記載。



# レビュー報告

**現状では、128→512 を「学習結果を変えない純粋な実行変更」とは扱えません。通常の PPO 部分は等価ですが、予測フェーズでは終了マスクに応じた損失の重みが変わります。** また、実験設計には、FLOPs の計数漏れと「α-vector 方策」という説明と実装の不一致があります。

ファイルは変更していません。対象 22 config の検証、関連する既存テスト **49 件の成功**、CPU 上での追加検証を確認しました。GPU の速度・メモリ測定は再実施していません。`results/` は `.gitkeep` のみで、本番結果・pilot の根拠データは確認できませんでした。探索した範囲に `CLAUDE.md` はありませんでした。

## 第1部：microbatch 高速化のレビュー

### 1. 【major】予測フェーズの損失が microbatch 分割に依存する

**根拠：** `src/sb_pomdp/ppo.py:520`, `:526`, `:555`, `:556`, `:760`, `:783`, `:786`、`src/sb_pomdp/train.py:2324`

`condition_prediction_loss` と `belief_mixture_prediction_loss` は、各 microbatch 内の **有効遷移数**で損失を割ります。しかし、勾配累積時の重みは **環境数の比率**です。

環境ごとの有効遷移数を \(n_e\)、その平均損失を \(L_e\) とすると、4 環境の minibatch に対して、

- 128 設定：\(\frac14\sum_e L_e\)
- 512 設定：\(\frac{\sum_e n_eL_e}{\sum_e n_e}\)

となり、終了回数が異なると一致しません。例えば、有効数が 2 と 1、平均損失が 1 と 9 なら、前者は **5**、後者は **11/3** です。

メモリ上の追加検証で、両予測関数についてこの差を再現しました。`update_ppo(objective="prediction")` を通す検証でも、同じ初期値から損失が **7.25→9.6667**、SGD 更新後のパラメータが **0.45→0.60** と変わりました。

対象 config は予測 pretrain と interleaved prediction を有効にしているため、これは未使用機能の問題ではありません。特に tagr は捕獲によって終了時刻が環境ごとに変わります。一方、mirror2／tps の現在の固定 horizon・`goal_end=false` では、環境が同期終了するため、この不一致が必ず発生するとは限りません。

**対応：** まず目的関数を決めてください。

- 従来の 128 と等価にするなら、512 でも環境ごとに有効遷移平均を計算してから環境平均する。
- 全有効遷移の平均に統一するなら、minibatch 全体の有効数で各 microbatch の寄与を正規化する。ただし、これは従来の 128 の学習目的を変更します。

同じ構造は PPO 中の観測・報酬補助損失にもあります（`ppo.py:166`, `:593`）。今回それらの係数はゼロですが、共通実装として修正対象です。

### 2. 【major】既存 run の再開・集計は config 変更だけでは通らない

**根拠：** `src/sb_pomdp/config.py:1142`、`src/sb_pomdp/compare.py:726`、`src/sb_pomdp/train.py:1518`、`src/sb_pomdp/aggregate.py:50`, `:91`

`EXECUTION_ONLY_KEYS` が除外するのは `model.belief_activation_checkpoint` だけです。したがって、128 で保存した run／checkpoint を 512 config で再開すると不一致になります。集計側も microbatch の差を許容しません。

追加検証で、設定比較が不一致になり、集計が次の差を理由に拒否することを確認しました。

```text
ppo.sequence_microbatch_size: expected 512, found 128
```

**対応：** 第1項の意味論を解決する前に、このキーを無条件で execution-only に追加してはいけません。従来の損失を保存できる場合に限って、run・checkpoint・集計の比較規則を揃え、変更履歴を残してください。目的関数を直す場合は別条件として管理すべきです。最初から 512 で開始する新規 run には、この再開問題はありません。

### 3. 【major・対象22設定では未使用】action-value critic では通常の PPO でも等価性が崩れる

**根拠：** `src/sb_pomdp/ppo.py:117`, `:124`、`src/sb_pomdp/networks.py:722`、`src/sb_pomdp/policies.py:690`

Q-based advantage は、その呼び出しに渡された microbatch 内で平均・標準偏差を計算します。128 と 512 では正規化範囲が変わります。連続行動の期待 Q も、その場で方策からサンプルするため、乱数の各サンプルへの割り当てが分割方法に依存します。

**対応：** Q advantage とその乱数を論理 minibatch 単位で固定し、正規化後に分割してください。今回の 22 設定は state-value critic なので直接の阻害要因ではありませんが、「このキーは常に execution-only」という一般化を妨げます。非ゼロ dropout を使う設定についても別途検証が必要です。

### 4. 【minor】引用されたテストは、今回必要な等価性を十分に検証していない

**根拠：** `src/tests/test_ppo_microbatch.py:28`, `:43`, `:94`, `:123`, `:144`

引用されたテストは、identity belief、categorical policy、空 prefix、終了なし、advantage 正規化なしの小型モデルです。予測フェーズ、実際の score 微分、diffusion、SVGD／SMI、mixed reset は検証していません。

追加で実際のモデルを用い、energy／direct／direct_joint、ULA／SVGD／SMI、Transformer／使用可能な alpha_pool の **15 組**を検証しました。不均等な prefix と mixed reset を含む小型条件で、通常 PPO の更新後パラメータ差は最大 **2.98×10⁻⁸** でした。ただし、これは CPU・短い系列・SGD による検証であり、本番 GPU・Adam・長時間実行の一致を保証するものではありません。

コード上、今回の通常 PPO 経路については次を確認しました。

| 確認箇所 | 判定 |
|---|---|
| advantage 正規化 | rollout 全体で実施され、分割に依存しない。`ppo.py:717` |
| policy／diffusion／value 損失 | 固定長系列の平均と環境数による重み付けが整合する。`ppo.py:432`, `:818` |
| 勾配 clip・optimizer・skip | 全 microbatch の累積後に一度実施。非有限時は論理 minibatch 全体を skip。`ppo.py:882` |
| metrics | microbatch を重み付き集計し、成功した minibatch 間で平均。`ppo.py:849`, `:896` |
| replay の乱数・prefix・reset | 保存 noise を使い、環境ごとの prefix と reset を維持。`ppo.py:331`, `:360`, `:405` |
| mixed branch | 行ごとの `torch.where`。有限値の範囲では環境間の情報混合なし。`belief.py:550`, `:597` |
| SVGD／SMI | median は batch 行ごと、混合演算は粒子軸内。`belief.py:106`, `:133` |
| Transformer／alpha_pool | LayerNorm は特徴軸、pooling は粒子軸。`networks.py:66`, `:96`, `:190`, `:204` |

**対応：** この組合せに、不均等な終了マスクの予測フェーズ、複数 epoch、Adam、skip の回帰検証を加えてください。「数学的に同じ目的関数」と浮動小数点の bitwise 一致は区別すべきです。

なお、128 を固定値として検査する `test_ppo_microbatch.py:192` と `test_genkai_submission.py:126` は、変更対象外の `production.json` を検査しています。今回の変更だけを理由に 512 へ書き換える必要はありません。

### 5. 【minor・改善機会】追加高速化は prefix／recondition のバッチ化を優先できる

**根拠：** `src/sb_pomdp/ppo.py:360`, `:383`、`src/sb_pomdp/train.py:428`, `:435`

512 にしても prefix は各環境を個別に再生し、recondition も 16 環境を順番に処理します。ここには、optimizer の更新単位を変えずに高速化する余地があります。

| 候補 | 等価性を保つ条件・期待できる範囲 |
|---|---|
| prefix の環境バッチ化 | 同じ prefix 長でまとめる、または有効行を管理する。4 環境分の呼び出しをまとめられるが、実測倍率は未確認。 |
| recondition のバッチ化 | 保存 noise と履歴を維持する。16 環境をまとめる余地がある。branch 判定のホスト転送もまとめられる。 |
| 不要な最終 score を省略 | prefix／recondition は戻り値の score を捨てる。alpha_pool の score 非使用経路も候補。ULA／SVGD の score 評価は 13 回から 12 回になり、該当部分の呼び出し数を約 7.7% 削減。`belief.py:965` |
| diffusion の保存 chain 評価をバッチ化 | PPO 中の 5 denoising step は保存入力なのでまとめられる。生成時の逐次依存とは区別する。`policies.py:226` |
| rollout の entropy 再計算を除去 | サンプル後に `log_prob_chain` を呼び直し、denoiser を再実行している。固定分散の entropy は別計算可能。score・baseline 共通。`policies.py:729`、`baselines.py:318` |
| CPU 同期・Python 処理削減 | metrics の `.cpu()` をまとめる、固定 schedule／index を再利用する。`ppo.py:849` |

prefix 長を一律 \(H\)、rollout 長を \(T\) とした呼び出し数の概算では、今回の変更は \(4(T+H)\) から \(T+4H\) への削減です。prefix が長いほど、rollout 部分だけをまとめる効果は薄まります。これは速度の測定値ではありません。

注意点もあります。

- **4 環境の論理 minibatch をさらに統合すると optimizer step 数が変わります。** epoch 間や optimizer 更新後に古い prefix 計算を使い回すことも等価ではありません。
- `tbptt_1` が切るのは時間境界だけです。現在 step の score 学習には `create_graph=True` が必要で、一律に外せません。rollout／recondition は既に `track_grad=False` です。
- 12 transport step は逐次依存しています。融合は候補ですが、単純な並列化ではありません。
- CUDA Graphs／`torch.compile` は小さな固定形状の部分から検証すべきです。可変 prefix、CPU 分岐・転送、高階微分があり、そのまま適用できるとは限りません。[CUDA Graphs の制約](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/best-practices.html)、[PyTorch の double-backward 関連 issue](https://github.com/pytorch/pytorch/issues/91469)

brief の測定値 **351→163 秒は 2.15 倍、約54%短縮**です。一方、4 秒の rollout を完全に除去しても、`rollout + PPO` 全体の短縮上限は変更前約1.1%、変更後約2.4%です。追加作業は PPO replay と prefix を優先するのが妥当です。

## 第2部：実験設計のレビュー

### 1. 【major】FLOPs 表を再現すると、Transformer 本体の計数漏れが発生する

**根拠：** `jobs/trysvgd/README.md:9`、`jobs/trysvgd_alpha/README.md:7`, `:20`、`src/sb_pomdp/networks.py:69`, `:95`、`src/sb_pomdp/train.py:756`

ローカルの指定依存バージョン `torch 2.3.1+cpu` で、実際の belief 更新と acting 経路を `FlopCounterMode` で計測すると、README と一致する数字が得られました。しかし、eval／no-grad 時の融合 Transformer 演算がカウンターの登録対象外で、本体が計上されません。

| tagr・energy ULA | FLOPs |
|---|---:|
| 通常 fast path での Transformer encoder 計数 | 12,288 |
| fast path を無効化し attention を展開した計数 | 307,200 |
| **漏れている差分** | **294,912** |
| README と一致する TF 全体の計数 | 2,034,368 |
| 上記漏れを補った全体 | **2,329,280** |
| alpha、幅30 | 2,034,816 |

展開前後の encoder 出力差は最大 **2.38×10⁻⁷** でした。幅30の alpha は、この補正後の TF より約 **12.6%少ない**計数になります。同じ測定範囲で alpha の幅を272にすると **2,329,088** となりました。

したがって、「belief が支配的だから幅26〜30で一致する」という説明は、そのままでは支持できません。少なくとも、この計測経路では **小さい幅を選んだ主因に計数漏れがあります**。

**対応：** 測定スクリプト、PyTorch バージョン、device、batch、初期／再帰 step、未計数 operator を保存し、融合演算を展開または独自計数してください。幅は再計算が必要です。上記272は現在の計測範囲を維持した検証値であり、最終推奨幅ではありません。

さらに、現在の acting 経路には critic と entropy 用の二重 denoiser 評価も含まれます（`train.py:778`、`policies.py:729`）。「配備時の行動選択」と「学習用 rollout の処理」のどちらを予算にするか、先に定義すべきです。元の測定スクリプトがないため、本番で用いた計測条件そのものは確定できません。

### 2. 【major】連続行動の `score_alpha` は「行動ごとの α-vector 方策」ではない

**根拠：** `jobs/trysvgd_alpha/README.md:3`、`src/sb_pomdp/compare.py:216`、`src/sb_pomdp/policies.py:562`, `:575`, `:596`、`src/tests/test_alpha_pool.py:128`

`score_alpha` は `alpha_pool + alpha_lse` を設定しますが、連続行動の場合、

- encoder：`AlphaPoolBeliefEncoder`
- critic：`AlphaLSEHead`
- actor：通常の `DiffusionPolicy` の MLP denoiser

になります。行動別の `AlphaLSEHead` が使われるのは離散行動だけです。既存テストも、連続 alpha が diffusion actor を維持することを明示しています。

また温度1の log-sum-exp は滑らかな最大値近似で、厳密な区分線形関数ではありません（`networks.py:978`）。belief-linear な pooling の性質も、その後に非線形 denoiser を通した方策全体には引き継がれません。

**対応：** 現状を **「粒子位置の mean-pooling encoder と smooth-max critic を持つ diffusion 方策」** と記述してください。古典的 α-vector 方策との比較を意図するなら、actor の定義から変更する必要があり、別実験になります。

### 3. 【major】TF 対 alpha は、readout 構造以外にも入力情報と critic が同時に変わる

**根拠：** `src/sb_pomdp/policies.py:447`, `:469`, `:562`、`src/sb_pomdp/networks.py:157`、`src/sb_pomdp/belief.py:975`

TF は粒子位置と score を読み、alpha は粒子位置だけを読みます。SMI の TF はさらに成分の `log scale` を読みます。score は現在観測などを直接条件にするので、位置に十分な情報が記憶されていなくても、観測特徴の経路として働き得ます。

加えて alpha は critic も変更します。したがって、差を「Transformer と α-vector readout の効果」に帰属できません。

**対応：** パッケージ全体の比較としては解釈可能ですが、構造の因果効果を調べるには、少なくとも **TF の score 入力を外した対照**と、critic を揃えた対照が必要です。score を alpha に足す対照も可能ですが、その場合は「固定した粒子特徴の belief-linear functional」という解釈を再検討してください。

予測ヘッドについては、今回の `condition` 設定では各手法が同じ32次元 condition から、同じ構造の4成分混合ガウスを学習します。baseline の予測フェーズも実装されています（`train_baselines.py:237`, `:799`）。ここでの「共通」は**構造・目的関数が共通**という意味で、手法間で重みを共有する意味ではありません。この点は概ね整っています。

### 4. 【major】5 seeds × 10 episodes から強い優位性・同等性を主張するには不足がある

**根拠：** `config/trysvgd_tagr_tf-ula.json:8`, `:18`、`src/sb_pomdp/train.py:904`, `:913`, `:2448`、`src/sb_pomdp/train_baselines.py:895`、`agent/experiment-validity-audit-2026-09-14.md`

評価環境 seed は `training_seed + 1,000,000 + episode` です。学習 seeds 10〜14、各10 episodesでは、異なる環境 seed は **14個**しかなく、隣接学習 seed 間で9個を共有します。50個の独立評価ケースとして扱えません。

学習 seed と評価集合も同時に変わるため、学習変動と環境難易度を分離できません。同じ seed 名でも、ネットワーク初期化や粒子処理の乱数消費は手法間で異なり、全乱数が対応した比較ではありません。

**対応：**

- 主指標、最終 checkpoint／評価窓を事前に固定する。
- 固定 checkpoint を、全手法・全学習 seed に共通する十分な数の held-out episodes で再評価する。
- 学習 seed 間と評価 episode 間の変動を分け、失敗 run も件数・割合として報告する。
- stochastic と mean-chain を両方示し、同等性には実用上の許容差と不確実性を示す。

5 seeds は探索的比較として無意味ではありませんが、小差や一般的優位性を保証する設計ではありません。mean-chain でも belief の初期化・ULA、DPFRL の内部乱数は残ります。

なお、評価は新しい環境を生成し、学習 RNG を保存・復元しています。**学習環境の使い回しや、今回の通常腕への真の状態の入力漏洩は確認しませんでした。** 真の状態は評価保存用、oracle 入力は専用分岐です（`train.py:952`、`train_baselines.py:277`, `:919`）。

### 5. 【major】「同じ計算予算」は acting・学習・サンプル数を分けて定義する必要がある

**根拠：** `jobs/trysvgd/README.md:9`、`jobs/trysvgd_tagr/README.md:7`、`src/sb_pomdp/ppo.py:383`, `:735`、`src/sb_pomdp/prediction_schedule.py:35`

acting FLOPs を揃えても、score の高階微分・full replay、baseline の学習、prefix 再生の費用は揃いません。episode が長く生存する腕ほど prefix が長くなり、同じ更新数でも訓練費用が変わります。

また baseline の幅は energy TF-ULA を基準に固定されています。直接 score 版や SVGD／SMI まで全腕が同額ではありません。FLOPs の計数漏れとは別に、「全腕で一致」という表現は不正確です。

予測フェーズも追加環境 steps を消費します。

| task | PPO steps | 予測を含む総学習 steps |
|---|---:|---:|
| mirror2 | 409,600 | 681,984 |
| tps | 204,800 | 436,224 |
| tagr | 2,048,000 | 2,648,064 |

pretrain は端数を切り上げるため、実際は **200,704 steps** です。

**対応：** 「固定環境 steps・所定の acting FLOPs 帯域での性能」と限定し、総環境 steps、学習時間、GPU 時間、ピークメモリ、モデル規模を併記してください。幅の一様拡大は一つの合理的な作成規則ですが、最良 baseline を得た証明ではありません。各手法に同程度の tuning 予算を与えるか、その制約を明記すべきです。

粒子数16は score と DPFRL で揃っていますが、DPFRL の粒子次元は16、tagr score は6です（`train_baselines.py:217`、`config/trysvgd_tagr_base.json:132`）。同じ粒子数を同じ表現容量と解釈してはいけません。

### 6. 【major】tagr の設計は部分観測課題として成立するが、識別可能性・収束・oracle の主張は未立証

**根拠：** `src/sb_pomdp/envs.py:1631`, `:1709`, `:1747`, `:1759`, `:1805`、`jobs/trysvgd_tagr/README.md:4`

観測は自分の状態5成分と、固定 ID 順の鬼3体への noisy range です。プレイヤー側の視野制限はありませんが、**鬼の追跡判定には視野・距離制限が残ります**。

距離は単時刻では方位を決めず、移動履歴によって曖昧さを減らせます。しかし鬼自身も確率的に動き、heading・追跡 mode・記憶した target・速度を持つため、「range なら真の状態を同定できる」とは保証できません。

さらに、

- simulator の状態次元は **25**。潜在次元6は学習表現の設計値で、真の状態次元ではありません。
- `oracle_features` も鬼の heading と記憶 target を省いています。これは privileged observation であり、完全な Markov state を与える oracle ではありません。
- 現設定は捕獲 step にも0.1を払い、`capture_reward=0` なので、return は **0.1×episode長**、最大20です。

README の「理想の信念で約1.5改善」「1000更新までに頭打ち」は、対応するコード・曲線・seed別結果がないため確認できません。

**対応：** pilot の設定と全 seed の曲線を保存し、1000更新はまず固定予算として記述してください。記憶なし、観測を使わない簡単な方策、privileged-state 参照を追加し、生存時間・捕獲率とともに評価すると、belief が有用かを判断しやすくなります。潜在次元6の十分性は容量対照や held-out 予測診断で検証すべきです。

### 7. 【major】学習する transport schedule の学習率が README と10倍違う

**根拠：** `jobs/trysvgd/README.md:7`、`src/sb_pomdp/config.py:298`、`src/sb_pomdp/train.py:723`, `:731`、`config/trysvgd_tagr_tf-ula.json:102`

README は schedule 学習率を **0.0025** としています。しかし対象 config に倍率指定はなく、既定の `langevin_schedule_lr_multiplier=1.0` が入り、実際は **0.00025** です。対象22設定すべてで確認しました。確認したジョブ経路にも倍率10の override はありません。

**対応：** 実行済み条件を確認して説明を直すか、倍率10を明示した別条件として実行してください。途中で変えて同じ条件の結果としてまとめるべきではありません。

### 8. 【minor・解釈上の制約】sampler 比較は、同じ確率分布を近似する方法だけの比較ではない

**根拠：** `config/trysvgd_tagr_tf-ula.json:64`, `:91`、`src/sb_pomdp/belief.py:942`, `:1017`, `:1023`

ULA は更新の雑音係数を \(\beta=0.2\) に固定し、drift の \(\alpha_l\) を学習します。連続時間の対応で見た実効温度は \(\beta^2/(2\alpha_l)\) なので、初期 \(\alpha=0.2\) では **0.1** です。SVGD／SMI と同じ温度の対象分布を追っているとは言えません。さらに direct／direct_joint は、一般にスカラー対数密度の勾配である保証がありません。

SMI も加法的 Langevin 雑音はありませんが、保存した noise で成分から draw して更新するので、単に「雑音なし」とすると誤解を招きます。

**対応：** 「学習された粒子更新器の比較」と記述してください。同じ対象分布に対する sampler の優劣を主張するなら、固定 energy・温度・初期分布での別検証が必要です。warm start off と learned schedule は各 score 腕内では共通ですが、結論もこの条件に限定されます。return や一段予測損失だけで、Bayesian posterior の校正や多峰性の正しさは示せません。

### 9. 【minor】評価間隔は「PPO 25更新ごと」ではない

**根拠：** `src/sb_pomdp/train.py:2431`、`src/sb_pomdp/train_baselines.py:1860`

評価の剰余判定に、pretrain／prediction を含む総 `update` を使っています。現設定では初期評価は PPO **2、47、67、87…更新後**となり、25、50、75…ではありません。

**対応：** PPO の学習曲線を意図するなら `ppo_updates_done` に基づいて評価し、ログには phase、総更新数、PPO 更新数、総環境 steps を分けて記録してください。現実装でも最終評価は実施され、同じ schedule の腕同士は比較できますが、途中評価や「末尾数回平均」の説明には実際の評価時点を使う必要があります。

---

## 依頼文(REVIEW_BRIEF.md)

# Review brief (2026-11-02)

You are reviewing a research codebase (score-based particle beliefs for POMDP reinforcement learning,
compared against GRU / RNN / DPFRL baselines). Please write your findings in Japanese, as a markdown
report with two parts, each with numbered findings ordered by severity (critical / major / minor), each
finding citing file:line and saying concretely what is wrong and what to do. Do NOT modify any file.

## Part 1. Review of a speed optimisation (proposed, applied here only in config files)

Observed on the production servers: one PPO update (16 envs x 128 steps) of the score-transformer arms
spends 4 s in `rollout_collection` and 350-2800 s in `ppo_update` (src/sb_pomdp/train.py phase timers).
The update replays the 128-step belief chain with the full autograd graph (src/sb_pomdp/ppo.py
`replay_belief_sequence`, `update_ppo`): epochs=4 x minibatches=4 (minibatch_size=512 = 4 envs) x
microbatches=4 (`sequence_microbatch_size=128` = 1 env per microbatch) = 64 replays per update, each with
tensors of size [1 env, 16 particles, 2-6 dims] -> kernel-launch / CPU bound (GPU utilisation per process
a few %).

The proposed change: in the 22 new config files `config/trysvgd_tagr_*.json`, `config/trysvgdlight_tagr_*.json`,
`config/trysvgdalpha_*.json` set `ppo.sequence_microbatch_size` 128 -> 512 (= minibatch_size, 4 envs replayed
together). Measured on an RTX 6000 Ada: ppo_update 351 s -> 163 s for config/trysvgd_tagr_tf-ula.json,
GPU memory 0.8 -> 1.2 GB. The claim is that this is a pure execution change (gradient accumulation over
independent environment sequences), mathematically identical to the 128 setting; see
src/tests/test_ppo_microbatch.py::test_sequence_microbatch_accumulation_matches_one_effective_minibatch.

Questions:
1. Verify the exactness claim by reading the code paths: `update_ppo` (loss weighting, advantage normalisation,
   metrics, gradient clipping, skipped-minibatch logic), `replay_belief_sequence` (saved noise, prefixes,
   episode resets, potential branch classification with a batch > 1), `_policy_objective`, the diffusion policy
   objective, the value / action-value heads, the prediction phases (`condition_prediction_loss`,
   `belief_mixture_prediction_loss`), SVGD (`_rbf_stein_direction`: median bandwidth per batch row) and SMI
   (`_mixture_entropy_gradient`) in src/sb_pomdp/belief.py, and the score_alpha encoder (alpha_pool) in
   src/sb_pomdp/networks.py / train.py. Anything that mixes information across the environment (batch) axis
   would break the equivalence - look for batch-wide statistics (means, medians, normalisations, LayerNorm
   over wrong axes, torch.where masks, generator consumption).
2. Any other correctness risk: resume / segmented runs (compare.py `resume_comparable_config`,
   `EXECUTION_ONLY_KEYS` in config.py), aggregation validation (aggregate.py `_validate_source_config`),
   test expectations that hard-code 128.
3. Are there further cheap, exact speedups of the same kind that we are missing (e.g. the 4-env minibatch
   could not be merged further without changing optimisation; what about the rollout, the prefix replay, the
   recondition phase, CPU-side per-step Python overhead, CUDA graphs, torch.compile, fusing the 12 Langevin
   steps, avoiding create_graph where gradient_mode allows)? Quantify expected gains when you can.

## Part 2. Review of the experimental design (is it a sound experiment?)

Read docs/ and agent/ (especially agent/experiment-validity-audit-2026-09-14.md for the house rules) and the
job READMEs jobs/trysvgd_tagr/README.md, jobs/trysvgd_alpha/README.md, jobs/trysvgd/README.md if present.

The experiment group "try-svgd" compares, with the same compute budget per acting step (FLOPs matched):
- score belief (K=16 particles, L=12 transport steps; samplers ULA / SVGD / SMI; score parameterisation
  energy (`energy`), direct (`direct`), direct with a joint context (`direct_joint`, "light" arms)) read by a
  small transformer ("TF") or by an alpha-vector policy (`score_alpha`: alpha_pool encoder + alpha_lse head,
  particle positions only, `alpha_use_scores=false`, trunk width `comparison.alpha_feedforward_dim` chosen so the
  acting FLOPs match the transformer of the same sampler and parameterisation);
- baselines GRU, RNN, DPFRL (`particle_filter`) with widths scaled to the same acting FLOPs;
- tasks: mirror2 (mountain hike mirror world), tps (hike with |s| observation + teleport), tagr (disk tag with
  range observations: noisy distance to every ghost, sigma = 0.05 + 0.2 d, no view cone, 1000 PPO updates).
- all arms share the PPO settings, a diffusion policy, prediction phases (200k random-action pretrain, then
  5 prediction updates every 25 PPO updates with `prediction_head=condition`), 5 seeds (10-14).

Please assess:
1. Fairness / confounders between arms (FLOPs matching only on the acting step vs training cost; particle
   count; the transformer's access to scores vs alpha_pool without scores; prediction head shared or not;
   baselines' prediction phases; learned alpha schedule only on score arms; warm start off).
2. Statistical adequacy (5 seeds, how evaluation is done: `evaluation` phase, stochastic vs deterministic
   readouts, how many episodes), and what claims the design can and cannot support.
3. The tagr task design (observation model, step_reward, horizon 200, latent dim 6, whether range observations
   make the task identifiable, whether 1000 PPO updates is justified, whether the baseline widths derived by
   uniform scaling are a reasonable matching).
4. The alpha arms: is matching the per-particle trunk width to the transformer meaningful when the belief
   sampling dominates FLOPs (ff 26-30 is tiny)? Does `alpha_use_scores=false` make the comparison with the
   transformer (which reads scores) uninterpretable? Is the alpha_lse head (16 linear pieces, temperature 1) a
   faithful alpha-vector policy?
5. Anything in the code that makes a stated comparison not what it claims (e.g. the oracle / true state leaking
   into learning; evaluation using training envs; seeds coupled across arms; FLOPs counter missing parts).

Where the data to verify a claim is not in the repository, say so rather than guessing.
