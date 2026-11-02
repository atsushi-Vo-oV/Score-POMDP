# Astra(GPT-6-Astra、推論 high、read-only)の提案 2026-11-02: 1 学習を GPU で速くするには

依頼文は scratchpad の PROPOSAL_BRIEF.md(本文末尾に再掲)。計測: CPU 1 スレッドの更新全体は 160 / 503 / 388 秒、GPU 単独は 225〜370 秒 → CPU 案は不成立。

**推奨は、MLP energyのscoreを解析式に置き換え、その12段transportと逆伝播を`torch.compile`で最適化することです。** ただし、解析式化だけで大幅に速くなるわけではありません。今回のCPU試作では再生＋逆伝播が**1.14倍**でした。最初の半日は、実サーバーでCPU 1スレッドとGPUを同条件で比較するのが最も判断材料になります。

ファイルは変更していません。指定された既存レビューと高速化報告を確認しました。`CLAUDE.md`は作業ツリーおよび親ディレクトリには見つかりませんでした。以下では、今回の実測、briefの測定値、未検証の見積もりを区別します。

**評価の前提**

現在は`sequence_microbatch_size=512`、`belief_activation_checkpoint=false`です。prefix／reconditionのバッチ化、不要な最終scoreの省略、`chain_entropy`も実装済みなので、これらの効果を追加高速化に重複計上しません。以前のレビューで指摘された予測損失の分割依存も、現在は系列ごとの平均になっています。  
根拠：`config/trysvgd_tagr_tf-ula.json:93`、`:101`、`agent/speedup-20261102.md:7`、`src/sb_pomdp/ppo.py:542`。

また、**ATen演算数とCUDA kernel起動数は同じではありません**。`agent/opcount.py:10`はdispatch回数を数えており、viewなども含みます。約3252演算×2048ステップ＝約666万ATen呼び出し／更新という規模感は有用ですが、実kernel数はGPU profilerで確認する必要があります。

追加最適化の優先順位は次のとおりです。倍率は互いに重なるため、掛け合わせません。

| 順位 | 提案 | 速度改善の見込み・根拠 | 試作工数 | 等価性 |
|---|---|---|---|---|
| 1 | 解析score＋`torch.compile` | 解析式単独は今回CPUで**1.14倍**。GPUでPPO全体**2〜5倍を検証目標**にする。ただし対象時間の割合次第 | 3〜5日 | 数学は同じ。浮動小数点差あり |
| 2 | CPU 1スレッドでrun全体を実行 | briefの単ステップ比較ではGPU比**2.1〜4.3倍**。更新全体は未確認 | 0.5〜1日 | 数学は同じ。丸め・乱数列が変わる |
| 3 | CUDA Graphsでforward／backwardをcapture | CPU dispatchが支配的なら有力。対象部分の**2〜5倍を探索範囲**とするが実測必須 | 単ステップ1日、系列対応3〜7日 | 同じkernelならbitwise維持の候補。保証は要検証 |
| 4 | 不変計算の外出し・cat／index／同期削減 | 単独では小幅。仮に時間の5〜20%を除去できれば**1.05〜1.25倍** | 1〜3日 | 多くは浮動小数点順序の差 |
| 5 | 手書きbackward／Triton融合 | 1〜4後の残存kernel時間が上限。現時点では信頼できる全体倍率を出せない | 1〜3週間 | 数学的等価を狙えるが、勾配実装リスクが大きい |

**1．解析score＋コンパイル：GPUでの本命**

現状は、各score評価で`autograd.grad(create_graph=True)`を実行し、PPOのbackwardでその微分グラフをさらに微分しています。解析式化は、この内側のautograd実行を通常のtensor演算に置き換えます。**数学的に必要な二階微分相当の計算は残りますが、autogradの入れ子を除けます。**  
根拠：`src/sb_pomdp/belief.py:51`、`:623`、`:936`。

対象の`energy_hidden=[8]`はLinear–SiLU–Linearです。実装上のネットワーク出力はlog-potentialなので、符号は次のようになります。

\[
q(x,c)=w^\top\operatorname{SiLU}(W_xx+W_cc+b)+b_2
\]

\[
\nabla_xq
=W_x^\top\left[
w\odot\{\sigma(h)+h\sigma(h)(1-\sigma(h))\}
\right],\qquad h=W_xx+W_cc+b
\]

initialとrecursiveのscoreは、

\[
s_{\rm initial}(x)=\nabla_x g_0(o,x)-x
\]

\[
s_{\rm recursive}(x)
=\nabla_x f(o,x,a)
+\sum_j\operatorname{softmax}_j(u)\,\nabla_xu(x,p_j,a)
\]

です。`logmeanexp`の\(-\log K\)は微分で消えます。**softmax重み、過去粒子、各weightへの計算グラフは残す必要があります。** 重みをdetachしたり単純平均に置き換えたりすると、同じPPO勾配にはなりません。  
根拠：[belief.py:480](C:/Users/Atsushi/AppData/Local/Temp/claude/c--Users-Atsushi-workspace-RL-Belief-Score-Based-Model/fa2c88fb-7178-4a9d-936c-3837421f8d69/scratchpad/codex_speed2/src/sb_pomdp/belief.py:480)、`src/sb_pomdp/belief.py:491`。

今回、ファイルを書かず、上式をメモリ内で差し替えて測定しました。

| CPU、PyTorch 2.3.1、1スレッド | 現実装 | 解析式 | 倍率 |
|---|---:|---:|---:|
| recursive、勾配追跡なし | 6.213 ms | 4.845 ms | **1.28倍** |
| recursive、再生＋backward | 13.109 ms | 11.485 ms | **1.14倍** |
| 再生＋backwardのATen数 | 3250 | 2561 | **約21%減** |

条件はB=4、K=16、D=6、L=12、最終scoreあり。warmup後、10反復×5ブロックの中央値です。backwardの損失は`agent/opcount.py:36`と同じ`particles.sum()+scores.sum()`で、Transformer・prefix・PPO更新は含みません。

別途、initial／recursive／mixedについて、粒子・scoreの二乗平均からの勾配を比較しました。最大絶対差は粒子**5.96×10⁻⁷**、score **2.98×10⁻⁷**、過去粒子とパラメータの勾配**4.77×10⁻⁷**でした。単一初期値・単ステップの確認なので、128ステップやAdam更新の許容差を満たす証明ではありません。

適用範囲は次のとおりです。

| 経路 | 対応 |
|---|---|
| `energy`＋MLP `[8]` | 最初の対象。initial／recursive／mixedすべて対応可能 |
| 任意の多層`energy_hidden` | 層ごとの連鎖律で一般化可能。フルJacobianの構築は不要 |
| `direct`／`direct_joint` | score自体は既に通常のforward。解析energy化の利益はないが、コンパイル対象になる |
| `energy_network_kind="kan"` | SiLU式だけでは不可。B-splineの微分を含む別実装が必要 |
| ULA | energy scoreを置換すれば、内側の主要な高階autogradを除去できる |
| SVGD | energy scoreは置換可能。RBFの引力・斥力は既に解析式 |
| SMI | energy scoreに加え、mixture entropyの`autograd.grad`も置換する必要がある |

根拠：`src/sb_pomdp/networks.py:14`、`:210`、`src/sb_pomdp/belief.py:364`、`:552`、`:92`、`:116`。

SMIでは、混合密度のscoreだけでは不十分です。再パラメータ化したdrawと、全混合成分への直接依存の両方を含めます。\(z_i=\theta_i+s_i\epsilon_i\)、\(r_{ij}\)を成分責任度、\(v_i=\nabla_{z_i}\log q(z_i)\)とすると、現在の実装が微分する\(F=\sum_i\log q(z_i)\)について、

\[
\nabla_{\theta_j}F
=v_j+\sum_i r_{ij}\frac{z_i-\theta_j}{s_j^2}
\]

\[
\nabla_{\log s_j}F
=v_j\odot s_j\epsilon_j+
\sum_i r_{ij}
\left(\frac{(z_i-\theta_j)^2}{s_j^2}-1\right)
\]

となります。entropy項はこれらの負号です。責任度もdetachしません。今回の小型CPU確認ではfloat64の値差は\(8.9\times10^{-16}\)、外側勾配差は\(2.9\times10^{-14}\)でした。SMI全chainの速度・一致は未測定です。

SVGDではmedian bandwidthをdetachする現仕様を維持します。ここまで微分する変更は、現在の学習勾配を変えます。  
根拠：`src/sb_pomdp/belief.py:103`、`:129`、`:1028`。

コンパイルの最初の境界は、**B=4のrecursive・12段transport＋最終score**が適切です。

- branch分類、shape検証、prefixのPython処理は境界の外に置く。
- `fullgraph=True`でgraph breakを顕在化させ、forwardだけでなくbackwardまで確認する。
- 初期段階はCUDA Graphsを無効にし、Inductorの効果を単独測定する。
- 成功後にinitial／mixed、prefix用B=1〜4、`need_scores=False`を追加する。
- CPUでも同じ解析式版をコンパイルして比較する。

現在のPyTorchは2.3.1固定です。energy MLPだけをコンパイルして、その出力を従来どおり二重微分する方法にはAOTAutogradの制約があります。ローカル2.3.1の実装にもdouble-backwardを拒否する処理があり、公式issueでも確認できます。[PyTorchの該当issue](https://github.com/pytorch/pytorch/issues/91469)

checkpointは対象設定で既に無効です。有効化による追加高速化はありません。解析式化後にメモリ不足が生じた場合だけ、再計算コストを含めて再評価します。  
根拠：`pyproject.toml:13`、`src/sb_pomdp/ppo.py:245`、`:291`。

**2〜5倍という目標の条件**は、最適化対象が更新時間の90%を占め、その部分が3〜10倍になる場合です。

\[
S_{\rm update}=\frac{1}{(1-p)+p/S_{\rm target}}
\]

\(p=0.9\)なら約2.5〜5.3倍ですが、prefixを対象外にして\(p=0.5\)なら、対象が5倍でも更新全体は**1.67倍**です。したがって、main replayだけでなくprefixの時間割合が重要です。

**2．CPU実行：最初に比較する価値が高い**

briefの18.7 ms対40〜80 msでは、CPUが単ステップで**2.1〜4.3倍**速い計算です。2048ステップ分はCPU約38秒、GPU約82〜164秒になります。ただし、これはprefix・readout・optimizerなどを含まない外挿であり、160〜400秒の更新時間との直接比較はできません。  
根拠：`PROPOSAL_BRIEF.md:12`、`:22`、`agent/opcount.py:33`。

最初はモデル、rollout buffer、optimizerをすべてCPUに置いて比較します。beliefだけCPUに移し、時刻ごとにGPUと往復する構成は、転送・同期の影響を分離しにくいため後回しにします。

`torch.set_num_threads(1)`を基準に、別プロセスで1／2／4スレッドを比較してください。CPUコア数を増やしても、Pythonによる逐次dispatchは並列化されません。一方、512サンプルをまとめたTransformer部分は複数スレッドが有利な可能性があるため、単ステップだけで決めません。

既存ジョブには通常GPU用16スレッド、MIG用`nproc`、CPU用既定30スレッドの設定があります。今回の小型scoreモデルでは、そのまま最適とは限りません。  
根拠：`jobs/genkai_production.sh:18`、`jobs/genkai_production_cpu.sh:21`、`src/sb_pomdp/ppo.py:795`。

**等価性の注意：** 同じ保存noiseによるreplay比較は可能ですが、CPUとCUDAでは同じseedでも新規生成するnoiseやminibatch permutationが一致するとは限りません。run全体では「同じ数学・異なる乱数列」として扱います。  
根拠：`src/sb_pomdp/belief.py:728`、`src/sb_pomdp/ppo.py:764`。

**3．CUDA Graphs：効果は期待できるが、full BPTTの保存領域が難所**

CUDA Graphsはkernelを融合する機能ではありません。既存のkernel列をまとめて投入し、Python／C++／driverの起動処理を省きます。今回の症状には適合します。[PyTorch 2.3.1のCUDA Graphs仕様](https://raw.githubusercontent.com/pytorch/pytorch/v2.3.1/docs/source/notes/cuda.rst)

保存済みnoiseを使う`particles_from_noise`は有利です。capture内でGeneratorを動かす必要がありません。ただし、`.item()`を含むbranch分類は外へ出し、固定shape・固定stride・固定アドレスを維持します。12回の固定Pythonループはcapture時に展開できますが、入力に応じたPython分岐はreplayされません。  
根拠：`src/sb_pomdp/belief.py:443`、`:705`、`:783`、`:920`。

最大の注意点は、**同じgraphed callableを128回forwardし、後から一括backwardする使い方**です。2.3.1の`make_graphed_callables`は固定input／output／中間領域を使うため、単純な再利用では過去時刻の保存値を上書きする危険があります。出力のcloneだけでは内部の保存値を保護できません。

系列全体のcapture、時刻ごとの保存領域、または再計算を組み込んだbackwardが必要です。この判断は固定領域を持つ公式実装に基づきます。[PyTorch 2.3.1の実装](https://raw.githubusercontent.com/pytorch/pytorch/v2.3.1/torch/cuda/graphs.py)

同APIの返すcallableは外側からの高階微分に対応しません。ただし、これを「`create_graph=True`を含む処理はmanual captureも不可能」と一般化するべきではありません。必要な微分を含むforward／backward全体をcaptureできるかは、小型例で確認できます。解析式化はこの検証を簡単にします。

試作順は、単ステップforward＋backward、続いて**異なる入力の2時刻を接続してから一括backward**です。そこで過去粒子・全パラメータの勾配を比較し、保存領域の問題を先に検出します。最初からoptimizerや非有限勾配時のskipまでcaptureしません。  
根拠：`src/sb_pomdp/ppo.py:414`、`:910`。

純粋なcaptureはbitwise一致を狙う候補ですが、非決定的kernelやbuffer管理を含む検証が必要です。compileとの併用では浮動小数点順序も変わり得ます。

**4．計算の外出し・演算整理：コンパイルと組み合わせる**

最も有望なのはtransitionネットワークの最初のLinearです。現在は全粒子対について、

\[
h_{ij}=W_xx_i+W_pp_j+W_aa+b
\]

を、連結した`[B,K,K,2D+A]`入力から計算しています。これを分解すれば、`W_p p_j`と`W_a a`は12段のtransport中に再利用でき、`W_x x_i`も粒子対ごとの重複計算を避けられます。  
根拠：`src/sb_pomdp/belief.py:505`。

B=4、K=16、D=6、A=2、幅8なら、最初のLinearの乗算加算数は、単一score評価について約114,688から6,208へ減ります。ただし、これは**そのLinear部分だけの約18倍削減**です。broadcast加算、SiLU、softmax、score、backwardは残り、eagerでは小さいmatmulが増えて遅くなる可能性もあります。

再利用するcontextには勾配を残し、同じ時刻・同じパラメータでの12段内に限定します。optimizer更新後やepochをまたぐ計算結果の使い回しは等価ではありません。

その他は次の優先順です。

- observation側の不変projection、schedule、prefixのindex／stack準備を外へ出す。
- 全recursiveで不要なobservation anchor計算を省けるか確認する。ただし、unused parameterの`.grad=None`とゼロ勾配の違いによるAdam状態への影響も検査する。
- SVGDの`einsum("bij,bjd->bid")`と`bmm`を比較する。既に同等にloweringされる可能性があり、置換だけの高速化は保証しない。
- metricsのCPU転送をまとめる。主にminibatch単位なので、12段chainの改善より優先度は低い。

根拠：`src/sb_pomdp/belief.py:111`、`:862`、`:903`、`src/sb_pomdp/ppo.py:393`、`:877`。

resetの`where`は既に環境時刻ごとに1回です。mixed branch内の`where`は更新後のqueryごとに必要なので、「すべて1回にまとめる」変更は単純にはできません。  
根拠：`src/sb_pomdp/ppo.py:417`、`src/sb_pomdp/belief.py:550`。

**5．手書きbackward／Triton：残ったボトルネックに限定する**

ULAの1段を

\[
x'=x+\alpha s_\theta(x,p,o,a)+\beta\epsilon
\]

とすると、backwardには少なくとも、

\[
\bar x=\bar x'+\alpha(\partial_xs)^\top\bar x',
\qquad
\bar p=\alpha(\partial_ps)^\top\bar x'
\]

に加え、energyパラメータ、学習schedule、anchor／proposalへの勾配が必要です。最終scoreがreadoutに使われる経路も含めます。

Pythonの`autograd.Function`に包むだけではkernel数は減りません。backwardの演算整理やCUDA／Triton融合が必要です。まず解析score＋ULA更新の1段を融合し、それでも起動時間が支配的なら12段へ拡張します。12段には逐次依存があり、SVGD／SMIでは各段で粒子間通信も必要です。  
根拠：`src/sb_pomdp/belief.py:959`、`:970`、`:1023`。

期待倍率は、1〜4を適用した後のprofileから決めます。例えば残存時間の50%しか対象にできなければ、その部分を無限に速くしても全体上限は2倍です。現時点で「Tritonなら10倍」とする根拠はありません。

**数学・学習条件を変える案は別枠で評価する**

| 案 | 計算時間への効果 | 変更されるもの |
|---|---|---|
| epochs 4→2 | replay中心の更新費用は概ね半分 | optimizer更新回数・学習軌道 |
| `full`→`tbptt_1` | 時間方向のbackwardとメモリを削減。倍率は未確定 | 過去beliefへの勾配 |
| replay履歴の短縮 | prefix費用次第で大きく減る | prefixを省けば再構築状態、detachすれば勾配 |
| L=12→6 | score評価は13→7回。該当部分の単純上限は約1.86倍 | 粒子更新器そのもの |
| scoreのdetach／`create_graph=False` | 微分処理を削減 | energyや過去粒子への必要な勾配 |
| FP16／BF16／TF32 | 起動律速のままでは効果限定的 | 数値精度・学習軌道 |

`tbptt_1`が切るのは環境時刻の境界です。各時刻の12段内部には勾配が残るため、これを選んでもscore学習用の微分を一律には外せません。  
根拠：`src/sb_pomdp/belief.py:800`、`:841`、`:943`。

低精度化はsoftmax、SMIのscale、diffusionのlog-probabilityを含めて確認します。既存のdiffusionバッチ化でも約\(10^{-5}\)の差が生じて不採用となっており、今回の解析式や融合にも同じ等価性基準を適用すべきです。  
根拠：`agent/speedup-20261102.md:20`。

また、4環境の論理minibatchを16環境へまとめる変更はoptimizerの更新単位を変えます。512へのmicrobatch拡大と同じ意味の実行最適化にはなりません。  
根拠：`src/sb_pomdp/ppo.py:732`、`:769`、`:922`。

**最初の測定と、日単位の実施順序**

**最初の測定は、同じ保存rollout・初期モデル・optimizer状態・minibatch順序による、CPU／GPUのPPO更新全体比較です。** 新たなrolloutを各実装で別々に生成すると、episode長やprefix長が変わり、速度差の解釈が難しくなります。

| 日程 | 実施内容 | 判断材料 |
|---|---|---|
| 0〜0.5日 | XeonでCPU 1／2／4スレッド、GPU単独、現行GPU共有状態、MIGを比較 | 実更新時間、競合の影響、最適thread数 |
| 1日目 | energy MLP `[8]`の解析scoreを実装・比較 | 出力、過去粒子勾配、全parameter勾配、Adam状態 |
| 2〜3日目 | B=4 recursiveの12段をcompile。CUDA／CPUを比較 | warmup後速度、graph break、再compile、ピークメモリ |
| 4〜5日目 | initial／mixed／prefixへ拡張。不変projectionを追加 | 実際のPPO・prediction更新全体の倍率 |
| 6日目以降 | 起動律速が残る場合にCUDA Graphs | 2時刻→128時刻の勾配一致、保存領域、capture費用 |
| その後 | profilerが示す部分だけcustom kernel化 | compile／graphs後に残る費用に見合うか |

計測では、main replay、prefix、backward、readout／loss、optimizerを分け、reset頻度とprefix長も記録します。GPUは同期付きwall timeとCUDA Eventの両方を取り、短いprofiler実行でkernel数とCPU launch間の空隙を確認します。CUDA Eventだけから純粋な演算時間を推定しないことが重要です。[PyTorchの非同期実行と計測方法](https://raw.githubusercontent.com/pytorch/pytorch/v2.3.1/docs/source/notes/cuda.rst)

正しさの検証には既存のprefix／reset／Adam比較を利用し、今回の変更経路を追加します。特に**128ステップ、非空prefix、mixed reset、PPOとpredictionの両目的、更新後パラメータ、保存noiseとRNG状態**を確認します。既存テストの成功件数は過去報告のものであり、今回再実行したものではありません。  
根拠：`src/tests/test_speedup_exactness.py:256`、`:331`、`src/tests/test_belief_sequence_invariants.py:305`。

採用判断は、単ステップの最高倍率ではなく、**コンパイル／captureの初期費用を含む1 runの総時間と、既存の等価性基準を満たすか**で行います。

---

## 依頼文

# Brief: how can ONE training run of the score belief be made much faster on a GPU? (2026-11-02)

Read-only. Do not modify files. Answer in Japanese as a markdown report: ranked proposals, each with the
mechanism, the expected speedup (with your reasoning / numbers), the implementation cost, the risk to exactness
(bitwise / up to float reordering / different random stream / different maths), and what to prototype first.
Cite file:line. Where you are unsure, say so; where a quick measurement would settle a question, say which.

## Context (read agent/codex-review-20261102.md and agent/speedup-20261102.md: your earlier review and the
exact speedups implemented today - prefix / recondition batching, need_scores, chain_entropy; and
ppo.sequence_microbatch_size 128 -> 512)

Workload per PPO update (16 envs x 128 steps, config/trysvgd_tagr_tf-ula.json): rollout 4 s; ppo_update
160-400 s = 16 belief-sequence replays (4 epochs x 4 minibatches of 4 envs) of 128 time steps, each time step =
12 Langevin transport steps of K=16 particles in R^6 driven by an energy MLP [8] (score = autograd.grad of the
energy with create_graph=True so PPO can differentiate through the whole chain), then a transformer readout.
The servers: RTX 6000 Ada + Xeon Gold 6326 (16 cores), and Genkai MIG slices; one process uses one CPU core,
GPU compute utilisation a few %. The time is kernel-launch / Python-dispatch bound:

    agent/opcount.py (CPU, batch 4, tagr TF-ULA):
    rollout step (no grad): 1362 aten ops            -> ~110 ops per Langevin step
    replay step: forward 1506 + backward 1746 = 3252 ops
    CPU (1 thread, Windows laptop): rollout step 10.1 ms, replay step fwd+bwd 18.7 ms
    GPU (RTX 6000 Ada, shared by 3-4 processes): rollout step ~31 ms, replay step ~40-80 ms  (slower than CPU!)

So a run is ~1300 updates x 2048 replayed time steps x ~3000 tiny kernels. Model batching across seeds (vmap)
would raise throughput but not shorten one run; we want to shorten one run.

## Candidate ideas to evaluate (add your own)

1. Analytic score for the MLP energy: write grad_x E(x) of Linear-SiLU-Linear(-sum) explicitly as forward ops
   (and the SMI mixture-entropy gradient, the SVGD kernel terms), removing autograd.grad(create_graph=True)
   from the inner loop; first-order autograd then gives PPO the same gradients. Fewer ops, no double-backward
   graph, and it unlocks torch.compile / CUDA graphs. Which parameterisations (energy / direct / direct_joint,
   `energy_hidden`, `network_kind` KAN?) and branches (initial / recursive / mixed, the log-mean-exp transition
   energy over K particles) does this cover? See src/sb_pomdp/belief.py.
2. torch.compile (inductor) of the per-time-step belief update (12 Langevin steps fused), on CUDA and on CPU;
   obstacles: double backward (see 1), dynamic shapes (prefix batching, episode resets), the non-reentrant
   checkpoint path, torch 2.3.1.
3. CUDA Graphs (torch.cuda.make_graphed_callables or manual capture) for the static-shape step; obstacles:
   create_graph, generator-based noise (already saved as tensors), the Python control flow per step.
4. Running score models on CPU instead of GPU (per the numbers above the CPU may already be faster per
   process); and the `torch.set_num_threads` / OMP question for such tiny tensors.
5. Reducing op count by restructuring: cat/stack elimination, fused Langevin update, fewer `.to()` / index ops,
   one `where` per step for resets, avoiding `torch.einsum` for tiny shapes, custom autograd.Function for a
   Langevin step with a hand-written backward (exact), Triton kernel for the 12-step chain.
6. Algorithmic (changes the maths - list separately): shorter replay (tbptt), fewer epochs, truncating the
   score graph, lower-precision.

Deliverable: the ranked list, a recommended order (prototype in days), and the measurement you would run first.
