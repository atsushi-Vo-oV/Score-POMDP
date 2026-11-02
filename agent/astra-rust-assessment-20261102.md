# Astra の評価 2026-11-02: Rust で書き直すと速くなるか

**Rustへの書き換えは可能ですが、同じ演算を tch-rs に移すだけでは、1 run は大幅には速くなりません。推奨は引き続き、Python／PyTorchを維持して「解析score → `torch.compile` → 必要ならCUDA Graphs」の順に進めることです。** その後も残る処理に限って、CUDA／Tritonの融合kernelと手書きbackwardを検討します。

指定資料とコードを確認しました。`CLAUDE.md`はリポジトリ内・親ディレクトリに見つかりませんでした。ファイル変更・新規ベンチマークは行っていません。以下の実測値はbriefと既存報告によるもので、速度・工数の予測とは区別しています。

**1．Rustで削れる時間、削れない時間**

今回のGPU更新時間は **225〜370秒**、CPU 1スレッドは **503秒**です。GPUは更新全体でCPUの **1.36〜2.24倍**速く、前回提案にあった「小さいbelief計算ならCPUが有利」という単ステップ測定を、run全体へ外挿することはできません。

主系列だけでも、1 PPO更新あたり、

\[
16\times128=2,048\text{ belief時刻}
\]

\[
2,048\times12=24,576\text{ Langevin段}
\]

\[
2,048\times3,250=6,656,000\text{ ATen演算}
\]

となります。最終scoreまで含めると、energyの内側微分は概ね **26,624回**です。prefixは別途追加されます。

ただし、`agent/opcount.py`は **ATen dispatch数**を数えています。viewなどの非kernel演算も含むため、666万回をCUDA起動数と解釈してはいけません。また、外側backward内の多くの演算は、既にC++ autogradから呼ばれています。

Xeon Gold 6326／RTX 6000 Adaでの内訳は未測定です。計画用には、小さいCUDA演算のホスト側経路を次の程度に仮置きします。

| 費用 | 仮の規模 | Rust＋tch-rsへの移植 |
|---|---:|---|
| Python呼び出し、引数処理、Tensorラッパー | Python経由の演算あたり **1〜4 µs** | 大部分を削減。ただしRust側FFI・ラッパー費用は残る |
| ATen dispatch、autograd記録、割当・参照管理 | 記録を伴う演算あたり **2〜6 µs程度** | 基本的に残る |
| CUDA driverへの投入、起動までの間隔など | kernelを起動する演算あたり **数〜20 µs程度** | 基本的に残る |
| GPU上の実計算 | 演算ごとに異なる | 同じkernelなら変わらない |

これはbriefの「演算ごとに10〜30 µs程度」という症状に対応する**仮説**であり、当該機で分離測定した数字ではありません。非同期実行の重なりもあるため、各行をそのまま足した値がwall timeになるわけではありません。

例えば、666万演算の40%がPython経由で、その費用を2 µsずつ削れた場合、主系列の短縮は **約5.3秒**です。全演算から4 µs削れるという過大な仮定でも約26.6秒であり、これだけで250秒が25秒にはなりません。prefixにも効果はありますが、その割合は別途測る必要があります。

したがって、**同じ演算構成のRust移植は、run全体で0〜13%程度の時間短縮、約1.0〜1.15倍を計画値**とします。20%短縮でも楽観側です。Rust化だけで2倍を期待する根拠はありません。

`create_graph=True`の費用も同様です。現在はscoreを作る内側backwardを記録し、そのグラフをPPOのbackwardが微分します。tch-rsなら同じlibtorch autogradが働くため、ノード構築・保存Tensor・必要な二階微分相当の処理は残ります。独自Rust autogradで管理費を減らせる可能性はありますが、それは微分エンジンの開発です。

一方、解析scoreなら内側のautograd呼び出しを通常の演算へ置換できます。前回報告ではCPUの再生＋backwardが **13.109→11.485 ms、1.14倍**、ATen数が **3,250→2,561、約21%減**でした。これはRustの効果ではなく、計算の表現を変えた効果です。

なお、briefの「起動遅延を除くには融合だけ」という点は補足が必要です。**融合はkernel数を減らしますが、CUDA Graphsも個々のCPU投入費用をまとめて削減できます。** Graphsはkernelを融合せず、記録した列を単一の`cudaGraphLaunch`で投入します。[PyTorchのCUDA Graphs説明](https://docs.pytorch.org/docs/main/notes/cuda.html)

**2．A〜Dの比較：1 runの速度、工数、保守**

以下は対象のscore-belief runについての未検証の見積もりです。工数は、このコードとGPU開発に習熟した担当者1人を想定しています。

| 案 | 1 runの速度見込み | 工数の目安 | 等価性・保守 |
|---|---|---|---|
| **A：全面tch-rs移植、演算は維持** | **1.0〜1.15倍** | **40〜80人日／8〜16週** | 同じlibtorchなら比較しやすいが、モデル・optimizer・RNG・resume互換の移植が必要。保守負担は高い |
| **A：全面Burn／Candle移植** | 現状の高階微分をそのまま移せず、**有効な倍率は未確定**。計画上は速度利益を計上しない | **80〜160人日以上／16〜32週以上** | 解析scoreや独自backward、演算互換検証が先に必要。保守負担は非常に高い |
| **B：解析score＋融合CUDA＋手書きbackward** | 対象がrun時間の80〜90%なら、**約2.8〜6.9倍を探索範囲**とする | ULA限定 **10〜20人日**。SVGD／SMI等込み **20〜40人日以上** | 数学的等価を狙える。勾配・数値差・CUDAビルドの保守が重い |
| **B：同じ処理をTritonで実装** | 融合範囲が同じならCUDA版と同程度の探索範囲。言語だけでは優劣を決められない | ULA限定 **7〜15人日**。拡張・検証込み **15〜30人日以上** | C++連携は軽くなるが、backwardと精度の検証は必要 |
| **C：解析score＋compile＋必要なGraphs** | **2〜5倍を検証目標** | **5〜10人日**。系列captureの難所があれば追加 | 既存PyTorch資産を維持。保守負担は低〜中 |
| **D：環境／rolloutのRust化** | **ほぼ1.0倍。上限でも1.016倍** | 1環境 **3〜7人日**、全環境と互換性対応 **10〜20人日以上** | NumPy乱数・境界条件・保存状態の再現が追加負担 |

BとCの倍率は実測予測ではなく、次の条件付き計算です。

\[
S_{\rm run}=\frac{1}{(1-p)+p/s}
\]

ここで \(p\) はrun全体の対象時間割合、\(s\) は対象部分の高速化倍率です。例えば \(p=0.8〜0.9\)、Bの対象部分が5〜20倍なら、全体は約2.8〜6.9倍です。**対象が半分しかなければ、無限に速くしても全体は2倍が上限**です。BとCの倍率は重なるため掛け合わせません。

代表的な250秒の周期なら、Cの目標は50〜125秒です。ただし、compile／captureの初期費用を含めて、

\[
T_{\rm new}=T_{\rm setup}+T_{\rm unaffected}+T_{\rm target}/s
\]

で判断します。

対象設定にはPPO **1,000更新**に加えて、prediction pretrain **98更新**、interleaved prediction **195更新**があります。PPO部分だけでも現在の225〜370秒を外挿すると **62.5〜102.8時間**です。prediction・recondition・評価・保存もあるため、PPO単体の倍率をそのままrun全体の倍率として報告すべきではありません。

**A：Rustライブラリの適合性**

| 項目 | tch-rs／libtorch | Burn | Candle |
|---|---|---|---|
| CUDA | 対応 | 対応 | 対応 |
| Transformer | ATen演算を使って現構成を再現可能 | 基本演算を組み合わせて再現可能。既定層の同等性は要確認 | Transformer実装・基本部品あり。学習経路の対応確認が必要 |
| このリポジトリのdiffusion policy | MLP、正規分布式、保存chain評価を移植 | 同左。既製の画像diffusionモデルでは代替できない | 同左 |
| `logsumexp` | 対応 | 基本演算による安定な合成も可能 | 基本演算による安定な合成も可能 |
| `autograd.grad(create_graph=True)`相当 | `run_backward(..., keep_graph, create_graph)`あり | 公式autodiffは一階微分のみ | 通常backwardは途中の勾配をdetach。直接の互換APIとみなせない |
| 現行energy経路のdouble-backward | 必要演算が対応していれば可能 | そのままの移植は不可 | 検証済みの移植前提にはできない |

tch-rsの`create_graph`は公式ソースで確認できます。Burnは一階微分のみと明記されています。Candleにはdetachを抑制する環境変数もありますが、それだけで必要な全演算・中間query・高階微分の互換性が保証されるわけではありません。[tch-rsの実装](https://docs.rs/tch/latest/src/tch/wrappers/tensor.rs.html)、[Burn autodiff](https://docs.rs/burn-autodiff/latest/burn_autodiff/)、[Candle backwardの実装](https://docs.rs/crate/candle-core/latest/source/src/backprop.rs)

Burn／Candleでも、解析scoreと解析SMI entropyを通常のforwardとして書けば、PPOには一階のreverse-modeだけで足りる構成にできます。しかし、その作業はPython側でも可能です。Burnの融合で速くなる場合も、利益の中心は融合です。[BurnのCUDA・融合対応](https://docs.rs/burn/latest/burn/)、[CandleのCUDA対応](https://github.com/huggingface/candle)、[Candleの演算・モデル部品](https://docs.rs/candle-nn/)

全面移植では、Transformerのpre-norm、LayerNormのepsilon、GELU、dropout、初期化、Adamの更新式まで合わせる必要があります。`train.py`のcheckpointにはモデル重みだけでなく、optimizer、belief履歴、環境状態、複数のRNG状態が入っています。「モデルがRustで動いた」は、この移植の完了条件にはなりません。

**BとC：融合する対象と実装順**

最初の対象は、`energy_hidden=[8]`の **initial／recursive scoreと12段ULA、必要な最終score**です。

Linear–SiLU–Linearのlog-potentialを

\[
q(x,c)=w^\top\mathrm{SiLU}(W_xx+W_cc+b)+b_2
\]

とすれば、

\[
\nabla_xq=W_x^\top\left[w\odot
\{\sigma(h)+h\sigma(h)(1-\sigma(h))\}\right]
\]

で計算できます。現在の符号に合わせ、initialではさらに \(-x\)、recursiveではobservation項とtransition項を加えます。

まずこの式を通常のPyTorch演算で実装し、12段をcompileします。energy MLPだけをcompileして従来の二重微分を外側からかける方法は、固定されているPyTorch 2.3.1のAOTAutograd制約に注意が必要です。[PyTorchのdouble-backward issue](https://github.com/pytorch/pytorch/issues/91469)

その後、起動待ちが残ればGraphsを追加します。branch分類や`.item()`を外へ出し、shapeと保存領域を固定します。特に、**同じcaptureを128回forwardしてからまとめてbackwardする場合、過去時刻の中間値を上書きしてはいけません**。系列全体のcapture、時刻別領域、またはbackward時の再計算が必要です。2.3.1の`make_graphed_callables`自体にも高階微分の制約があります。[PyTorch 2.3.1の実装](https://raw.githubusercontent.com/pytorch/pytorch/v2.3.1/torch/cuda/graphs.py)

手書きkernelは、まずscore＋ULAの1段を融合し、その後12段の融合を判断します。B=4では環境ごとに1 blockだけを割り当てるとGPUを使い切れません。粒子単位の並列化、パラメータ勾配の還元、register使用量を含めて比較する必要があります。SVGD／SMIでは各段に粒子間通信があり、12段全体を一つにすれば必ず速い、とはいえません。

**D：環境移植の上限は小さい**

rollout全体が250秒中4秒なら、それを完全に消しても、

\[
250/(250-4)=1.0163
\]

です。10倍にしても約1.0146倍にとどまります。

しかも4秒にはbelief推論、方策推論、GPUとの転送・同期も含まれます。NumPy環境だけの移植対象はさらに小さく、Rust環境にしてもGPUから行動を受け取る同期は残ります。今回のrun高速化のためには優先しません。

**3．現在の許容差を維持できるか**

**維持を採用条件にできますが、融合前に保証はできません。** 同じ数学でも、還元順序、FMA、指数関数の実装、行列積の分割が変わります。12段×128時刻を通じて差が増幅される可能性があります。

現行の`test_speedup_exactness.py`では、次の基準です。古い報告の「Adamも1e-5」より、現行テストを優先します。

| 比較対象 | 現行の許容差 |
|---|---|
| 粒子・score | `atol=rtol=1e-6` |
| 勾配 | `atol=rtol=1e-5` |
| Adam更新後パラメータ | `atol=1e-4, rtol=1e-5` |
| Adam内部状態 | `atol=1e-6, rtol=1e-5` |

特に注意が必要な箇所は次のとおりです。

- **transitionの粒子softmax**：scoreは \(\sum_j w_j\nabla_xu_j\)、\(w_j=\mathrm{softmax}(u)_j\)です。外側backwardでは \(w_j\) の微分も必要です。重みのdetach、単純平均、最大成分への置換は不可です。安定なmax-shiftを使っても、既存`logsumexp` backwardとの丸め差は検査します。
- **SVGD median bandwidth**：対角を除く \(K(K-1)\) 個の距離を使い、現行`torch.median`と同じ偶数個時の選択規則を保ちます。K=16なら240個です。帯域幅のdetach、`clamp_min(1e-6)`、K=1の分岐も維持します。距離計算の丸めで順位が変わるケースを検査します。
- **SMI mixture entropy**：再パラメータ化したdrawへの依存と、全混合成分への直接依存の両方が必要です。混合密度のscoreだけでは不足します。責任度の微分、log-scale更新、scaleのclampも含めます。energy scoreだけ解析化しても、entropyの内側autogradは残ります。
- **clampとAdam**：clamp境界の微分規則を合わせます。また、小さい勾配の符号差をAdamが増幅する場合があり、勾配一致だけでは更新パラメータの一致を保証できません。
- **diffusion log-probability**：既存報告では、denoiserのバッチ化だけで最大絶対差 **1.07×10⁻⁵**となり不採用でした。belief粒子が許容差内でも、方策log-probabilityとPPO ratioまで確認する必要があります。

初版はFP32、現在と同じ精度設定で作り、fast-mathや低精度化を追加しません。必要なら還元順序を参照実装に近づけ、一部の融合を取りやめます。FP64蓄積も診断には有用ですが、FP32参照結果への近さを必ず改善するとは限りません。

これらは保存データに対する数値的等価性です。長いオンライン学習で、同じseedから最終報酬まで一致する保証ではありません。

**4．拡張を作る場合の具体的な境界**

初版はULA＋幅8 MLP専用とし、既存の`particles_from_noise`を外側の互換インターフェースとして残します。内部の融合関数は次の形が適切です。

```text
ula_chain(
    x0:          [B, K, D],
    previous:    [B, Kprev, D],
    observation: [B, O],
    action:      [B, A],
    initial:     [B] bool,
    noise:       [B, L, K, D],
    alpha:       [L],
    beta:        [L],
    *energy_parameters,
    branch, score_clip, particle_clip, need_scores
) -> particles [B, K, D], scores [B, K, D] | None
```

- `x0`は既存Python側で保存initial noise、observation anchor、warm-start／proposalから作ります。これにより、拡張が返す`grad_x0`を既存グラフへ流せます。
- `alpha`、`beta`も既存側でscheduleのbound、指数化、temperature、固定noise scaleの優先順位を反映します。学習対象scheduleへの勾配を保持します。
- energyパラメータは`autograd.Function.apply`へ個々のTensor引数として渡します。Python辞書の内部に隠してはいけません。
- noiseは保存済み入力を使い、勾配を返しません。kernel内では新しい乱数を生成しません。
- `full`では過去粒子への勾配を返し、`tbptt_1`のdetachとepisode resetは既存の呼び出し側で維持します。

backwardには、粒子出力と最終score出力の**両方の上流勾配**を渡します。ULAの1段

\[
x'=x+\alpha s_\theta(x,p,o,a)+\beta\epsilon
\]

に対して、少なくとも

\[
\bar{x}=\bar{x}'+\alpha(\partial_xs)^\top\bar{x}',
\qquad
\bar{p}=\alpha(\partial_ps)^\top\bar{x}'
\]

と、energyパラメータ、\(\alpha\)、\(\beta\)への勾配を計算します。フルHessianは作らず、必要な積だけを求めます。

これはenergyの二階微分相当を手書きで処理するものですが、**PPOから見ればchain関数の通常の一回のbackward**です。PPOのためだけに、このcustom Function自身のdouble-backwardまで実装する必要はありません。

保存対象は各段の粒子、context、noise、schedule、パラメータと、必要なclamp情報です。SiLU活性やsoftmax責任度は保存と再計算を比較します。粒子列だけなら、

\[
13\times4\times16\times6\times4
=19,968\text{ bytes}
\]

で、1時刻約19.5 KiB、128時刻約2.44 MiBです。ただし粒子対MLPの中間値まで全保存すると大幅に増えます。時刻ごとの保存値を共有作業領域で上書きしない設計が必要です。

Bは実行時引数、K／D／幅8／L=12は最適化用の特殊化候補とします。prefix用のB=1〜4を扱い、非対応shape・KAN・別samplerは既存PyTorchへ戻します。

ビルドは **`torch.utils.cpp_extension.CUDAExtension`＋C++／CUDA**を第一候補にします。既存Torch Tensor、allocator、現在のCUDA streamを使い、CPUコピーを挟みません。PyTorch 2.3.1と実サーバーのCUDA／コンパイラに合わせて固定します。[PyTorch 2.3.1の拡張ビルド実装](https://raw.githubusercontent.com/pytorch/pytorch/v2.3.1/torch/utils/cpp_extension.py)

Rustを採用するならPyO3／maturinでPython公開できますが、Tensor所有権・stream・CUDA連携・autograd接続は別途必要です。同じCUDA kernelを呼ぶだけなら、Rust層を追加する性能上の利点は小さいと判断します。

**5．検証と採用判断**

1. **解析score単体**：float64で参照autogradとの値・VJPを比較し、float32で現行許容差を確認します。過去粒子と全energyパラメータを対象にします。
2. **12段chain**：initial／recursive／mixed、anchor、warm-start、各schedule、clip、最終score有無を比較します。単純な粒子和だけでなく、任意の上流勾配を使います。
3. **系列**：異なる入力の2時刻を接続して保存領域を検査し、その後128時刻、不均一prefix、reset、`full`／`tbptt_1`へ拡張します。
4. **実際の更新**：同じrollout、noise、初期重み、Adam状態、minibatch順序でPPOとpredictionを比較します。loss、log-probability、勾配、更新後パラメータ、optimizer状態、RNG、resumeを確認します。
5. **速度**：対象GPUを単独使用し、cold startとwarm状態を分けます。main replay、prefix、backward、readout／loss、optimizerを測り、profilerで実kernel数とホスト起動間隔を確認します。採否は初期compile費用を含むrun全体で決めます。

既存報告の **748 passed／1 deselected**は過去の結果であり、今回再実行した結果ではありません。

**最初に投じるべき工数はCの5〜10人日です。** そこで得た残存profileを基に、Bへ進む価値を判断します。例えばC適用後の残存時間の30%しかBで触れないなら、その部分が10倍でも全体への追加効果は約1.37倍です。

今回、全面Rust化する価値が低いのは、**環境、PPOの管理処理、Transformer／diffusion全体、GRU／RNN／DPFRL、TD3の訓練管理、checkpoint／resume／集計**です。TD3のscore経路などは共通のbelief拡張から利益を得られますが、PPOで得た倍率を別方式へ流用せず個別に測定します。高速化の投資先は、繰り返されるbelief演算とそのbackwardに絞るのが妥当です。

---

## 依頼文

# Brief: should the training loop be rewritten in Rust? (2026-11-02)

Read-only; do not modify files. Answer in Japanese, markdown. The user asks: "the code is Python - can it be
rewritten in Rust, and would that make one training run fast?" Give a frank engineering assessment, not a sales
pitch. Read agent/astra-speed-proposal-20261102.md (your own proposal from an hour ago: analytic score +
torch.compile / CUDA Graphs ranked first), agent/speedup-20261102.md, agent/opcount.py and the code
(src/sb_pomdp: belief.py, ppo.py, train.py, policies.py, networks.py, envs.py, td3.py, train_baselines.py).

Facts: one PPO update = 16 belief-sequence replays x 128 time steps x 12 Langevin steps on [4 env, 16
particles, 6 dims] tensors with an energy MLP [8]; ~3250 aten ops per replayed time step (forward + backward);
GPU time is dominated by per-op dispatch + kernel launch (10-30 us each), the GPU's arithmetic units are idle.
Measured today on the Xeon Gold 6326 / RTX 6000 Ada: one PPO update 225-370 s on the GPU (one process alone on
the GPU), 503 s on CPU 1 thread (the transformer readout, diffusion policy and PPO loss are heavier on CPU). The
repo also includes environments (envs.py, numpy), GRU / RNN / DPFRL baselines, TD3, prediction phases,
checkpoint / resume / aggregation tooling, and ~750 tests.

Questions to answer, each with numbers / reasoning:
1. Where would Rust actually remove time? Separate (a) Python interpreter + torch dispatcher overhead per op
   (Rust with tch-rs / libtorch still pays the ATen dispatch and the CUDA launch, only the Python part goes),
   (b) kernel launch latency (only kernel fusion removes it: hand-written CUDA/Triton kernels or a compiler),
   (c) autograd graph construction for create_graph=True (what does a Rust autograd give?). Estimate the per-op
   cost split on this hardware and what fraction Rust alone would save.
2. Options, each with expected speedup of one run, effort (person-days / weeks), exactness vs the current
   runs, and maintenance cost:
   A. Full rewrite in Rust (burn / candle / tch-rs; which has double-backward, CUDA, the ops needed:
      transformer, diffusion policy, logsumexp, autograd.grad with create_graph).
   B. Rust (or C++/CUDA) extension for the hot loop only: the 12-step Langevin chain with the analytic score
      and a hand-written backward, exposed to PyTorch as a custom autograd.Function (so PPO's outer backward
      still works). Compare with writing the same fused kernel in Triton or getting it from torch.compile.
   C. Stay in Python: analytic score + torch.compile + CUDA Graphs (your earlier ranking).
   D. Rust for the environments / rollout only (is that any part of the time? rollout is 4 s of 250 s).
3. The exactness question: a fused kernel sums in a different order; can the current tolerance regime
   (particles 1e-6, gradients 1e-5, Adam params 1e-4) be kept; what about SVGD median bandwidth, SMI mixture
   entropy, the softmax over K particles in the transition energy?
4. A recommendation and, if a Rust/CUDA extension is worth it, the concrete interface (inputs/outputs of the
   step function, what is saved for backward, batch/particle/dim as template parameters or runtime), the build
   (maturin / pyo3 vs torch cpp_extension / CUDA), and how it would be validated against the Python path.
Be concrete about what is NOT worth rewriting.
