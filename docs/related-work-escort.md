# ESCORT との比較 / Comparison with ESCORT

ESCORT: Efficient Stein-variational and Sliced Consistency-Optimized Temporal Belief Representation for POMDPs.
Yunuo Zhang, Baiting Luo, Ayan Mukhopadhyay, Gabor Karsai, Abhishek Dubey (Vanderbilt). NeurIPS 2025.
arXiv: https://arxiv.org/abs/2510.21107 / code: https://github.com/scope-lab-vu/ESCORT

メモ作成日 / written: 2026-11-04。本文（arXiv HTML v1）と付録から読み取った内容で、
表の数値は論文の記載をそのまま引いている（一部、本文と付録で粒子数や帯域幅の記載が食い違う）。

---

## 日本語

### ESCORT の要点

- **目的**: POMDP の信念を粒子集合で表し、高次元・多峰な信念を粒子フィルタの縮退（再サンプリング）や
  パラメトリック近似の容量制限なしに表現する。
- **信念更新（1 時刻）**: 既知モデルを使う 3 段 + SVGD。
  1. 遷移 x̃_i = T(x_i, a) + η_i（η はガウス雑音）。
  2. 観測尤度 w_i = O(o | x̃_i, a) を計算し、尤度加重平均 μ_obs へ粒子を引き寄せる補正
     Δx_i = α w_i (μ_obs − x̃_i)（厳密なベイズ更新ではなく発見的な局所補正）。
  3. SVGD 速度場 φ*(x) = (1/n) Σ_j [k(x_j, x) ∇log p(x_j) + ∇_{x_j} k(x_j, x)] − λ ∇R_corr。
     カーネルは RBF、score は既知の観測モデル log O の有限差分、ステップ幅 ε = 0.01（減衰あり）、
     縮退防止に小さなノイズを加える。粒子数は 100〜1000。
- **相関を保つ正則化 R_corr**: generalized sliced Wasserstein に着想を得た射影行列 A_i（目標分布と
  現在分布の相関差の固有ベクトルで初期化し、射影距離を最大化するように最適化）で、
  速度場の射影方向の分散を抑え、状態次元間の相関構造を保つ。
- **時間的一貫性**: 連続する信念 b_t, b_{t+1} の sliced 1-Wasserstein 距離を罰則にし、1 次元射影の
  ソートで得た粒子の対応 σ に沿って「前時刻の対応粒子へ戻す力」をかける（物理的にあり得ない
  信念の跳びを抑える）。
- **学習されるもの**: 方策ネットワークだけ（粒子ごとのエンコーダ → 平均プーリング → MLP、
  REINFORCE 型の方策勾配）。遷移・観測モデルは**既知を前提**（著者も限界として明記）。
  信念更新は方策の損失から勾配を受けない。
- **評価**: Light-Dark 10 次元、Kidnapped Robot 20 次元、多目標追跡 20 次元、合成混合ガウス 1〜200 次元。
  比較は素の SVGD、SIR 粒子フィルタ、DVRL、POMCPOW。真の状態に対する位置誤差で素の SVGD より
  8.5 % / 16.9 % / 24.7 % 改善。付録の Flickering Atari では DVRL と同程度（MsPacman と Asteroids で
  上回り、Pong では僅かに下回る）。計算量は 10 次元で 6.7 GFLOP/update、200 次元で 926 GFLOP/update。

### 対応表

| 観点 | ESCORT | 本研究（score-based particle belief、light TF-SVGD など） |
| --- | --- | --- |
| 遷移・観測モデル | **既知**（必須と明記） | **未知**。score 場（direct / energy）を報酬（PPO）と次観測予測から学習 |
| 粒子の空間 | 物理状態空間（10〜20 次元）、粒子 100〜1000 | 学習した潜在空間（2〜6 次元）、粒子 16 |
| 遷移の入り方 | 既知 T で粒子を 1 回動かす（+ 雑音） | 前粒子の混合を score の遷移項 Σ_j softmax_j · MLP_trans(x, x′_j, a) として持つ |
| 観測の入り方 | 既知 O の尤度で加重平均へ引き寄せ + SVGD の score（log O の有限差分） | 学習した観測 score MLP_obs(o, x, a) が粒子を動かす。初期時刻は観測 anchor |
| SVGD | RBF、ε = 0.01 固定（減衰）、縮退防止ノイズ、相関正則化 | RBF（median heuristic）、12 段、段ごとの学習ステップ幅 α_l（≈ 0.2）、ノイズなし |
| 粒子の出発点 | 前時刻の粒子（時間的一貫性の罰則つき） | 既定は毎時刻 N(0, I)、prev_init 変種は前時刻の粒子（罰則なし） |
| 相関構造 | 学習した射影行列で保つ | なし（潜在が低次元） |
| 読み出し | 粒子ごとエンコーダ → 平均プーリング | transformer（粒子と score をトークン） |
| 方策学習 | REINFORCE 型、信念は固定入力 | PPO、保存ノイズで連鎖を再生して**信念まで逆伝播**、予測フェーズを交互に実施 |
| 信念の評価 | 真の状態に対する位置誤差、MMD、mode coverage | return と、粒子・埋め込みに対するプローブ（線形 / kNN / MLP / DeepSets） |
| 再サンプリング・重み | なし（SVGD の反発で代替） | なし（同じ） |

### 一致点

- 信念を等重みの粒子集合で持ち、再サンプリングを使わず、SVGD の引力・反発で多峰性と広がりを保つ、
  という設計思想は同じ。粒子の輸送でベイズ更新を置き換える「重み付けから輸送へ」の系譜
  （Stein particle filter、Ensemble Score Filter）に両者とも属する。
- 置換不変な読み出しから方策勾配で方策を学ぶ点も同じ。
- ESCORT の時間的一貫性（前時刻の対応粒子へ戻す力）は、本研究の prev_init（前粒子から出発）と
  同じ狙いで、罰則として実装するか初期化として実装するかの違い。

### 相違点

1. **モデルの有無が本質的な差。** ESCORT は既知の T, O から score を有限差分で求めるので、信念は
   「近似ベイズ推定」として真の状態に対する誤差で評価できる。本研究は score 自体を学習対象にし、
   信念の良さは return と下流の予測損失でしか定義されない。これは本研究の弱点（信念が鏡像フレームを
   解決しない、観測の体座標表現に留まる）の根源でもある。
2. **学習信号の向き。** ESCORT は信念を方策の固定入力とし、方策だけを学ぶ。本研究は信念の連鎖を通して
   報酬から逆伝播する（実装上は保存ノイズによる再生）。
3. **規模。** ESCORT は 100〜1000 粒子、1 時刻 1 回の既知遷移 + SVGD。本研究は FLOPs を基準線に
   揃えるため 16 粒子 × 12 段で、潜在次元も小さい。
4. **正則化の有無。** 相関射影と時間的一貫性は本研究にはない。逆に、段ごとの学習ステップ幅と
   観測 anchor は ESCORT にはない。
5. **ベンチマーク。** ESCORT は既知モデルの 10〜20 次元ドメイン（+ Flickering Atari）、本研究は
   mirror2 / tps / 鬼ごっこ / Minecraft（モデルなし、画像入力を含む）。

### 本研究への示唆

- **既知モデルの上限参照として使える。** mirror2 / tps は遷移・観測が解析的に書けるので、ESCORT 型の
  既知モデル SVGD 信念 + 本研究の方策を走らせれば「信念が正しければどこまで行けるか」の参照になる
  （GRU + 理想の信念の参照と同系統）。Weight-Space Programming の検証で得た「標識を使えば −150 前後」
  という参照点とも比較できる。
- **時間的一貫性の罰則**は prev_init の代替・補完として安価に試せる（連続する粒子雲の sliced W1 を
  補助損失にする）。ただし本研究の解析では問題は安定性より学習信号なので、優先度は中。
- **相関射影**は潜在次元が大きい設定（鬼ごっこ 6 次元、Minecraft）でのみ意味がある。
- ESCORT の Atari 結果（DVRL と同程度、反応の速い環境では時間的一貫性が足かせ）は、本研究の
  Minecraft 実験の解釈にも当てはまりうる。

---

## English

### What ESCORT does

- **Goal**: represent POMDP beliefs with a particle set that captures high-dimensional, multi-modal
  structure without the degeneracy of particle-filter resampling or the capacity limits of
  parametric families.
- **Belief update per time step**: a model-based three-step update plus SVGD.
  1. Transition x̃_i = T(x_i, a) + η_i with Gaussian noise η.
  2. Observation likelihood w_i = O(o | x̃_i, a), then a heuristic correction that pulls particles toward
     the likelihood-weighted mean μ_obs: Δx_i = α w_i (μ_obs − x̃_i) (a local adjustment, not an exact
     Bayes update).
  3. SVGD velocity φ*(x) = (1/n) Σ_j [k(x_j, x) ∇log p(x_j) + ∇_{x_j} k(x_j, x)] − λ ∇R_corr with an RBF
     kernel; the score comes from finite differences of the known observation model log O; step size
     ε = 0.01 with decay; a small noise term prevents collapse; 100 to 1000 particles.
- **Correlation-aware regularizer R_corr**: projection matrices A_i inspired by generalized sliced
  Wasserstein distances (initialised from eigenvectors of the correlation difference between target
  and current particles, then optimised to maximise projected distances) penalise the variance of the
  velocity field along those projections, so correlations between state dimensions survive the update.
- **Temporal consistency**: a sliced 1-Wasserstein penalty between consecutive beliefs b_t and b_{t+1};
  sorting in each 1-D projection gives a particle matching σ, and each particle is pushed toward its
  matched particle from the previous step (to suppress physically implausible jumps).
- **What is learned**: only the policy network (per-particle encoder, mean pooling, MLP; REINFORCE-style
  policy gradient). Transition and observation models are **assumed known** (listed as a limitation by the
  authors). The belief update receives no gradient from the policy loss.
- **Evaluation**: Light-Dark navigation (10-D), Kidnapped Robot (20-D), multiple-target tracking (20-D),
  synthetic Gaussian mixtures from 1-D to 200-D. Baselines: plain SVGD, SIR particle filter, DVRL,
  POMCPOW. Position error against the true state improves over plain SVGD by 8.5 % / 16.9 % / 24.7 %.
  On Flickering Atari (appendix) ESCORT is on par with DVRL (better on MsPacman and Asteroids, slightly
  worse on Pong). Cost: 6.7 GFLOP per update at 10-D, 926 GFLOP at 200-D.

### Side by side

| Aspect | ESCORT | This project (score-based particle belief, e.g. light TF-SVGD) |
| --- | --- | --- |
| Transition / observation model | **known** (stated as essential) | **unknown**; the score field (direct / energy) is learned from reward (PPO) and next-observation prediction |
| Particle space | physical state space (10 to 20 dims), 100 to 1000 particles | learned latent space (2 to 6 dims), 16 particles |
| How the transition enters | one step of the known T plus noise | mixture over previous particles inside the score: Σ_j softmax_j · MLP_trans(x, x′_j, a) |
| How the observation enters | known likelihood pulls particles to the weighted mean, plus the SVGD score (finite differences of log O) | learned observation score MLP_obs(o, x, a) moves the particles; an observation anchor at the first step |
| SVGD | RBF, fixed ε = 0.01 with decay, anti-collapse noise, correlation regulariser | RBF (median heuristic), 12 steps, learned per-step step size α_l (≈ 0.2), no noise |
| Starting particles | previous particles, with the temporal-consistency penalty | default: fresh N(0, I) every step; the prev_init variant starts from the previous particles without a penalty |
| Correlation structure | preserved by learned projections | none (latent is low-dimensional) |
| Readout | per-particle encoder, mean pooling | transformer over particles and scores |
| Policy learning | REINFORCE-style; the belief is a fixed input | PPO; the chain is replayed from stored noise so gradients **reach the belief**; alternating prediction phases |
| Belief evaluation | position error vs. true state, MMD, mode coverage | return, plus probes on particles and embeddings (linear / kNN / MLP / DeepSets) |
| Resampling / weights | none (SVGD repulsion instead) | none (same) |

### Shared design

- Both keep the belief as an equally weighted particle set, avoid resampling, and rely on SVGD's
  attractive-repulsive dynamics for multi-modality and spread. Both belong to the "from reweighting to
  transport" lineage (Stein particle filter, Ensemble Score Filter).
- Both read the particle set with a permutation-invariant module and train the policy by policy gradient.
- ESCORT's temporal consistency (a force toward the matched particle of the previous step) has the same
  intent as our prev_init variant (start from the previous particles); one is a penalty, the other an
  initialisation.

### Differences

1. **Model knowledge is the essential difference.** ESCORT obtains the score from known T and O, so its
   belief is an approximate Bayes filter whose error against the true state can be measured. We learn the
   score itself, so the belief is only defined through return and the downstream prediction loss. This is
   also the root of our weakness (the belief does not resolve the mirror frame and stays close to a
   body-frame encoding of the observation).
2. **Direction of the learning signal.** ESCORT treats the belief as a fixed input and learns only the
   policy. We backpropagate reward through the belief chain (replayed from stored noise).
3. **Scale.** ESCORT uses 100 to 1000 particles and one known transition step plus SVGD per time step. We
   match FLOPs to the baselines with 16 particles times 12 steps in a small latent space.
4. **Regularisers.** Correlation projections and temporal consistency have no counterpart here; learned
   per-step step sizes and the observation anchor have no counterpart in ESCORT.
5. **Benchmarks.** ESCORT: known-model 10 to 20-D domains plus Flickering Atari. Here: mirror2 / tps /
   disk tag / Minecraft, all model-free, one with image input.

### What this suggests for the project

- **A known-model upper reference.** mirror2 and tps have analytic transition and observation models, so
  an ESCORT-style known-model SVGD belief combined with our policy would show how far a correct belief
  gets (in the same spirit as the "GRU + ideal belief" reference). It can also be compared with the
  reference point from the weight-space-programming test (about −150 when the signpost is used).
- **Temporal-consistency penalty** is a cheap complement to prev_init (sliced W1 between consecutive
  particle clouds as an auxiliary loss). Our analysis points at the learning signal rather than stability,
  so the priority is moderate.
- **Correlation projections** only matter where the latent is larger (disk tag with 6 dims, Minecraft).
- ESCORT's Atari result (on par with DVRL, temporal consistency hurting fast reactive dynamics) may apply
  to the interpretation of our Minecraft runs as well.
