# try-svgd-light — スコアを NN で直接出力する transformer(ULA / SVGD / SMI)

try-svgd の transformer 3 腕と同じ設定(K=16、L=12、ネットワーク幅 [8]、warm start なし、α 学習、予測フェーズ、
diffusion 方策の PPO)で、エネルギーを自動微分する代わりにネットワークがスコアを直接出力する
(`model.score_parameterization: "direct"`)。t=0 は g(o, x) − x、t≥1 は f(o, x, a) + Σ_j softmax_j(k_j) v_j
((k_j, v_j) は前の粒子 x_{t-1,j} との組を読む遷移ネットワークの出力)。2 階微分が不要なぶん軽い。
`-joint` の腕(`score_parameterization: "direct_joint"`)は観測の項と遷移の項を足さず、遷移ネットワークの
注意重みと特徴 (k_j, h_j) から前時刻の信念の要約 c(x) = Σ_j softmax_j(k_j) h_j を作り、(o, x, a, c(x)) を
1 つのネットワークに入れてスコアを出す(観測と遷移をまとめて学習する版)。
基準線(GRU / RNN / DPFRL)とエネルギー版の transformer は try-svgd の結果と比べる。

| task | arm | config | campaign | shards | updates |
|---|---|---|---|---|---|
| mirror2 | tf-ula | config/trysvgdlight_mirror2_tf-ula.json | trysvgdlight-mirror2-tf-ula-v1 | 5 | 333 |
| mirror2 | tf-svgd | config/trysvgdlight_mirror2_tf-svgd.json | trysvgdlight-mirror2-tf-svgd-v1 | 5 | 333 |
| mirror2 | tf-smi | config/trysvgdlight_mirror2_tf-smi.json | trysvgdlight-mirror2-tf-smi-v1 | 5 | 333 |
| mirror2 | tf-ula-joint | config/trysvgdlight_mirror2_tf-ula-joint.json | trysvgdlight-mirror2-tf-ula-joint-v1 | 5 | 333 |
| mirror2 | tf-svgd-joint | config/trysvgdlight_mirror2_tf-svgd-joint.json | trysvgdlight-mirror2-tf-svgd-joint-v1 | 5 | 333 |
| mirror2 | tf-smi-joint | config/trysvgdlight_mirror2_tf-smi-joint.json | trysvgdlight-mirror2-tf-smi-joint-v1 | 5 | 333 |
| tps | tf-ula | config/trysvgdlight_tps_tf-ula.json | trysvgdlight-tps-tf-ula-v1 | 5 | 213 |
| tps | tf-svgd | config/trysvgdlight_tps_tf-svgd.json | trysvgdlight-tps-tf-svgd-v1 | 5 | 213 |
| tps | tf-smi | config/trysvgdlight_tps_tf-smi.json | trysvgdlight-tps-tf-smi-v1 | 5 | 213 |
| tps | tf-ula-joint | config/trysvgdlight_tps_tf-ula-joint.json | trysvgdlight-tps-tf-ula-joint-v1 | 5 | 213 |
| tps | tf-svgd-joint | config/trysvgdlight_tps_tf-svgd-joint.json | trysvgdlight-tps-tf-svgd-joint-v1 | 5 | 213 |
| tps | tf-smi-joint | config/trysvgdlight_tps_tf-smi-joint.json | trysvgdlight-tps-tf-smi-joint-v1 | 5 | 213 |

## Learned chain start (2026-11-03): param_init / func_init / prev_init

Same light TF-SVGD configs, only the start of the 12-step SVGD chain changes (`model.langevin_proposal`,
`model.langevin_warm_start`).  The plain arm starts every recursive time step at standard-normal noise
(initial steps use the observation anchor in all four arms).

| arm | config change | start of the chain at a recursive step |
|---|---|---|
| tf-svgd (reference) | - | `xi ~ N(0, I)` |
| tf-svgd-param-init | `langevin_proposal = "affine"` | `mu + exp(log sigma) * xi`, one learned `(mu, log sigma)` per latent dimension |
| tf-svgd-func-init | `langevin_proposal = "network"` | per particle `mu_k + exp(log sigma_k) * xi_k` with `(mu_k, log sigma_k) = h(x^k_{t-1}, a_{t-1}, o_t)` (MLP `proposal_hidden`, zero-initialised final layer) |
| tf-svgd-prev-init | `langevin_warm_start = true` | the previous particles `x_{t-1}` themselves (no transition proposal) |
| tf-svgd-uniform-init (2026-11-10) | `langevin_proposal = "uniform"` | `(2 Phi(xi) - 1) * sqrt(3)`, i.e. `U(-sqrt(3), sqrt(3))` per latent coordinate: fixed, no parameters, same variance as `N(0, I)`, bounded support (`model.langevin_uniform_half_width`) |

Campaigns `trysvgdlight-<task>-tf-svgd-{param,func,prev}-init-v1`, configs
`config/trysvgdlight_<task>_tf-svgd-{param,func,prev}-init.json`, jobs
`trysvgdlight_<task>_svgd-{param,func,prev}init_s{10..14}.sh` for mirror2 (333 updates), tps (213) and tagr (1293).
The affine / network proposals are zero-initialised, so a fresh model is bit-identical to the reference arm
(`src/tests/test_proposal_init.py`); they are trained only through the gradients that flow into the chain.
The uniform start (campaigns `trysvgdlight-<task>-tf-svgd-uniform-init-v1`, configs
`config/trysvgdlight_<task>_tf-svgd-uniform-init.json`, jobs `trysvgdlight_<task>_svgd-uniforminit_s{10..14}.sh`)
maps the stored standard-normal noise through the normal CDF, so the PPO replay reproduces the same chain start
from the same stored noise; the initial (anchored) rows are unchanged, as in the other three arms.

## Particle count (2026-11-10): tf-svgd-k64 / tf-svgd-k256

Plain light TF-SVGD with `model.num_particles` 16 -> 64 / 256 and nothing else changed (latent dimension stays the
state dimension; compute is not matched, the transition pairs and the attention grow with K^2).  Configs
`config/trysvgdlight_<task>_tf-svgd-k{64,256}.json`, campaigns `trysvgdlight-<task>-tf-svgd-k{64,256}-v1`, jobs
`trysvgdlight_<task>_svgd-k{64,256}_s{10..14}.sh` for mirror2 (333), tps (213), tagr (1293) and carflag (333).
Measured on math (shared GPU, mirror2): one update 30-37 s at K = 64, 45-48 s at K = 128, 86-103 s at K = 256
(K = 16 under the same load: 35-49 s); peak GPU memory 0.3 / 0.4 / 0.7 GB.  Experiment 15.

### Car-Flag (2026-11-10): a fourth task for the chain-start comparison

`car_flag` (horizon 160, no observation noise): the green flag is at +1 or -1 with equal probability, the hidden
side is observed only within 0.2 of the priest at 0.5, so the belief is a two-point mixture that must be carried to
the flag.  Same light TF-SVGD model and schedule as mirror2 (333 updates = 98 + 200 PPO + 35).  Configs
`config/trysvgdlight_carflag_tf-svgd{,-param-init,-func-init,-prev-init,-uniform-init}.json`, campaigns
`trysvgdlight-carflag-tf-svgd{,-param-init,-func-init,-prev-init,-uniform-init}-v1`, jobs
`trysvgdlight_carflag_svgd{,-paraminit,-funcinit,-previnit,-uniforminit}_s{10..14}.sh`.  Baselines matched to the
light TF-SVGD acting cost (1.072 MFLOP, scratchpad flops_carflag.py): GRU [800, 308] / hidden 147
(`config/trysvgd_carflag_gru.json`, campaign `trysvgd-carflag-gru-v1`) and PF v3 mlp [181, 322]
(`config/trysvgd_carflag_pf_v3.json`, `trysvgd-carflag-pf-v3-v1`), jobs `jobs/trysvgd/trysvgd_carflag_{gru,pf_v3}_s{10..14}.sh`.
Sampler controls with the plain N(0, I) chain start: energy TF-ULA (`config/trysvgd_carflag_tf-ula.json`, campaign
`trysvgd-carflag-tf-ula-v1`, jobs `jobs/trysvgd/trysvgd_carflag_ula_s{10..14}.sh`, 1.36 MFLOP) and light TF-ULA
(`config/trysvgdlight_carflag_tf-ula.json`, `trysvgdlight-carflag-tf-ula-v1`, `trysvgdlight_carflag_ula_s{10..14}.sh`).

## One input domain for t = 0 and t >= 1 (2026-11-05): tf-svgd-prior

`model.initial_belief = "prior"` (with `langevin_observation_anchor = false`): the initial network `g(o_0, x) - x`
and the observation anchor are removed.  An initial step is scored by the *recursive* networks exactly like every
later step, with `K` learned prior particles (`num_particles x state_dim`, standard-normal initialised, trained
through the chain) in place of the previous particles and a zero previous action, and its chain starts at
`N(0, I)` like every recursive step.  Motivation: in the reference arm the t = 0 cloud is 2-4x wider and its
score 2-3x larger than at t >= 1 (the anchor log-scale saturates at its upper bound), so the readout sees t = 0
in a different input domain; with one network and one chain start for all t the scales agree by construction.
The chain cost is unchanged (one recursive score per step); the initial network's parameters disappear.
Campaigns `trysvgdlight-<task>-tf-svgd-prior-v1`, configs `config/trysvgdlight_<task>_tf-svgd-prior.json`,
jobs `trysvgdlight_<task>_svgd-prior_s{10..14}.sh` (mirror2 333, tps 213, tagr 1293 updates).  The zero-code
control `tf-svgd-noanchor` is the reference config with only `langevin_observation_anchor = false` (initial
network kept, t = 0 chain starts at noise): campaigns `trysvgdlight-<task>-tf-svgd-noanchor-v1`, configs
`config/trysvgdlight_<task>_tf-svgd-noanchor.json`, jobs `trysvgdlight_<task>_svgd-noanchor_s{10..14}.sh`.
Tests: `src/tests/test_prior_initial_belief.py`.

## SVGD-joint without structure (2026-11-03): tf-svgd-flat

`score_parameterization = "direct_flat"`: the recursive score is one MLP over the plain concatenation
`(o_t, x, a_{t-1}, x_{t-1,1}, ..., x_{t-1,K})` of the query with every previous particle in its stored order; no
attention, no pairwise terms, no permutation invariance (the initial-step network is unchanged).  The hidden width
is set so that the per-query multiply-adds equal the joint arm's (`K` pairwise transition evaluations + the context
network): mirror2 49, tps 51, tagr 27 (vs. 8 in the joint arm).  Campaigns `trysvgdlight-<task>-tf-svgd-flat-v1`,
configs `config/trysvgdlight_<task>_tf-svgd-flat.json`, jobs `trysvgdlight_<task>_svgd-flat_s{10..14}.sh`.
