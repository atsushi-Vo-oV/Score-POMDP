# pf_dpfrl — PF-RNN / DPFRL 原典準拠の particle-filter baseline

旧 pf baseline(決定論的 ensemble)は粒子が 1 step で 1 点に崩壊し重みが常に一様
(ESS = K)で、PF として機能していなかった。`comparison.pf_variant=dpfrl` は
Ma, Karkus, Hsu, Lee(AAAI 2020 / ICLR 2020)の機構を忠実に実装する:

| 要素 | 実装 |
|---|---|
| 遷移 | PF-GRU セル: n = tanh(W_n[r∘h, e] + σ(h,e)·ξ)、ξ〜N(0,I) 再パラメータ化、σ = softplus(W_Σ[h,e]) |
| 初期粒子 | h_0 = μ(e_0) + σ_0(e_0)·ξ |
| 観測関数 | f_obs = 線形 1 層(活性化なし)、log w ← log w + f_obs、log_softmax で正規化(η) |
| soft resampling | 毎 step: q = αw + (1−α)/K からサンプル(保存した一様乱数で逆 CDF)、w' = w_a/q_a → 正規化。勾配は w/q 経由 |
| belief 要約 | 重み付き平均粒子 h̄ + MGF 特徴 Σ w' exp(v_jᵀh)(v 学習、`pf_mgf_features`) |
| 乱数の再生 | ξ と一様乱数を入力列に連結して保存(`noise_dim = K(D+1)`)→ prefix 再生・recondition・resume が厳密一致(テスト済) |

α は既存 `pf_soft_alpha`(0.9 = DPFRL の Mountain Hike 値)。既定 `pf_variant=deterministic`
で旧挙動・旧 checkpoint 互換。

| arm | 設定 | params |
|---|---|---|
| pfdpfrl_matched | K16, D16, pf_hidden [108,192], mgf 8 | 76,080(旧 pf 75,952 に一致) |
| pfdpfrl_paper | K30, D128, pf_hidden [52,124], mgf 8(DPFRL の Mountain Hike サイズ) | 211,536 |

```bash
pjsub jobs/pf_dpfrl/debug_pfdpfrl.sh
pjsub jobs/pf_dpfrl/pfdpfrl_matched.sh
pjsub jobs/pf_dpfrl/pfdpfrl_paper.sh
```
