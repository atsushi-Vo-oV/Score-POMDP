# belief_aux job set — 予測十分性による belief 直接教師

診断アークの帰結(2026-09-01: 探索を直し、行動をデモで与えても belief は
死んだまま)を受けた、**状態を一切観測しないモデルフリーの直接教師**。

## 機構(`model.observation_prediction_coef`)

補助損失 = −log (1/K) Σ_k N(o_{t+1}; μ(s_k, a_t), σ(s_k, a_t))

- 使う信号はエージェント自身の (観測, 行動) 列のみ。真の状態・oracle
  特徴・環境モデルは不使用(予測ヘッドは補助タスクで、方策・価値は
  その出力を消費しない = model-free control)。
- **予測器の入力は粒子位置のみ**(score ベクトルを渡さない)ので、
  観測ショートカットではこの損失を下げられない。勾配は replay された
  ULA chain を逆伝播して energy 網に直接届く — これまでで最強かつ
  唯一の「belief を情報化せよ」という 1 次シグナル。
- 予測十分性(PSR)の論理: 未来の観測を予測できる表現は状態の十分統計量。
  CartPole では速度の符号化を、LD ではフィルタリングを強制する。
- 分散も予測する(heteroscedastic)ので、LD の暗所の大ノイズは大 σ で
  説明でき、損失が暗所ノイズに支配されない。
- 既定 coef 0.0 = ヘッド不生成・従来と bit 同一(legacy 正規化済み)。

## アーム

| arm | 構成 | 判定 |
|---|---|---|
| score/alpha/ds_aux | デモ種 WML + aux(coef 1.0、config/ld2d_p3o_wml_demo_aux.json) | wmldemo 対照(stochastic ピーク −15.4、belief 死亡)に対し、プローブ R²(condition/粒子→位置)が立つか。alpha はショートカット経路自体がないので最も純粋 |
| cartpole_aux | v2 PPO + aux、100 updates 上限 | 粒子から次の (位置, 角度) を予測するには速度が要る → 粒子雲の収縮と速度 R² の上昇。return は記憶なし天井のままでも表現が立てば成功 |

## 手順(リポジトリ root、全 MIG)

```bash
pjsub jobs/belief_aux/debug_aux.sh
pjsub jobs/belief_aux/score_aux.sh
pjsub jobs/belief_aux/alpha_aux.sh
pjsub jobs/belief_aux/ds_aux.sh
pjsub jobs/belief_aux/cartpole_aux.sh
```

LD 3 本 ≈ 2h、CartPole ≈ 10-17h(100 updates、resume 不可)。完走後は
scratchpad/wmldemo_probe.py(LD)・analyze_v2 系(CartPole)でプローブ。
