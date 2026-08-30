# ld2d_p3o_wml job set

P3O(arXiv 2505.16732)の**モデルフリー移植**。P3O のモデルベース部分は既知
観測モデルによる内側 belief のみで、外側機構は (a) 報酬傾斜リサンプリング
(シミュレータ状態の複製のみ必要、学習モデル不要)、(b) critic なし・clip
なしの重み付き最尤(Fisher 恒等式: ∇log E[exp(ηR)] = E_tilted[Σ∇log π])。
本実装 `ppo.algorithm=p3o_wml` は belief を各手法の学習器に差し替えて
(a)(b) を移植する。方策勾配(score-function 推定量)・モデルフリー。

## 機構(commit 参照: buffer.resample / _wml_policy_objective / train.py hook)

- 64 スロットの同期エピソード集団(rollout_steps=30=horizon、時間切れ以外の
  終了は未対応 → goal_radius 1e-6 の P3O 定数タスク限定。早期終了を検知
  したら即 RuntimeError)。
- `p3o_resample_interval` ごとに区間報酬の exp(η·Δr) で systematic
  resampling: env deepcopy + BeliefContext / RolloutBuilder 系譜 /
  belief 履歴の置換。本番は interval=30(エピソード単位の傾斜のみ —
  このタスクは中間報酬がほぼゼロで区間重みが平坦なため)。DEBUG は
  interval=5 でリサンプリング経路も実走させる。
- 最終区間の重み softmax(η·R) を `wml_weights` として update に渡し、
  方策損失 = −Σ_i w_i · mean_t log π(a_t|cond_t)。ratio/clip/advantage
  なし。critic は symlog 目標で従来どおり学習(診断用、方策に不関与)。
- η=0.05(return 差 20 で重み比 e¹)。稀な良エピソードを PPO のように
  希釈・clip せず、指数選抜で増幅するのが狙い(サボり局所解対策)。

## 制限(v0)

- baseline(gru/rnn/pf)は未対応(train_baselines は明示的に
  NotImplementedError)。score 系で機構の生死を確認してから移植判断。
- campaign 機構なし(direct 起動、resume 不可)。200 updates ≈ 19h 見込み。

## 手順(リポジトリ root、全 MIG)

```bash
pjsub jobs/ld2d_p3o_wml/debug_wml.sh
pjsub jobs/ld2d_p3o_wml/score_wml.sh
pjsub jobs/ld2d_p3o_wml/alpha_wml.sh
pjsub jobs/ld2d_p3o_wml/ds_wml.sh
```

## 判定

- PPO 対照(symlog): gru −28.5 / score_transformer −29.0 / alpha −84.6 /
  deepsets −28.7(最終 checkpoint、±300 振動)。スクリプト参照:
  dead-reckon −20.1、light-visit(素朴)−455、盲目定数最適 ≈ −20。
- 見る点: (1) mean_chain / stochastic の水準と安定性(WML は方策を
  最良軌道へ収縮させるので stochastic の分散が縮むはず)、(2) 評価軌跡の
  max x1(光平面 5.0 への接近 = サボり盆地からの脱出)、(3) log の
  p3o resample 行(mean_unique_ancestors)。
- 脱出が観測されなければ、次の一手はデモ種(スクリプト light-visit 軌道の
  buffer 注入)または情報利得シェーピング。
