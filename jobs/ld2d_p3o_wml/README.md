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

## デモ種入り WML(第 2 弾、config/ld2d_p3o_wml_demo.json)

素の WML の結果: 200 updates 完走も return は PPO と同帯で振動(transformer
−220 / alpha −86 / deepsets −485、いずれも最終単点)。stochastic 評価軌跡は
max x1 7〜10 と**光平面を越えていた**のに選抜が光訪問を増幅しなかった =
「光を見ても belief が観測を使えないので return が良くならない」という
**探索と belief の鶏と卵**を確認。

対策として 64 スロット中 8 を **スクリプト passive-Kalman 制御**(参照
return ≈ −15.5、全学習方策より上)に置き換える。デモ行動は forward-noised
拡散チェーン(DDPM の変分下界)を通じて重み付き最尤に入り、初期は実質 BC、
方策が追い付くと通常の選抜に戻る。デモ制御器は真のタスク定数で Kalman を
回す(デモ生成のみのオラクル、学習器には行動列しか渡らない)。

```bash
pjsub jobs/ld2d_p3o_wml/debug_demo.sh
pjsub jobs/ld2d_p3o_wml/score_demo.sh
pjsub jobs/ld2d_p3o_wml/alpha_demo.sh
pjsub jobs/ld2d_p3o_wml/ds_demo.sh
```

判定: (1) stochastic 評価が passive-KF 帯(−15〜−30)に近づくか =
「行動を与えれば belief は観測を活用できるか」の直接テスト、(2) 3 encoder
間の差(ここで初めて belief 表現の質が露出するはず)、(3) 収束後に demo
重みが下がるか(episodes.csv の slot 別 return)。
