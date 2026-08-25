# ld_tau_tuning job set

score の belief 再帰を殺している「ノイズ支配の ULA」への介入実験。
full/score_transformer/light_dark seed 10・100 updates(v2 プロトコル準拠、
τ=1.0 の対照は完走済み v2 score/light_dark の update 100 時点)。

**全アームでドリフトスケジュール α_l(step 別 step size、init 0.02、
[1e-4,0.5] clamp)を学習する**(config の `langevin_step_size_learnable: true`
がデフォルト)。固定に戻すには `--override model.langevin_step_size_learnable=false`。

| arm | 内容 |
|---|---|
| alpha_learn | **α 学習のみ**(τ=1 固定)— α 学習という新デフォルト単独の効果を v2 対照(α 固定)から分離 |
| tau030 / tau010 / tau003 | 固定温度 τ ∈ {0.3, 0.1, 0.03}(ノイズ √τ 倍 = 実効 energy 1/τ 倍シャープ化)+ α 学習 |
| tau_learn | **完全学習スケジュール**: step 別 log τ(init τ=1、[1e-3,4] clamp)+ α_l を両方学習 |
| warm_tau010 | **warm-start**: 再帰 step の連鎖初期値を fresh N(0,1) でなく前粒子にする + τ=0.1 + α 学習(PF 型の記憶キャリア) |

手順(リポジトリ root から):

```bash
# 1. DEBUG gate(30分枠、3機構 × 1 update)
pjsub jobs/ld_tau_tuning/debug_tau.sh
# 2. 合格後に 6 arms(各 24h 枠、実測見込み ~17h)
pjsub jobs/ld_tau_tuning/alpha_learn.sh
pjsub jobs/ld_tau_tuning/tau030.sh
pjsub jobs/ld_tau_tuning/tau010.sh
pjsub jobs/ld_tau_tuning/tau003.sh
pjsub jobs/ld_tau_tuning/tau_learn.sh
pjsub jobs/ld_tau_tuning/warm_tau010.sh
```

判定は return(update 100 評価)だけでなく **表現プローブ**で行う:
checkpoint を fetch して scratchpad の analyze_v2.py 系スクリプトにかけ、
(a) condition からの位置復元 R²/RMSE が obs-only を超えるか、
(b) 粒子(平均±分散)自体が状態情報を持ち始めるか、
(c) tau_learn は学習された τ スケジュールの軌跡(checkpoint の
`belief.langevin_log_temperature`)がどこへ動くか、を見る。

注意: 各 arm は直接 compare 起動(campaign 機構なし)なので途中中断時は
最初からやり直し(24h 枠に対し実測 ~17h 見込みで完走想定)。
