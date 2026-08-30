# ld_tau_lr job set

τ study の tau_learn アーム(step 別 log α_l と log τ_l を両方学習)は 100
updates でスケジュールがほぼ動かなかった(α 0.020→0.0200-0.0204、
τ 1.0→0.968-0.991)。「勾配に一貫した圧力がない」のか「単に lr が低くて
動けない」のかを切り分けるため、**スケジュールパラメータだけ lr を上げる**
(`ppo.langevin_schedule_lr_multiplier`、optimizer を 2 param group 化。
既定 1.0 では従来と同一の単一 group Adam で checkpoint 互換)。

log 空間パラメータの 1 step 移動量は Adam ではほぼ lr なので、×1
(2.5e-4)では 100 updates で最大 0.1-0.4、×100(2.5e-2)なら符号が揃えば
150 step で e^3.7(床 0.2 → 0.03 相当)動ける。

| arm | 内容 | 対照 |
|---|---|---|
| tau_lr10 | LD-5D bonus1、α_l+τ_l 学習、倍率 10 | τ study の tau_learn(×1) |
| tau_lr100 | 同上、倍率 100 | 同上 |
| cartpole_lr100 | masked_cartpole、α_l+τ_l 学習、倍率 100、50 updates 上限 | v2 score/cartpole @50(固定スケジュール、greedy 67.2) |
| tau_lr100_sig | tau_lr100 と同一だが `model.langevin_schedule_bound=sigmoid`(clamp の代わりに sigmoid 再パラメータ化、範囲は同じ) | tau_lr100(clamp 版) |

判定: return ではなく **checkpoint の `belief.langevin_log_step_size` /
`belief.langevin_log_temperature` の軌跡**を第一に見る。
- ×100 でも動かない → 勾配に一貫した圧力がない(lr 説は棄却)。
- 動くが return/プローブが変わらない → 学習は機能するが鋭さが効かない課題。
- α が下がり τ が下がる方向へ動き、CartPole の greedy が v2 @50 を超える
  → 床の低下が効いた(スケジュール固定化の根拠になる)。
- α が上がる/τ が上がる方向 → 「雲を広げる」圧力(ショートカット説と整合)。

## 手順(リポジトリ root から、全て MIG queue)

```bash
pjsub jobs/ld_tau_lr/debug_tau_lr.sh
pjsub jobs/ld_tau_lr/debug_tau_sig.sh
pjsub jobs/ld_tau_lr/tau_lr10.sh
pjsub jobs/ld_tau_lr/tau_lr100.sh
pjsub jobs/ld_tau_lr/cartpole_lr100.sh
pjsub jobs/ld_tau_lr/tau_lr100_sig.sh
```

各 arm は直接 compare 起動(campaign 機構なし)で resume 不可。LD は ~10
分/update × 100 ≈ 17 h、CartPole は 50 updates 上限で ≈ 8-15 h、いずれも
24 h 枠内。解析は scratchpad の tau_probe.py(スケジュール値の読み出しと
表現プローブ)を流用する。

## 結果(2026-08-30、clamp 版)

- ×100 では 25 updates でスケジュールが大きく動いた(lr 説は実証)。LD ×100 は @50 までに α が
  clamp 上限 0.5 を突き抜け(raw log 0.65-0.76)、clamp の勾配ゼロで以後凍結、τ は 0.003-0.026
  まで低下 = 決定論的な大歩幅輸送。return @100 mean_chain −4.38(×10 −5.37、固定対照 −5.05)。
- CartPole ×100(50 upd): α→0.0015-0.0028、τ→0.07-0.13(床 0.014-0.028)だが粒子雲は不変
  (chain 停止)、greedy @50 29.6 vs 固定 67.2 = 悪化。
- 凍結対策として `langevin_schedule_bound=sigmoid` を追加し tau_lr100_sig で再実行。
