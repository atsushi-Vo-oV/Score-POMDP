# mountain_hike_steps — MCMC(ULA)ステップ予算の実験

粒子が位置を表さない根本原因が「1 step の粒子変位上限 L·α·|∇E| ≈ 0.02-0.04」
という移動予算の不足だったことを受け、予算を直接増やす。

| arm | L | α | 予算 L·α(×|∇E|) | 対照比 | updates | 見込みコスト |
|---|---|---|---|---|---|---|
| L16 | 16 | 0.02 | 0.32 | ×4 | 50 | ~1 h/update(4×) |
| L16a01 | 16 | 0.10 | 1.60 | ×20 | 50 | ~1 h/update |
| L8a005 | 8 | 0.05 | 0.40 | ×5 | 50 | ~30 min/update |

各 arm を score_transformer / score_alpha(パラメータ一致版)で走らせる(6 本)。
評価は mountain-hike-v1 の @25/@50 と比較し、プローブで粒子平均→位置の
R²/RMSE と粒子雲の移動量を見る。GPU メモリは chain のグラフが L に比例
(L=4 で 1.8 GB → L=16 で ~7 GB、MIG 12 GB 内)。

```bash
pjsub jobs/mountain_hike_steps/L16_score.sh
pjsub jobs/mountain_hike_steps/L16_alpha.sh
pjsub jobs/mountain_hike_steps/L16a01_score.sh
pjsub jobs/mountain_hike_steps/L16a01_alpha.sh
pjsub jobs/mountain_hike_steps/L8a005_score.sh
pjsub jobs/mountain_hike_steps/L8a005_alpha.sh
```
コード変更なし(既存キー langevin_steps / langevin_step_size)なので DEBUG は
variants の DEBUG 合格で兼ねる。24h 壁で止まったら同じファイルを再投入。
