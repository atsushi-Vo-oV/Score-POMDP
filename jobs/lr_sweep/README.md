# lr_sweep — 学習が進まない腕の学習率調整

診断(metrics): 罠にかかった Mountain Hike の transformer / deepsets / pf も、学習できた
gru も KL 中央値 0.002-0.003・clip 0.02-0.03 で同じ速度で動いている → 方策が
「動けない」のではなく、箱外の平坦報酬(勾配 0)に早期に落ちて戻れない。
lr は両方向を試す:

| arm | lr | 狙い |
|---|---|---|
| lr1e3_score / lr1e3_pf | 1e-3(×4) | 罠に落ちる前に地形勾配を掴む / 早く抜ける |
| lr6e5_score / lr6e5_pf | 6.25e-5(×1/4) | 初期の方策ドリフトを遅くし、箱外への逸走を防ぐ |
| p3o_lr6e5_score / gru | 6.25e-5 | LD2D P3O の ±300 振動が step 幅由来かを検証(gru は対照) |

全て campaign(segmented)、200 updates。score 系は ~16 min/update なので 24h 壁で再投入。
