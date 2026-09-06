# mountain_hike_tp — テレポート版 Mountain Hike(kidnapped robot)

`environment.tasks.mountain_hike.teleport_probability = 0.03`: 各 step で確率 p により遷移が箱内一様な再配置に置き換わる(info["teleported"])。belief は跳んだ後に作り直す必要があり、warm start の連鎖(1 step の移動予算が小さい)には不利、粒子フィルタのリサンプリングには有利な性質。v1〜v4 の score 3 エンコーダ(MIG)と、v1 の baseline gru / rnn / 原典準拠 PF(A キュー CPU)を 100 updates で比較する。

| config | methods | updates |
|---|---|---|
| mountain_hike_tp_v1.json | 6 手法(pf は dpfrl) | 100 |
| mountain_hike_tp_v2.json | score × 3 | 100 |
| mountain_hike_tp_v3.json | score × 3 | 100 |
| mountain_hike_tp_v4.json | score × 3 | 100 |

```bash
pjsub jobs/mountain_hike_tp/debug_tp_cpu.sh   # A キューで DEBUG、合格後にゲートが下記を投入
pjsub jobs/mountain_hike_tp/tp_v1_score.sh
pjsub jobs/mountain_hike_tp/tp_v1_alpha.sh
pjsub jobs/mountain_hike_tp/tp_v1_ds.sh
pjsub jobs/mountain_hike_tp/tp_v2_score.sh
pjsub jobs/mountain_hike_tp/tp_v2_alpha.sh
pjsub jobs/mountain_hike_tp/tp_v2_ds.sh
pjsub jobs/mountain_hike_tp/tp_v3_score.sh
pjsub jobs/mountain_hike_tp/tp_v3_alpha.sh
pjsub jobs/mountain_hike_tp/tp_v3_ds.sh
pjsub jobs/mountain_hike_tp/tp_v4_score.sh
pjsub jobs/mountain_hike_tp/tp_v4_alpha.sh
pjsub jobs/mountain_hike_tp/tp_v4_ds.sh
pjsub jobs/mountain_hike_tp/cpu_tp_v1_gru.sh
pjsub jobs/mountain_hike_tp/cpu_tp_v1_rnn.sh
pjsub jobs/mountain_hike_tp/cpu_tp_v1_pf.sh
```
