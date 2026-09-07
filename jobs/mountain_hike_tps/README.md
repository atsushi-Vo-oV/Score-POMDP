# mountain_hike_tps — テレポート + 対称観測版 Mountain Hike

`teleport_probability = 0.03`(各 step で確率 p により箱内一様へ再配置)+ `observation_symmetry = abs`(観測は |位置| + ノイズで、符号が分からない = 4 象限の多峰 belief。地形報酬は非対称なので報酬を使える belief だけが峰を絞れる)。belief は跳んだ後に作り直す必要があり、warm start の連鎖(1 step の移動予算が小さい)には不利、粒子フィルタのリサンプリングには有利な性質。v1〜v4 の score 3 エンコーダ(MIG)と、v1 の baseline gru / rnn / 原典準拠 PF(A キュー CPU)を 100 updates で比較する。

| config | methods | updates |
|---|---|---|
| mountain_hike_tps_v1.json | 6 手法(pf は dpfrl) | 100 |
| mountain_hike_tps_v2.json | score × 3 | 100 |
| mountain_hike_tps_v3.json | score × 3 | 100 |
| mountain_hike_tps_v4.json | score × 3 | 100 |

```bash
pjsub jobs/mountain_hike_tp/debug_tps_cpu.sh   # A キューで DEBUG、合格後にゲートが下記を投入
pjsub jobs/mountain_hike_tp/tps_v1_score.sh
pjsub jobs/mountain_hike_tp/tps_v1_alpha.sh
pjsub jobs/mountain_hike_tp/tps_v1_ds.sh
pjsub jobs/mountain_hike_tp/tps_v2_score.sh
pjsub jobs/mountain_hike_tp/tps_v2_alpha.sh
pjsub jobs/mountain_hike_tp/tps_v2_ds.sh
pjsub jobs/mountain_hike_tp/tps_v3_score.sh
pjsub jobs/mountain_hike_tp/tps_v3_alpha.sh
pjsub jobs/mountain_hike_tp/tps_v3_ds.sh
pjsub jobs/mountain_hike_tp/tps_v4_score.sh
pjsub jobs/mountain_hike_tp/tps_v4_alpha.sh
pjsub jobs/mountain_hike_tp/tps_v4_ds.sh
pjsub jobs/mountain_hike_tp/cpu_tps_v1_gru.sh
pjsub jobs/mountain_hike_tp/cpu_tps_v1_rnn.sh
pjsub jobs/mountain_hike_tp/cpu_tps_v1_pf.sh
```
