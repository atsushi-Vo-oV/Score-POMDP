# trysvgd_obs: 観測直結（leak）の 2 × 3 比較セット — tagr（env16、K16、1293 update）と mirror2（333 update）

ユーザー決定（2026-11-07）「既存系も leak して比較実験 OK（これを 1 つの実験セットとして）」。

`model.policy_observation_input = true` は、actor と critic の入力を「信念の条件ベクトル c_t」から「[c_t, o_t]」（生の現在観測を連結）に変える。
次観測予測ヘッドは c_t だけを読み続けるので、予測フェーズの学習圧は信念を通ったまま。全腕（スコア信念、GRU、PF v3）に同じ変更を入れ、
同じ seed・同じ学習手順・同じ FLOPs 予算（基準線は energy TF、light はその約 78 %、leak の追加は約 0.009 MFLOP = 0.5 %、重み +1,536）で比べる。
対照腕も同じ木で走らせ直す（旧 GRU v2 / PF v3 の結果は別の木・別時期なので参照に留める）。seed 10〜14、shard = seed − 9。

| 課題 | 腕 | config | campaign | ジョブ |
|---|---|---|---|---|
| tagr | light-ctl | `config/trysvgdlight_tagr_tf-svgd.json` | `trysvgdlight-tagr-tf-svgd-ctl-v1` | `jobs/trysvgd_obs/trysvgdlight_tagr_svgd-ctl_s1X.sh` |
| tagr | light-obs | `config/trysvgdlight_tagr_tf-svgd-obs.json` | `trysvgdlight-tagr-tf-svgd-obs-v1` | `jobs/trysvgd_obs/trysvgdlight_tagr_svgd-obs_s1X.sh` |
| tagr | gru-ctl | `config/trysvgd_tagr_gru.json` | `trysvgd-tagr-gru-ctl-v1` | `jobs/trysvgd_obs/trysvgd_tagr_gru-ctl_s1X.sh` |
| tagr | gru-obs | `config/trysvgd_tagr_gru-obs.json` | `trysvgd-tagr-gru-obs-v1` | `jobs/trysvgd_obs/trysvgd_tagr_gru-obs_s1X.sh` |
| tagr | pfv3-ctl | `config/trysvgd_tagr_pf_v3.json` | `trysvgd-tagr-pf-v3-ctl-v1` | `jobs/trysvgd_obs/trysvgd_tagr_pf_v3-ctl_s1X.sh` |
| tagr | pfv3-obs | `config/trysvgd_tagr_pf_v3-obs.json` | `trysvgd-tagr-pf-v3-obs-v1` | `jobs/trysvgd_obs/trysvgd_tagr_pf_v3-obs_s1X.sh` |
| mirror2 | light-ctl | `config/trysvgdlight_mirror2_tf-svgd.json` | `trysvgdlight-mirror2-tf-svgd-ctl-v1` | `jobs/trysvgd_obs/trysvgdlight_mirror2_svgd-ctl_s1X.sh` |
| mirror2 | light-obs | `config/trysvgdlight_mirror2_tf-svgd-obs.json` | `trysvgdlight-mirror2-tf-svgd-obs-v1` | `jobs/trysvgd_obs/trysvgdlight_mirror2_svgd-obs_s1X.sh` |
| mirror2 | gru-ctl | `config/trysvgd_mirror2_gru.json` | `trysvgd-mirror2-gru-ctl-v1` | `jobs/trysvgd_obs/trysvgd_mirror2_gru-ctl_s1X.sh` |
| mirror2 | gru-obs | `config/trysvgd_mirror2_gru-obs.json` | `trysvgd-mirror2-gru-obs-v1` | `jobs/trysvgd_obs/trysvgd_mirror2_gru-obs_s1X.sh` |
| mirror2 | pfv3-ctl | `config/trysvgd_mirror2_pf_v3.json` | `trysvgd-mirror2-pf-v3-ctl-v1` | `jobs/trysvgd_obs/trysvgd_mirror2_pf_v3-ctl_s1X.sh` |
| mirror2 | pfv3-obs | `config/trysvgd_mirror2_pf_v3-obs.json` | `trysvgd-mirror2-pf-v3-obs-v1` | `jobs/trysvgd_obs/trysvgd_mirror2_pf_v3-obs_s1X.sh` |

読み方: light だけが大きく伸びれば「読み出しのボトルネック」、GRU も同じだけ伸びれば課題が直接観測を好むだけで差は記憶側に残る、
両方 leak しても残る差が記憶・表現の差。mirror2 では、現在観測から読める鏡の符号 m を light の粒子が落としている（MDN 2026-11-07）ので、
leak で m が方策に届くかが直接の検証になる。1 update ≈ light 30〜40 秒（tagr）/ 100〜170 秒（mirror2）、GRU 3〜10 秒、PF 13〜15 秒。
