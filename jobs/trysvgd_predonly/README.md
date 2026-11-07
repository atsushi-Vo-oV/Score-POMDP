# trysvgd_predonly: 表現を観測予測だけで学ぶ（actor–critic からは止める）— tagr と mirror2

ユーザー提案（2026-11-07）「観測再構成ロスだけで学習して、actor–critic のときは belief を止める」。

`model.belief_training = prediction_only`: 信念（スコア網・提案）と読み出し（transformer）は次観測予測 −log p(o_{t+1} | c_t, a_t) だけで学び、
その損失は予測フェーズに加えて **PPO の全ロールアウトにも**かける（Dreamer のように表現学習を毎 update 行う）。PPO の勾配は actor / critic の
ヘッドだけに届く（条件ベクトルに stop-gradient）。GRU / PF v3 にも同じ扱い（再帰状態・フィルタは予測だけで学ぶ = AIS 型）。
対照は `jobs/trysvgd_obs` の `*-ctl-v1`（joint: PPO + 予測の両方が表現を形づくる、従来どおり）。seed 10〜14、shard = seed − 9。

| 課題 | 腕 | config | campaign | ジョブ |
|---|---|---|---|---|
| tagr | light | `config/trysvgdlight_tagr_tf-svgd-predonly.json` | `trysvgdlight-tagr-tf-svgd-predonly-v1` | `jobs/trysvgd_predonly/trysvgdlight_tagr_svgd-predonly_s1X.sh` |
| tagr | gru | `config/trysvgd_tagr_gru-predonly.json` | `trysvgd-tagr-gru-predonly-v1` | `jobs/trysvgd_predonly/trysvgd_tagr_gru-predonly_s1X.sh` |
| tagr | pfv3 | `config/trysvgd_tagr_pf_v3-predonly.json` | `trysvgd-tagr-pf-v3-predonly-v1` | `jobs/trysvgd_predonly/trysvgd_tagr_pf_v3-predonly_s1X.sh` |
| mirror2 | light | `config/trysvgdlight_mirror2_tf-svgd-predonly.json` | `trysvgdlight-mirror2-tf-svgd-predonly-v1` | `jobs/trysvgd_predonly/trysvgdlight_mirror2_svgd-predonly_s1X.sh` |
| mirror2 | gru | `config/trysvgd_mirror2_gru-predonly.json` | `trysvgd-mirror2-gru-predonly-v1` | `jobs/trysvgd_predonly/trysvgd_mirror2_gru-predonly_s1X.sh` |
| mirror2 | pfv3 | `config/trysvgd_mirror2_pf_v3-predonly.json` | `trysvgd-mirror2-pf-v3-predonly-v1` | `jobs/trysvgd_predonly/trysvgd_mirror2_pf_v3-predonly_s1X.sh` |

読み方: prediction_only で light が改善すれば「PPO の勾配が信念を壊していた / 共適応のドリフト」が原因側、悪化すれば
「1 step 予測だけでは制御に要る情報（特に記憶）が入らない」ことの証拠。GRU との差がどう変わるかで、表現学習の信号を制御から分離する
Dreamer 型の設計の価値も読める。PPO 側は BPTT を使わないが、予測損失のために連鎖の再生は残るので実時間はほぼ同じ。
