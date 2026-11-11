# 20 BC 初期化 actor から PPO を再開すると保てるか / 稀な成功を再利用する self-imitation

- **問い**: (A) light の信念を凍結し、BC（behaviour cloning）で得た良い actor（凍結表現上で 3.1 → 4.0〜4.2）から PPO を続けると、その水準を保って伸びるか、約 3 へ戻るか。戻れば on-policy 方策勾配が局所解へ引き戻している証拠。(B) 稀な成功が「訪問頻度 × 行動確率 × 時間減衰 × 1 回限り」で薄まるなら、過去の成功遷移を (R − V)₊ で優先再生して模倣する self-imitation learning（SIL、Oh 2018）や指数重み（AWR 型）で改善するか（ユーザー 2026-11-11）。
- **腕（campaign）**: (A) `bcresume-tagr-{orig,bc,bcreset}-v1`（seed 10 / 11。元 actor + 凍結 / BC actor + 凍結 + critic warm-up 10 / BC actor + 凍結 + 価値ヘッド初期化 + warm-up 10、PPO 100 update、評価 10 update ごと）。(B) `trysvgdlight-tagr-tf-svgd-sil-v1`、`-silexp-v1`、`trysvgd-tagr-gru-sil-v1`、`trysvgdlight-carflag-tf-svgd-long-sil-v1`、対照 `trysvgdlight-tagr-tf-svgd-lam99-v1`（GAE λ 0.99）、各 5 seed。
- **設定とジョブ**: 新オプション（既定 off、従来とビット同一）`model.init_from_checkpoint`、`model.actor_init_path`、`model.freeze_modules`、`model.reset_value_head`、`ppo.critic_warmup_updates`、`ppo.self_imitation`。`../config/bcresume_tagr_s1{0,1}_*.json`、`../jobs/bcresume/`、`../jobs/sil/`、BC actor は `../assets/bc_actors/`。SIL はヘッド水準（保存した c_t を再利用、信念の再計算なし）で PPO の各 update の後に 4 step。
- **状態**: (A) 6 本を 2026-11-12 に math へ投入（先頭）。(B) 25 本は準備済み・未投入。

## 結果の解説

- 実装報告と「報酬依存係数の方策勾配」の整理（RWR / AWR / MPO / SIL / 優先再生 / hindsight）: [SIL_REPORT.md](../results/reports/sil_2026-11-11/SIL_REPORT.md)
- 背景の検証: [STALL_REPORT.md](../results/reports/codex_stall_2026-11-10/STALL_REPORT.md)、[STALL_FOLLOWUP.md](../results/reports/codex_stall_2026-11-10/STALL_FOLLOWUP.md)

## 結論メモ

- （走行中）
