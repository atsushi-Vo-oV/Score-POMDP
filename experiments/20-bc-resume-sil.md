# 20 BC 初期化 actor から PPO を再開すると保てるか / 稀な成功を再利用する self-imitation

- **問い**: (A) light の信念を凍結し、BC（behaviour cloning）で得た良い actor（凍結表現上で 3.1 → 4.0〜4.2）から PPO を続けると、その水準を保って伸びるか、約 3 へ戻るか。戻れば on-policy 方策勾配が局所解へ引き戻している証拠。(B) 稀な成功が「訪問頻度 × 行動確率 × 時間減衰 × 1 回限り」で薄まるなら、過去の成功遷移を (R − V)₊ で優先再生して模倣する self-imitation learning（SIL、Oh 2018）や指数重み（AWR 型）で改善するか（ユーザー 2026-11-11）。
- **腕（campaign）**: (A) `bcresume-tagr-{orig,bc,bcreset}-v1`（seed 10 / 11。元 actor + 凍結 / BC actor + 凍結 + critic warm-up 10 / BC actor + 凍結 + 価値ヘッド初期化 + warm-up 10、PPO 100 update、評価 10 update ごと）。(B) `trysvgdlight-tagr-tf-svgd-sil-v1`、`-silexp-v1`、`trysvgd-tagr-gru-sil-v1`、`trysvgdlight-carflag-tf-svgd-long-sil-v1`、対照 `trysvgdlight-tagr-tf-svgd-lam99-v1`（GAE λ 0.99）、各 5 seed。
- **設定とジョブ**: 新オプション（既定 off、従来とビット同一）`model.init_from_checkpoint`、`model.actor_init_path`、`model.freeze_modules`、`model.reset_value_head`、`ppo.critic_warmup_updates`、`ppo.self_imitation`。`../config/bcresume_tagr_s1{0,1}_*.json`、`../jobs/bcresume/`、`../jobs/sil/`、BC actor は `../assets/bc_actors/`。SIL はヘッド水準（保存した c_t を再利用、信念の再計算なし）で PPO の各 update の後に 4 step。
- **状態**: (A) 6 本を 2026-11-12 に math へ投入（先頭）。(B) 25 本は準備済み・未投入。

## 結果の解説

- 実装報告と「報酬依存係数の方策勾配」の整理（RWR / AWR / MPO / SIL / 優先再生 / hindsight）: [SIL_REPORT.md](../results/reports/sil_2026-11-11/SIL_REPORT.md)
- 背景の検証: [STALL_REPORT.md](../results/reports/codex_stall_2026-11-10/STALL_REPORT.md)、[STALL_FOLLOWUP.md](../results/reports/codex_stall_2026-11-10/STALL_FOLLOWUP.md)

### 2026-11-13 の結果（seed 10〜12、mean_chain 評価 10 話。検証は [LOCAL_REPORT.md](../results/reports/codex_local_2026-11-13/LOCAL_REPORT.md)）

**BC（Behavior Cloning）再開（tagr、信念と encoder を凍結、PPO 100 update、seed 10 / 11）**

| 開始 actor | 10 話評価の最終 | 共通 256 話評価（Codex） |
|---|---|---|
| 元の actor | 2.6 / 2.2 | 開始 3.5 / 3.4 → PPO 後も改善なし |
| BC actor | 7.5 / 6.4 | BC 直後 4.2 / 4.2 → PPO 後 5.2 / 4.7（差 +1.05 / +0.54、信頼区間は正） |
| BC actor + 価値ヘッド初期化 | 6.9 / 5.0 | — |

10 話評価の「5 → 7」は過大で、256 話では 4.2 → 5 前後。BC 系には critic warm-up 10 update が余分に入っている（交絡）。

**SIL（Self-Imitation Learning、既定値 policy_coef 0.1、4 回 / PPO update）**

| 課題 | 素 | SIL（relu 重み） | SIL（指数重み） | GAE λ 0.99 | GRU + SIL |
|---|---|---|---|---|---|
| Car-Flag（333 update） | −0.4 | −0.4 | −0.7 | −0.4 | — |
| mirror2 | −369 | −1278（3 seed とも悪化） | −700（1 seed 崩壊） | −560（1 seed 崩壊） | −1661 |
| tagr（675 update 時点） | 2.4 | 2.3 | 2.3 | 2.5 | — |
| tagr（seed 10、完走、2026-11-13 夜） | 2.4 | 2.3 | 2.1 | 2.9 | 2.7（GRU の素は 5.3、標準偏差 5.9） |

- mirror2 の悪化は手法の評価になっていない。relu 重み (R − V)₊ が無正規化で、係数込みの勾配が PPO の 54 倍になっていた（正規化すると 0.15 倍）。R と V の尺度の不整合（バグ）は無い。修正案は `results/reports/codex_local_2026-11-13/sil_fix.patch`（未適用）。
- tagr で SIL が選ぶのは「長く生き延びた軌跡の前半」で、BC 方向との余弦は 0.003。模倣しても方策は良い方向に動かない。
- Car-Flag は訓練中に緑旗へ 538〜793 回 / run 到達している（片側に賭けた偶然を含む）。「成功を一度も経験していない」は誤り。

## 結論メモ

- 同じ凍結表現・同じ PPO でも、actor の初期値で結果が変わる。BC actor は PPO で保たれ、さらに伸びる。元の actor は伸びない。
- 元 → BC の重みの直線上は谷ではなく台地。BC の行動は 1 step だけ取っても得にならず、5 step で +0.30、20 step で +0.86、最後まで続けて +2.05。単発の探索では拾えず、時間的に一貫した行動の変化が要る → [22-population-exploration.md](22-population-exploration.md)。
- SIL は 3 課題とも改善なし。mirror2 の悪化は重みの無正規化が原因で、正規化版の再実験は未実施。
