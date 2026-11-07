# 実験セット一覧（2026-11-07 時点）

スコア粒子信念（energy / direct = light）× サンプラー（ULA / SVGD / SMI）を、FLOPs をそろえた基準線（GRU / RNN / 粒子フィルタ PF）と
PPO で比べる実験群「try-svgd」とその派生。課題は mirror2（鏡像 2 軸の mountain hike、隠れ変数 m と原点の標識）、tps（teleport 版）、
tagr（鬼ごっこ disk_tag、鬼 3 体への距離観測）、tagre（距離を向き依存の楕円で測る tagr、環境 64・粒子 64・PPO 2 倍）。
学習は「ランダム方策 20 万 step の次観測予測 pretrain → PPO → 現在方策で予測 → PPO …」の交互（25 PPO ごとに 5 予測）。
1 update = 全環境で 128 step のロールアウト 1 回 + 16 optimizer step。seed は 10〜14（shard = seed − 9）。
実行は math サーバー（`~/workspace/agent/sbpomdp/run1102/queue6.txt` を scheduler2 が読む、同時 13 本）、一部は Genkai。

| # | セット | ジョブ | 課題 | 腕 | update | 状態 | 目的 |
|---|---|---|---|---|---|---|---|
| 1 | try-svgd（本体） | `jobs/trysvgd/` | mirror2 / tps / tagr | energy TF-ULA / TF-SVGD / TF-SMI、基準線 GRU / RNN / PF（v2 = 2026-11-02 の FLOPs 再計数で幅合わせ） | mirror2 333、tps 213、tagr 1293 | mirror2 / tps / tagr の TF-SVGD・ULA と基準線 v2 は完走。SMI の一部は待ち行列 | サンプラーの違いを同 FLOPs で基準線と比べる |
| 2 | try-svgd-light | `jobs/trysvgd_light/` | mirror2 / tps / tagr | direct スコア（light）の ULA / SVGD / SMI、joint 版、初期化変種 param / func / prev、prior / noanchor / flat | 同上 | mirror2 / tps 完走。tagr は変種 6 種が完走、素の light tagr は本体セットの ctl 腕（#11）で走らせる | スコアを直接出す軽い版。t = 0 の尺度不一致（prior / noanchor）、開始分布の学習（param / func / prev）の検証 |
| 3 | try-svgd-alpha | `jobs/trysvgd_alpha/` | tagr（mirror2 / tps は Genkai） | score_alpha（α-vector 型の読み出し）× ULA / SVGD × energy | 1293 | tagr v2 完走（3 seed） | 凸な価値読み出しの検証。連続行動では critic 側のみ |
| 4 | PF v3 / v3b | `jobs/trysvgd/*_pf_v3*`, `jobs/trysvgd_tagr/` | mirror2 / tps / tagr | DPFRL 粒子フィルタの観測適合度を MLP（v3）/ 双線形（v3b）に | 同上 | 完走 | PF v1 の構造的欠陥（観測項が粒子間で相殺）の修正 |
| 5 | try-svgd-td3 | `jobs/trysvgd_td3/` | mirror2 / tps | PPO を TD3 に置換 | 同上 | 一部完走（Genkai / math の batch 木） | 方策最適化器の影響 |
| 6 | try-svgd-big | `jobs/trysvgd_big/` | mirror2 / tagr | 粒子数 × 潜在次元 = GRU の隠れ次元（mirror2 80 × 2、tagr 36 × 6）の light ULA / SVGD | 同上 | mirror2 SVGD 3 本が走行中、残り 17 本は待ち行列 | 「表現の次元が足りない」説の検証 |
| 7 | tagre env64 / k64 | `jobs/trysvgd_tagre/` | tagre | GRU、PF v3、light func-init / param-init / 素、energy TF-SVGD（未投入） | 2420（25 + 2000 + 395） | GRU 2 本完走（7.05 / 4.06）、func s10 完走（1.99）、他は走行中。**FLOPs は未一致**（K = 64 で TF が GRU の 9〜13 倍） | 向き依存観測で能動的センシングが要る課題。比較には K16 対照か幅合わせ GRU が要る |
| 8 | Minecraft | `agent/minecraft/`, `config/minecraft_*.json` | MineRL 1 方向歩行（VAE 潜在） | GRU、TF-SVGD | 405 | 走行中（1 update 数分） | 画像観測での動作確認 |
| 9 | 初期化変種（#2 に含む） | `jobs/trysvgd_light/*-{param,func,prev}init*` | mirror2 / tps / tagr | 開始分布 N(μ, σ) 学習 / 網で出す / warm start | 同上 | 45 本完走 | 連鎖の開始を学べば信念が良くなるか（結論: ならない。func は勾配爆発、prev は座標が流れる） |
| 10 | 検証解析（学習なし） | `results/reports/`、ビューア | 全課題 | DeepSets / MDN 復号、記憶切断、摂動、段数診断、Codex 監査 | — | MDN 2026-11-07 完了（118 run、基準線含む） | 信念が何を持っているか。結論: 粒子は現在観測より少ない情報、GRU は積分する |
| 11 | 観測直結（leak）比較 | `jobs/trysvgd_obs/` | tagr / mirror2 | light / GRU / PF v3 × {対照 ctl, leak obs}（`model.policy_observation_input`） | tagr 1293、mirror2 333 | **60 本を 2026-11-07 にキュー先頭へ投入**（木 v16） | 読み出しのボトルネックか記憶の差かを切り分ける |
| 12 | 予測だけで表現を学ぶ | `jobs/trysvgd_predonly/` | tagr / mirror2 | light / GRU / PF v3 の `model.belief_training = prediction_only`（対照は #11 の ctl） | 同上 | 準備済み（30 本）、投入待ち | PPO の勾配を表現から外す（Dreamer 型の分離）。1 step 予測だけで制御に要る情報が入るか |
| 13 | Dreamer 基準線 | （未実装） | tagr / mirror2 | DreamerV3 を低次元向けに縮めた版（RSSM + 想像での actor–critic） | 環境遷移数を PPO 腕に合わせる | 実装予定（2〜3 日） | 生成的な遷移モデルと想像学習が同じ課題でどう出るか |

## 読むべき結果の場所

- 全 run の最新評価: `results/reports/codex_why_2026-11-07/results_summary_1107.md`（math の evaluation.csv を集計、mean_chain = 決定的評価）。
- 信念の復号（MDN）: `results/reports/mdn_2026-11-07/MDN_REPORT.md`、ビューア https://claude.ai/artifact/9DeCqi4xsok7n1VCPY5Rf3 。
- 原因調査と改善案（Codex / Astra）: `results/reports/codex_why_2026-11-07/WHY_REPORT.md`、以前の監査は `results/reports/codex_2026-11-06/`。
- FLOPs: `results/reports/codex_why_2026-11-07/flops_tagre_1107.json`（tagr は一致、tagre は未一致）、再一致幅 `rematch_math.json`。

## 進め方の約束

- 新しいセットの投入は、比較表と費用（slot 時間）を出して「投入 OK」をもらってから。math のキューは `run1102/add_*.py` で先頭に挿入する。
- 木は campaign ごとに凍結（`Belief-Score-Based-Model-vNN-日付`、sha256 一致を確認、`.sb_overrides` で解析スコア + CUDA graphs + microbatch）。
- 対照腕は同じ木・同じ seed で走らせ直す（別時期の結果は参照に留める）。
