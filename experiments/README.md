# 実験一覧（2026-11-09 時点）

ここでの「実験」は **1 つのまとまった比較**（1 つの問いに答える腕の組）を指す。ジョブをまとめて投入した単位ではない。
同じ run（campaign）が複数の実験に現れることがある（例: leak 比較の対照腕は予測専用比較の対照でもある）。
各ファイルに、問い・比較する腕（campaign）・課題と update 数・状態・**結果の解説へのリンク**・結論メモを置く。
解説の多くは `results/reports/`（git 管理外）にあるので、リンクは手元の作業ツリーで開く。

共通の枠組み: スコア粒子信念（energy / direct = light）× サンプラー（ULA / SVGD / SMI）を、行動 1 step の FLOPs をそろえた基準線
（GRU / RNN / 粒子フィルタ PF）と PPO で比べる。課題は mirror2（鏡像 2 軸の mountain hike、333 update）、tps（teleport 版、213）、
tagr（鬼ごっこ、距離観測、1293）、tagre（楕円距離、環境 64・粒子 64、2420）。学習は「ランダム方策の次観測予測 pretrain → PPO → 予測 → PPO …」の交互。
1 update = 全環境 128 step のロールアウト 1 回 + 16 optimizer step。seed は 10〜14。実行は math サーバー（`run1102/queue6.txt`、同時 13 本）。

| # | 実験（問い） | ファイル | 課題 | 状態 |
|---|---|---|---|---|
| 1 | サンプラーの違いは効くか（energy ULA / SVGD / SMI vs GRU / RNN / PF） | [01-sampler-energy.md](01-sampler-energy.md) | mirror2 / tps / tagr | 主要腕完走、SMI 一部待ち |
| 2 | スコアを直接出す light は energy / PF と比べてどうか | [02-light-vs-energy.md](02-light-vs-energy.md) | mirror2 / tps / tagr | 完走 |
| 3 | joint 変種（予測と PPO の同時学習）は効くか | [03-joint-variants.md](03-joint-variants.md) | mirror2 / tps / tagr | mirror2 / tps 完走、tagr 待ち |
| 4 | 連鎖の開始分布を変えれば良くなるか（param / func / prev / uniform） | [04-chain-start.md](04-chain-start.md) | mirror2 / tps / tagr / Car-Flag | param / func / prev 完走、uniform と Car-Flag は seed 10–12 完走。t = 0 を一様にする 2 腕（prev-init-u0 / uniform-init-u0、40 本）を 2026-11-14 に投入、結果待ち |
| 5 | t = 0 の尺度不一致を直せば良くなるか（prior / noanchor / flat） | [05-t0-scale.md](05-t0-scale.md) | mirror2 / tps / tagr | 完走 |
| 6 | PF 基準線の観測適合度を非線形にすると（v1 / v2 / v3 / v3b） | [06-pf-observation-function.md](06-pf-observation-function.md) | mirror2 / tps / tagr | 完走 |
| 7 | 凸な α-vector 型読み出しは TF 読み出しに勝つか | [07-alpha-readout.md](07-alpha-readout.md) | tagr / mirror2（tps 待ち） | tagr 完走、mirror2 energy-ULA 完走（全 seed 崩壊）、残り走行中 |
| 8 | 最適化器を TD3 にすると（PPO vs TD3） | [08-optimizer-td3.md](08-optimizer-td3.md) | mirror2 / tps | 完走 |
| 9 | 表現の次元を GRU に合わせると（K × D = 隠れ次元） | [09-representation-size.md](09-representation-size.md) | mirror2 / tagr | 完走 |
| 15 | 粒子数を増やせば良くなるか（K = 16 / 64 / 256 / 1024） | [15-particle-count.md](15-particle-count.md) | mirror2 / tps / tagr / Car-Flag | 投入済み（2026-11-10、tagr K1024 は GPU が空いてから） |
| 16 | 粒子の潜在次元をそろえると（PF 2 次元、light 16 次元） | [16-latent-dimension.md](16-latent-dimension.md) | mirror2 / Car-Flag | 投入済み（2026-11-10） |
| 17 | スコアの有界化と t = 0 の −x は要るか（nobound / logbound） | [17-unbounded-score.md](17-unbounded-score.md) | mirror2 / tagr / Car-Flag | nobound 投入（2026-11-10、seed 10–12 完走: 発散せず素と同帯）、logbound は不要 |
| 18 | 学習量を伸ばすと抜けるか（Car-Flag の PPO 2000 update） | [18-training-budget.md](18-training-budget.md) | Car-Flag | 投入済み（2026-11-11） |
| 19 | スコア網・方策 / 価値ヘッドを KAN にすると | [19-kan.md](19-kan.md) | mirror2 / tagr | 投入済み（2026-11-12） |
| 20 | BC 初期化 actor から PPO を再開すると保てるか / self-imitation（SIL） | [20-bc-resume-sil.md](20-bc-resume-sil.md) | tagr / Car-Flag | BC 再開 6 本投入（2026-11-12）、SIL は準備済み |
| 21 | light の curl を一定 / 増加罰則で抑えると（素の light・GRU と比較） | [21-curl-penalty.md](21-curl-penalty.md) | mirror2 / tagr | 実装・CPU 検証・設定 10 個・ジョブ 50 本を math へ投入済み、seed 10–12 の途中結果あり（2026-11-13） |
| 22 | actor の集団で探索し、最適化専用エージェントに取り込むと（部品の ON / OFF 比較） | [22-population-exploration.md](22-population-exploration.md) | Car-Flag / mirror2 / tagr | 実装・設定 42 個・ジョブ 210 本（light 12 腕 + GRU 2 腕）を 2026-11-13 に math へ投入（seed 10–12 は優先、13–14 は最後尾、mirror2 → Car-Flag → tagr の順）、結果待ち |
| 10 | 楕円距離観測（能動的センシング）で信念は効くか | [10-tagre-elliptic.md](10-tagre-elliptic.md) | tagre | 2 seed 完走（FLOPs 未一致に注意） |
| 11 | 画像観測（Minecraft）で動くか | [11-minecraft.md](11-minecraft.md) | MineRL | ほぼ完走 |
| 12 | 現在観測を方策に直結すると差は埋まるか（leak 2 × 3） | [12-obs-leak.md](12-obs-leak.md) | tagr / mirror2 | seed 10〜11 完走、12〜14 待ち |
| 13 | 表現を予測だけで学び PPO から切ると（prediction_only） | [13-prediction-only.md](13-prediction-only.md) | tagr / mirror2 | seed 10〜11 完走、12〜14 待ち |
| 14 | 世界モデル（Dreamer）は同じ課題でどうか | [14-dreamer.md](14-dreamer.md) | tagr / mirror2 | 実装予定 |

補助: [analyses.md](analyses.md) — 学習を伴わない検証解析（復号、記憶切断、段数診断、Codex 監査）の一覧。

## 共通の結果ファイル

- 全 run の最新評価: [results_summary_1109.md](../results/reports/summary_2026-11-09/results_summary_1109.md)（旧版 [1107](../results/reports/codex_why_2026-11-07/results_summary_1107.md)）
- 信念ビューア（粒子・復号・run 横断表）: https://claude.ai/artifact/9DeCqi4xsok7n1VCPY5Rf3
- mirror2 の腕ごとの動きの比較（日英）: https://claude.ai/artifact/12CJkE2XUB5wLjhk8a7Zh9 （鬼ごっこの比較は信念ビューア内）
- 中間報告 PDF: [2026-11-02](../results/reports/trysvgd_interim_20261102.pdf)、[2026-11-05](../results/reports/trysvgd_interim_20261105.pdf)

## 進め方の約束

- 投入は比較表と費用（slot 時間）を出して「投入 OK」をもらってから。各実験は **seed 2 本を先に**、残りは後ろに並べる。
- 木は campaign ごとに凍結（`Belief-Score-Based-Model-vNN-日付`、sha256 一致、`.sb_overrides`）。対照腕は同じ木・同じ seed で走らせ直す。
- 実験が完走したら、そのファイルの「結果の解説」に集計と解説のリンクを追記する。
