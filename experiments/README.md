# 実験セット一覧（2026-11-07 時点）

セットごとに 1 ファイル。各ファイルに目的・設定とジョブ・状態・**結果の解説へのリンク**・結論メモを置く。
解説の多くは `results/reports/`（git 管理外）にあるので、リンクは手元の作業ツリーで開く。

スコア粒子信念（energy / direct = light）× サンプラー（ULA / SVGD / SMI）を、FLOPs をそろえた基準線（GRU / RNN / 粒子フィルタ PF）と
PPO で比べる実験群「try-svgd」とその派生。課題は mirror2（鏡像 2 軸の mountain hike、隠れ変数 m と原点の標識）、tps（teleport 版）、
tagr（鬼ごっこ disk_tag、鬼 3 体への距離観測）、tagre（距離を向き依存の楕円で測る tagr、環境 64・粒子 64・PPO 2 倍）。
学習は「ランダム方策 20 万 step の次観測予測 pretrain → PPO → 現在方策で予測 → PPO …」の交互（25 PPO ごとに 5 予測）。
1 update = 全環境で 128 step のロールアウト 1 回 + 16 optimizer step。seed は 10〜14（shard = seed − 9）。
実行は math サーバー（`run1102/queue6.txt` を scheduler2 が読む、同時 13 本）、一部は Genkai。

| # | セット | ファイル | 課題 | 腕 | 状態 |
|---|---|---|---|---|---|
| 1 | try-svgd（本体） | [01-trysvgd-core.md](01-trysvgd-core.md) | mirror2 / tps / tagr | energy TF-ULA / SVGD / SMI、GRU / RNN / PF | 主要腕は完走、SMI 一部待ち |
| 2 | try-svgd-light | [02-trysvgd-light.md](02-trysvgd-light.md) | mirror2 / tps / tagr | light ULA / SVGD / SMI、joint、prior / noanchor / flat | mirror2 / tps 完走、tagr 変種完走、素 light tagr は #11 の ctl |
| 3 | try-svgd-alpha | [03-trysvgd-alpha.md](03-trysvgd-alpha.md) | tagr（mirror2 / tps は待ち） | score_alpha × ULA / SVGD | tagr v2 完走 |
| 4 | PF v3 / v3b | [04-pf-v3.md](04-pf-v3.md) | mirror2 / tps / tagr | 観測適合度 MLP / 双線形 | 完走 |
| 5 | try-svgd-td3 | [05-trysvgd-td3.md](05-trysvgd-td3.md) | mirror2 / tps | PPO → TD3 | 一部完走、残り待ち |
| 6 | try-svgd-big | [06-trysvgd-big.md](06-trysvgd-big.md) | mirror2 / tagr | K × D = GRU 隠れ次元 | 3 本走行中、17 本待ち |
| 7 | tagre env64 / k64 | [07-tagre-env64-k64.md](07-tagre-env64-k64.md) | tagre | GRU、PF v3、light func / param / 素 | 走行中（FLOPs 未一致に注意） |
| 8 | Minecraft | [08-minecraft.md](08-minecraft.md) | MineRL 歩行 | GRU、TF-SVGD | 走行中 |
| 9 | 初期化変種 | [09-init-variants.md](09-init-variants.md) | mirror2 / tps / tagr | param / func / prev | 完走 |
| 10 | 検証解析（学習なし） | [10-belief-analyses.md](10-belief-analyses.md) | 全課題 | DeepSets / MDN 復号、記憶切断、段数診断、Codex 監査 | 随時 |
| 11 | 観測直結（leak）比較 | [11-obs-leak.md](11-obs-leak.md) | tagr / mirror2 | light / GRU / PF v3 × {ctl, leak} | 60 本投入済み（キュー先頭） |
| 12 | 予測だけで表現を学ぶ | [12-prediction-only.md](12-prediction-only.md) | tagr / mirror2 | light / GRU / PF v3 の prediction_only | 30 本配置済み、投入待ち |
| 13 | Dreamer 基準線 | [13-dreamer.md](13-dreamer.md) | tagr / mirror2 | DreamerV3 縮小版 | 実装予定 |

## 共通の結果ファイル

- 全 run の最新評価: [results_summary_1107.md](../results/reports/codex_why_2026-11-07/results_summary_1107.md)
- 信念ビューア（粒子・復号・run 横断表、58 run + 基準線）: https://claude.ai/artifact/9DeCqi4xsok7n1VCPY5Rf3
- 中間報告 PDF: [2026-11-02](../results/reports/trysvgd_interim_20261102.pdf)、[2026-11-05](../results/reports/trysvgd_interim_20261105.pdf)

## 進め方の約束

- 新しいセットの投入は、比較表と費用（slot 時間）を出して「投入 OK」をもらってから。math のキューは `run1102/add_*.py` で挿入する。
- 木は campaign ごとに凍結（`Belief-Score-Based-Model-vNN-日付`、sha256 一致を確認、`.sb_overrides` で解析スコア + CUDA graphs + microbatch）。
- 対照腕は同じ木・同じ seed で走らせ直す（別時期の結果は参照に留める）。
- セットが完走したら、そのファイルの「結果の解説」に集計と解説のリンクを追記する。
