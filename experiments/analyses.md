# 補助: 検証解析（学習なし）— 信念は何を持っているか、なぜ勝てないか

学習ジョブではなく、保存 checkpoint・ロールアウト・学習ログを使った解析のまとめ。

| 解析 | 内容 | 解説 |
|---|---|---|
| DeepSets 復号（2026-11-05〜06） | 粒子集合から真の状態を点推定で復号（R²） | ビューアの復号表、[JOINT_REPORT.md](../results/reports/codex_2026-11-06/JOINT_REPORT.md) |
| MDN 復号（2026-11-07） | 8 成分の混合密度で復号。多峰性は本物だが情報源の順位は不変。基準線（GRU / RNN / PF）も含む 118 run | [MDN_REPORT.md](../results/reports/mdn_2026-11-07/MDN_REPORT.md)、スクリプト `probe_mdn.py`（同じ場所）、ビューア https://claude.ai/artifact/9DeCqi4xsok7n1VCPY5Rf3 |
| 記憶切断・摂動（2026-11-05） | 前粒子を N(0, I) に置換 / 開始ノイズの尺度・平行移動を変えて再評価 | [ASTRA2_MEMORY_REPORT.md](../results/reports/codex_2026-11-06/ASTRA2_MEMORY_REPORT.md)、[ASTRA1_VERIFY_REPORT.md](../results/reports/codex_2026-11-06/ASTRA1_VERIFY_REPORT.md) |
| 段数診断（2026-11-04） | 12 段の輸送の性質、延長時の発散、スコアゼロ化 | [STEPS_RESULT.md](../results/reports/steps_2026-11-04/STEPS_RESULT.md)、[IMPROVE_REPORT.final.md](../results/reports/steps_2026-11-04/IMPROVE_REPORT.final.md) |
| 原因調査と改善案（2026-11-07） | FLOPs、リターン、統計、原因候補 8 件、改善案 9 件、第一波計画 | [WHY_REPORT.md](../results/reports/codex_why_2026-11-07/WHY_REPORT.md)、依頼文 `REVIEW_BRIEF.md`（同じ場所） |
| 勾配帰属（2026-11-07） | 「観測勾配が支配して遷移が学べない」は不支持。短い経路（観測 → 終端スコア → condition）への偏りは支持 | [GRAD_REPORT.md](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md) |
| 多峰性の検証（2026-11-09） | 粒子雲の形（k-means silhouette、gap、dip、近傍）を全 88 run・6 万時刻で測定し、N(0,I) 16 点で校正。「多群化の例はない」は誤り（func / prev に明瞭な分離、固定文脈で再現、tps prev に符号候補への部分対応 14 時刻）だが、校正された多峰事後の学習は未確認。MDN の 2 峰率は復号器の出力（無条件でも 100 %） | [MULTI_REPORT.md](../results/reports/codex_multi_2026-11-09/MULTI_REPORT.md) |
| 埋め込みの解析（2026-11-03） | 体座標位置は持つが世界座標・鏡像フレームは復号不可 | [ANALYSIS_REPORT_astra.md](../results/reports/belief_2026-11-03/ANALYSIS_REPORT_astra.md)、[ANALYSIS_REPORT_terra.md](../results/reports/belief_2026-11-03/ANALYSIS_REPORT_terra.md) |
| mountain hike の解析（2026-09） | 旧実装での信念の挙動 | [mountain-hike-analysis-2026-09-08.md](../agent/mountain-hike-analysis-2026-09-08.md) |
| 2026-09 の比較結果 | 旧実験群の結果比較 | [results-comparison-2026-09.md](../docs/results-comparison-2026-09.md) |

## 主な結論（2026-11-07 時点）

- 粒子 + スコアは現在の観測より少ない情報しか持たず（mirror2 の体座標位置、標識、鏡の符号 m、tagr の鬼の距離）、GRU / RNN / PF の再帰状態は観測を時間積分して観測より多くを持つ。
- 学習は「現在観測 → 終端スコア → condition」の短い経路に偏る（tagr energy で顕著）。遷移網にも勾配は来るが記憶を書くための勾配ではない。
- 粒子雲が複数の塊に分かれる例はある（func / prev、素 mirror2 s12）が、真の仮説と質量を正しく表す多峰事後は未確認。MDN の 2 峰率は復号器の出力であって粒子雲の形ではない。
- 最適化の問題か目的の問題かは未決。識別実験は [GRAD_REPORT.md §6.2](../results/reports/codex_grad_2026-11-07/GRAD_REPORT.md)。

- **学習が進まない理由の検証（Codex / Astra、2026-11-10）**: [STALL_REPORT.md](../results/reports/codex_stall_2026-11-10/STALL_REPORT.md)。PPO の更新は動いている（KL・clip 率は伸びた PF と同程度）、価値のノイズでは順位を説明できない（light EV 0.30、停滞 GRU s11 は 0.59）、凍結した light 表現に GRU 教師の行動クローンを載せると 512 話で 3.1 → 4.0 に改善（表現の絶対天井ではない）。残る候補は表現の距離保持不足（最近接距離 R² 0.55 vs GRU 0.88）、予測目的と制御目的の不整合、交互学習のドリフト、信用割当。次は「同じ凍結表現・同じ教師で拡散ヘッドも BC」と「予測頻度をそろえた PPO 勾配遮断 2×2」。私の「light tagr は何をしても 2.2〜2.6」は言い過ぎ（noanchor s13 9.39、func s10 4.39、param s13 4.89）。
- **追試（Codex / Astra 第 2 ラウンド、2026-11-10）**: [STALL_FOLLOWUP.md](../results/reports/codex_stall_2026-11-10/STALL_FOLLOWUP.md)。元と同じ 5 段拡散ヘッドの行動クローンでも凍結 light 表現で 3.10 → 4.16 / 2.90 → 3.98（512 話）→ 方策クラスの表現力不足は不支持。行動 ±0.2 の介入で Q 差が安定する 5 成分のうち局所 PPO 勾配が逆行・反転する例あり（限定支持、GRU にもある）。観測窓 8 / 32 でも最近接距離の復号は light < GRU（0.49〜0.52 vs 0.83）。GPU の次手は「表現凍結 + 拡散 BC actor で PPO 再開」4 本。
