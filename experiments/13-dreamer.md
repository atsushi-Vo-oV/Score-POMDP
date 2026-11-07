# 13 Dreamer 基準線（実装予定）

ユーザー要望（2026-11-07）「基準線に Dreamer も入れて」。実装する版の設計メモ。

## Dreamer とは

Hafner らの世界モデル系列（PlaNet → Dreamer → V2 → V3）。環境との相互作用をリプレイバッファに貯め、(1) 潜在状態の世界モデル、(2) その中で想像したロールアウトによる actor–critic、を学ぶ。方策勾配を実環境ではなく学習済み潜在ダイナミクスの中で計算する点が PPO 系（本研究の全腕）との最大の違い。基準線には DreamerV3 の処方を低次元観測向けに縮めた版を用いる（固定ハイパーパラメータで多領域に効くため）。

## 構成

1. **世界モデル（RSSM）**: h_t = GRU(h_{t−1}, z_{t−1}, a_{t−1})、事前 p(z_t | h_t)、事後 q(z_t | h_t, o_t)（V3 はカテゴリカル潜在 + 1 % unimix。本課題は 8 × 8 程度）。復号ヘッド: 観測（symlog MSE）、報酬（symlog two-hot）、継続（Bernoulli）。損失 = 復号 NLL + 1.0·KL(sg(q) ‖ p) + 0.1·KL(q ‖ sg(p))、free bits 1 nat。
2. **actor–critic（想像）**: リプレイ系列の事後状態から事前で 15 step 想像。critic は two-hot 分布回帰、λ-return（λ = 0.95）、EMA 正則化。actor は 5〜95 パーセンタイル幅で正規化したリターンを最大化、エントロピー係数 3e-4。連続行動は再パラメータ化勾配（REINFORCE も設定で可）。方策は tanh 正規分布。
3. **本比較への組み込み**: 環境遷移数を PPO 腕に合わせる（tagr 2,048,000）、ランダム prefill 200,000 遷移、行動 1 step の FLOPs を energy TF（2.16 MFLOP）に幅合わせ（学習計算は PPO より重いことを表に明記）、予測フェーズなし、評価は他腕と同じ 10 エピソード、段階的 checkpoint 再開に対応。

## TF-SVGD（PPO）との違い（方策・価値の学習）

| 観点 | TF-SVGD（PPO） | Dreamer |
|---|---|---|
| 方策の学習データ | 実環境のオンポリシー・ロールアウト | 学習済み遷移で作る想像ロールアウト |
| 価値の目標 | 実報酬の GAE λ-return | 想像上の報酬ヘッドの λ-return |
| 表現への勾配 | PPO + 価値 + 予測フェーズが信念まで流れる | 流れない。表現は世界モデル損失だけで学ぶ |
| 時間方向の学習圧 | 間接的（PPO の BPTT、1 step 予測） | 直接的（復元と KL が系列全体にかかる） |
| 不確かさの表現 | K 粒子の集合 | z のカテゴリカル分布の 1 サンプル |

## 状態

- 未実装。見積り: 新モジュール（RSSM、ヘッド、想像、λ-return、リプレイ）、`comparison.methods` に `dreamer`、artifacts / 評価 / 再開の契約、FLOPs 幅合わせ、テストで 2〜3 日 + math でのデバッグ 1 日。
- 参考: [DreamerV3 論文](https://arxiv.org/pdf/2301.04104)、[TorchRL の実装](https://docs.pytorch.org/rl/main/reference/dreamer_v3.html)。関連研究の整理は [related-work.pdf](../docs/related-work.pdf)。
