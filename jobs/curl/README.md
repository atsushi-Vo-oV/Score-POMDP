# SVGD-light の curl penalty 比較

mirror2 / tagr × 5 腕 × seed 10–14、設定 10 個・ジョブ 50 本。**未投入**。GRU の新規ジョブは作らず、既存 campaign を比較に使う。

| 腕 | `score_curl_penalty` | schedule | monitor | 意味 |
|---|---:|---|---|---|
| mon | 0 | constant | true | 素の light と同じ学習、curl ratio のみ追加記録 |
| c001 | 0.01 | constant | false | 小さい側の一定係数 |
| c01 | 0.1 | constant | false | 大きい側の一定係数 |
| r01 | 0.1 | ramp | false | PPO の進行で 0 → 0.1 |
| r1 | 1.0 | ramp | false | PPO の進行で 0 → 1.0 |

共通: `score_curl_samples=8`、`score_curl_estimator=auto`、`score_curl_probes=4`。mirror2 は潜在次元 2 で exact、tagr は潜在次元 6 で Hutchinson。既存 light の設定を継承し、label と curl 設定だけ変更した。

| task | config（`config/` 以下） | campaign | 総 update |
|---|---|---|---:|
| mirror2 | trysvgdlight_mirror2_tf-svgd-curl-mon.json | trysvgdlight-mirror2-tf-svgd-curl-mon-v1 | 333 |
| mirror2 | trysvgdlight_mirror2_tf-svgd-curl-c001.json | trysvgdlight-mirror2-tf-svgd-curl-c001-v1 | 333 |
| mirror2 | trysvgdlight_mirror2_tf-svgd-curl-c01.json | trysvgdlight-mirror2-tf-svgd-curl-c01-v1 | 333 |
| mirror2 | trysvgdlight_mirror2_tf-svgd-curl-r01.json | trysvgdlight-mirror2-tf-svgd-curl-r01-v1 | 333 |
| mirror2 | trysvgdlight_mirror2_tf-svgd-curl-r1.json | trysvgdlight-mirror2-tf-svgd-curl-r1-v1 | 333 |
| tagr | trysvgdlight_tagr_tf-svgd-curl-mon.json | trysvgdlight-tagr-tf-svgd-curl-mon-v1 | 1293 |
| tagr | trysvgdlight_tagr_tf-svgd-curl-c001.json | trysvgdlight-tagr-tf-svgd-curl-c001-v1 | 1293 |
| tagr | trysvgdlight_tagr_tf-svgd-curl-c01.json | trysvgdlight-tagr-tf-svgd-curl-c01-v1 | 1293 |
| tagr | trysvgdlight_tagr_tf-svgd-curl-r01.json | trysvgdlight-tagr-tf-svgd-curl-r01-v1 | 1293 |
| tagr | trysvgdlight_tagr_tf-svgd-curl-r1.json | trysvgdlight-tagr-tf-svgd-curl-r1-v1 | 1293 |

各行に `jobs/curl/<config の拡張子なし>_s{10,11,12,13,14}.sh` が対応する。各ジョブは `SB_POMDP_CONFIG`、`SB_POMDP_CAMPAIGN_ID`、`SB_POMDP_SHARD_INDEX=SB_POMDP_BULK_INDEX=seed-9`、`SB_POMDP_SEGMENT_UPDATES` を設定し、`exec bash jobs/genkai_production.sh` を呼ぶ。再開時は同じジョブを使う。

## 対照と評価

- 素の SVGD-light: `trysvgdlight-mirror2-tf-svgd-ctl-v1`、`trysvgdlight-tagr-tf-svgd-ctl-v1`。今回の mon が同じ学習を再現しながら curl 推移を記録する対照。
- GRU: `trysvgd-mirror2-gru-ctl-v1`、`trysvgd-tagr-gru-ctl-v1`。既存結果を使う。木・override・seed の一致を集計時に確認する。
- 最終評価 return（mean_chain / stochastic、GRU は対応する評価モード）、seed ごとの分散、予測損失、curl ratio、所要時間を併記する。curl の減少だけで性能改善とは判定しない。
- 比較ノート: [21-curl-penalty.md](../../experiments/21-curl-penalty.md)。本番の学習結果はまだない。

## 損失・スケジュール・ログ

1 optimizer minibatch につき、再生対象の環境・時刻の組（現在 rollout と有効エピソード prefix）から復元抽出で 8 組、その連鎖の入力段 `0..L-1` から一様に段を選ぶ。各組の全 K 粒子で `mean ||J-Jᵀ||²_F` を評価する。位置と過去の信念を detach して、スコア網のパラメータだけに正則化勾配を流す。初期枝・再帰枝は実際の initial flag に従い、tanh 等の有界化、score_clip、初期 prior を含む実輸送スコアを使う。

constant は pretrain を含む予測更新と PPO に同じ係数を適用する。ramp は pretrain で 0、PPO 第 j 回で `λ × j / N_PPO`、途中の予測では直前の値を保つ。mirror2 は `N_PPO=200`、tagr は `1000`（総 333 / 1293 と区別する）。critic warm-up があればその間は罰則を掛けない。分割再開でも絶対 update 番号から復元する。

罰則有効時の追加列は `score_curl_loss`（係数前）、`score_curl_ratio`、`score_curl_coef`。mon の追加列は `score_curl_ratio` のみ。既定 off では CSV の列・学習・乱数消費を変えない。ratio は粒子ごとの `||J-Jᵀ||_F / ||J||_F` の平均で、通常の反対称成分 `(J-Jᵀ)/2` の割合の **2 倍**。範囲は 0–2、100% を超えても異常ではない。J=0 では 0 とする。

Hutchinson の `mean_p (uᵀJv−vᵀJu)²` は独立 Rademacher u,v を使い、`||J-Jᵀ||²_F` の不偏推定（1/4 や次元による除算なし）。ratio だけは D 回の診断用 VJP で厳密に求めるため、Hutchinson を選んでも診断部分には O(D) の計算が残る。

## 係数をこの水準にした理由

学習済み対照 s10 / s11 の有効エピソード履歴から 128 組 × 16 粒子を CPU で測定したところ、curl の平均は mirror2 0.333–0.712、tagr 4.077–5.568。最終 metrics の `policy_loss + 0.5 value_loss − 0.01 entropy` はそれぞれ約 0.101–0.108、0.0249–0.0305 だった。

依頼の共通係数を維持する。mirror2 の λ=0.01 は総損失の約 3–7%、0.1 は約 33–66% の規模。tagr は λ=0.01 でも約 134–224%、0.1 は約 13–22 倍なので、「弱」は腕間の相対的な呼称であり、PPO に対して弱いとは限らない。ramp は初期の学習自由度を残し、r1 は強い制約側の境界を調べる。損失値の比は勾配の比ではなく、最適係数の保証でもない。

## math の override と実行範囲

次の既存 override の組み合わせで全設定の検証が通る。

```text
--override model.score_implementation=analytic
--override model.belief_cuda_graphs=true
--override ppo.sequence_microbatch_size=512
```

通常の信念輸送は従来の CUDA Graphs を利用でき、curl 用の途中粒子の補助再生と VJP / backward は graphs の外で実行する。CPU ではこの分離と設定をテストしたが、CUDA 実機での capture・性能計測は未実施。seedbatch は penalty / monitor とも明示的に拒否し、energy 版・TD3 への有効化も拒否する。GRU / PF の学習コードには追加しない。

本ディレクトリは投入準備のみ。50 本合計は mirror2 8,325 更新 + tagr 32,325 更新 = 40,650 更新（うち PPO 30,000 更新）。GPU slot 時間の実測値はない。
