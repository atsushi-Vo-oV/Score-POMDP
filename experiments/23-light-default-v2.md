# 23 light TF-SVGD の既定 v2（一様開始・アンカーなし・非有界）は v1 と比べてどうか

- **問い**: light TF-SVGD（スコアを直接出す Transformer 読み出し + SVGD: Stein Variational Gradient Descent）の素の設定を、ユーザー指定（2026-11-14「一旦 SVGD のデフォルトとして次に変更できる？ 実験セット群を ver.2 か 3 にしても OK」）の v2 に替える。v2 は v1 と比べて良いか、少なくとも悪くならないか。
- **v2 の定義**（v1 = try-svgd-light の `tf-svgd` からの差分はこの 5 キーだけ）:
  1. 連鎖の開始は t = 0 も t ≥ 1 も一様分布 U(−√3, √3): `langevin_proposal: "uniform"`、`langevin_initial_noise: "uniform"`
  2. 観測アンカーなし（t = 0 と t ≥ 1 の粒子の尺度をそろえる）: `langevin_observation_anchor: false`
  3. スコアの有界化なし: `direct_score_bound: null`。あわせて t = 0 の −x も外す: `initial_score_prior: 0.0`（2026-11-10 の指示「非有界なやつはついでに初期値の −x も消して」に合わせた。−x を残す版が要るなら 1 キーで作れる）
- **腕（campaign）**: `trysvgdlight2-<task>-tf-svgd-v1`（v2 の素）vs v1 の素 `trysvgdlight-<task>-tf-svgd-v1`（tagr は `-ctl-v1` と `-curl-mon-v1` の seed 12）。1 点だけ変えた既存の腕が切り分けの参照になる: 開始分布 = `-uniform-init-u0-v1`（全時刻一様、アンカーあり。[04](04-chain-start.md)）、アンカー = `-noanchor-v1`（[05](05-t0-scale.md)）、有界化 = `-nobound-v1`（[17](17-unbounded-score.md)）。基準線は GRU / PF v3。
- **課題**: mirror2（333 update）/ tps（213）/ tagr（1293）/ Car-Flag（333）× seed 10–14 = 20 本。
- **設定とジョブ**: `../config/trysvgdlight2_<task>_tf-svgd.json`、`../jobs/trysvgd_light2/trysvgdlight2_<task>_svgd_s{10..14}.sh`、説明 [jobs/trysvgd_light2/README.md](../jobs/trysvgd_light2/README.md)、テスト `src/tests/test_light_v2_default.py`。コードの既定値（`config.py`）は変えていない（過去の run の再現性のため）。以後の新しい比較は、指定がない限り `trysvgdlight2_<task>_tf-svgd.json` を素にして作る（接頭辞 `trysvgdlight2-`）。
- **事前に分かっていること**: 3 点のうち、一様開始と非有界は単独では v1 と同じ帯だった。アンカーなしは単独で mirror2 を悪化させた（noanchor −784、5 seed 中 2 崩壊。v1 −368）。tps は少し悪化（−308 対 −287）、tagr は少し上（中央値 3.37 対 2.44）。3 点の組み合わせは未検証。
- **状態**: 2026-11-14、設定 4 個・ジョブ 20 本・テストを準備、CPU の短い試走（4 update、mirror2 / tagr / Car-Flag）で損失が有限であることを確認。同日 04:20 に math へ 20 本を投入（木 v29、ユーザー「v2の投入OK」）: seed 10–12 の 12 本は待ちの先頭（tps → mirror2 → Car-Flag → tagr）、seed 13–14 の 8 本は最後尾。math 側の投入は `run1102/add_v2.py`。2026-11-15 朝に mirror2 / tps / Car-Flag の seed 10–12 が完走、tagr は走行中（seed 10 が 190 / 1293 update）。

## 結果の解説

2026-11-15 時点。数値は全 run 共通の 100 話の検証平均（決定的方策、環境 seed 2,000,000〜。seed 10–12 の平均、括弧は seed ごと）。

| 課題 | v1 の素 | **v2 の素** | uniform（t ≥ 1 だけ一様） | noanchor | nobound | GRU | PF v3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mirror2 | −357（−391 / −329 / −353） | **−918**（−1651 / −789 / −312） | −364 | −1020（−1633 / −1088 / −338） | −391 | −982 | −1246 |
| tps | −287（−269 / −298 / −293） | **−309**（−292 / −328 / −309） | −291 | −302（−281 / −329 / −295） | — | −290 | −315 |
| Car-Flag | −0.31（−0.28 / −0.33 / −0.31） | **−0.47**（−0.47 / −0.47 / −0.47） | −0.29 | — | −0.35 | −0.49 | −0.48 |
| tagr | 3.42 | 走行中 | 3.44 | 3.26 | 3.56 | 4.27 | 4.10 |

- **mirror2**: v2 は seed 10 と 11 で崩れた。崩れる seed も崩れ方も noanchor 単独（−1633 / −1088 / −338）と同じで、uniform 単独と nobound 単独は v1 と同じ帯にいる。3 点のうち効いているのは観測アンカーを外したこと。
- **tps**: v2 は v1 より 22 悪い（seed 間の標準誤差 10）。noanchor 単独（−302）と同じ向き。
- **Car-Flag**: v2 は 3 seed とも −0.47 で、GRU・PF v3 と同じ「旗の色を当てずっぽう」の水準に落ちた。v1 は −0.31、uniform 単独 −0.29、nobound 単独 −0.35 なので、ここでも残る候補はアンカーなし（Car-Flag の noanchor 単独は未実行）か 3 点の組み合わせ。1 run の 100 話の標準誤差が 0.10 あるので、v1 との差 0.16 は seed 3 本がそろって同じ値であることが根拠。
- **64 環境では崩れていない**: 集団探索（[22](22-population-exploration.md)）の対照 `ppo` 腕は同じ v2 の素を 64 環境 × 128 step で学習したもので、mirror2 は −329 / −309 / −305、Car-Flag は −0.37 / −0.25（seed 12 は未完走）。こちらは学習器内の評価（検証の 100 話ではない）で、環境 step が 4 倍なので v2 どうしの厳密な比較ではないが、v2 が「表現として悪い」のではなく「16 環境の学習で不安定」である可能性を示す。v1 の 64 環境版（`popx-<task>-ppo-v1`）はキューの最後尾にあり未実行。

## 結論メモ

- 完走した 3 課題すべてで v2 は v1 より悪い: mirror2 −918 対 −357（2 / 3 seed が崩壊）、tps −309 対 −287、Car-Flag −0.47 対 −0.31（検証 100 話、seed 10–12）。tagr は走行中。
- 悪化の原因は 3 点のうち観測アンカーなし。mirror2 では noanchor 単独と同じ seed が同じ値まで崩れ、uniform 単独・nobound 単独は v1 と同じ帯。
- 同じ v2 を 64 環境で学習した集団探索の `ppo` 腕は mirror2 −314・Car-Flag −0.31 前後で崩れていない（学習器内の評価）。16 環境での不安定さが疑わしく、v1 の 64 環境版との比較で確かめられる。
