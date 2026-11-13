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
- **状態**: 2026-11-14、設定 4 個・ジョブ 20 本・テストを準備、CPU の短い試走（4 update、mirror2 / tagr / Car-Flag）で損失が有限であることを確認。同日 04:20 に math へ 20 本を投入（木 v29、ユーザー「v2の投入OK」）: seed 10–12 の 12 本は待ちの先頭（tps → mirror2 → Car-Flag → tagr）、seed 13–14 の 8 本は最後尾。math 側の投入は `run1102/add_v2.py`。結果待ち。

## 結果の解説

未取得。

## 結論メモ

- （結果待ち）
