# 25 鬼の足を遅くした楕円距離の鬼ごっこ（tagres）で、信念の差は出るか

- **問い**: 鬼を遅くした鬼ごっこ（tagrs、[24](24-slow-tag.md)）では学習が進み、PF v3 > GRU > light v2 の差が出た。観測を楕円距離（体を回して見回さないと方向が分からない能動的センシング、[10](10-tagre-elliptic.md)）にした版でも同じか。見回す必要がある分、信念を持つ腕（light、PF v3）は GRU に対して有利になるか（ユーザー 2026-11-15「あと、楕円距離鬼ごっこも遅くした実験もお願い」）。
- **課題 tagres**: tagre（`observation_mode=elliptic_ranges`、前方 1.0・横 0.6、鬼 3 体、horizon 200、1 step 0.1、環境 64、粒子 64、2420 update = 25 + 2000 + 395）と同じで、鬼の追跡の速さだけをプレイヤー最大速度の 0.8〜1.0 倍から **0.5〜0.7 倍**にした（`ghost_speed_min: 0.5`、`ghost_speed_max: 0.7`。tagrs と同じ速さ）。
- **目安**（鬼の動きと報酬は tagrs と同じなので、状態が全部見える方策の値は [24](24-slow-tag.md) の表と共通。リターン）: 何もしない 4.1〜4.5、旋回し続ける 8.1、全状態の反発 13.7、先読み 18.4。
- **腕（campaign）**: light v2 の素 `trysvgdlight2-tagres-tf-svgd-env64-k64-v1`、light v1 の素 `trysvgdlight-tagres-tf-svgd-env64-k64-v1`（ユーザー 2026-11-15「いったんデフォルトは v2 にするものの、v1 の比較も続けてください」）、GRU `trysvgd-tagres-gru-env64-k64-v1`、粒子フィルタ `trysvgd-tagres-pf-v3-env64-k64-v1`。幅は tagre と同じ。seed 10–14、20 本。
- **注意**: tagre と同じく **FLOPs は未一致**（K = 64 の light は GRU の約 9 倍: light 20.0 / PF v3 6.4 / GRU 2.16 MFLOP）。tagre の既存の結果は seed 10・11 の 2 本だけ。
- **設定とジョブ**: `../config/trysvgdlight2_tagres_tf-svgd.json`、`../config/trysvgdlight_tagres_tf-svgd.json`、`../config/trysvgd_tagres_{gru,pf_v3}.json`、`../jobs/tagres/*_s{10..14}.sh`、生成 `scripts/tagres_generate.py`、テスト `src/tests/test_tagres.py`（tagre との差が鬼の速さだけ、v2 は加えて既定 v2 の 5 キーだけであること）。
- **状態**: 2026-11-15、設定 4 個・ジョブ 20 本・テストを準備。未投入。

## 結果の解説

未取得。

## 結論メモ

- （結果待ち）
