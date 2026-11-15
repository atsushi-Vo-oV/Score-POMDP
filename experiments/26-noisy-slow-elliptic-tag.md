# 26 観測ノイズを強くした楕円距離の鬼ごっこ（tagresn）で、信念の差は出るか

- **問い**: 距離観測のノイズを強くすると、1 回の観測では鬼の位置が決まらず、複数時刻の観測を統合する必要が増す。この条件で、粒子で信念を持つ腕（light、PF v3）は GRU に対して有利になるか。腕の順位は [25](25-slow-elliptic-tag.md)（同じ課題でノイズが元のまま）から変わるか（ユーザー 2026-11-15「観測ノイズの大きさを楕円距離鬼ごっこのノイズが強い版でも各モデルを比較して」）。
- **課題 tagresn**: tagres（楕円距離、鬼の速さ 0.5〜0.7 倍、環境 64、粒子 64、2420 update）と同じで、距離のノイズの標準偏差だけを 0.05 + 0.20 d から **0.10 + 0.40 d** に 2 倍にした（d = 楕円距離。`range_noise_base: 0.10`、`range_noise_scale: 0.40`）。距離 0.5 で標準偏差 0.15 → 0.30、距離 1.0 で 0.25 → 0.50（円盤の半径は 1）。
- **私が選んだ点**（ユーザーの指定は「ノイズが強い版」だけ）: (1) 鬼の速さは遅い版（tagres）を土台にした。元の速さの tagre では学習した方策が「何もしない」水準に固まり、腕の差が読めないため。(2) 倍率は 2 倍。3 倍（距離 1.0 で 0.75）は観測がほぼ使えなくなる。どちらも `scripts/tagres_generate.py` の 1 行で変えられる。
- **目安**: 鬼の動きと報酬は tagrs / tagres と同じなので、状態が全部見える方策の値は [24](24-slow-tag.md) の表と共通（何もしない 4.1〜4.5、旋回し続ける 8.1、全状態の反発 13.7、先読み 18.4）。
- **腕（campaign）**: light v2 の素 `trysvgdlight2-tagresn-tf-svgd-env64-k64-v1`、light v1 の素 `trysvgdlight-tagresn-tf-svgd-env64-k64-v1`、GRU `trysvgd-tagresn-gru-env64-k64-v1`、粒子フィルタ `trysvgd-tagresn-pf-v3-env64-k64-v1`。幅は tagre と同じ。seed 10–14、20 本。
- **注意**: tagre と同じく **FLOPs は未一致**（K = 64 の light は GRU の約 9 倍）。
- **設定とジョブ**: `../config/trysvgdlight2_tagresn_tf-svgd.json`、`../config/trysvgdlight_tagresn_tf-svgd.json`、`../config/trysvgd_tagresn_{gru,pf_v3}.json`、`../jobs/tagresn/*_s{10..14}.sh`、生成 `scripts/tagres_generate.py`、テスト `src/tests/test_tagres.py`（tagre との差が鬼の速さとノイズの 2 キーだけ、v2 は加えて既定 v2 の 5 キーだけであること）。
- **状態**: 2026-11-15、設定 4 個・ジョブ 20 本・テストを準備。2026-11-15 12:36 に math へ投入（木 v33、ユーザー「投入はおｋ」）: seed 10–12 は待ちの先頭、seed 13–14 は最後尾。結果待ち。

## 結果の解説

未取得。

## 結論メモ

- （結果待ち）
