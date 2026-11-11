# 19 スコア網・方策 / 価値ヘッドを KAN にすると

- **問い**: 関数近似器を MLP から KAN（Kolmogorov–Arnold network、B-spline 基底、grid 8・次数 3、パラメータ数は MLP に一致）に替えると良くなるか（ユーザー 2026-11-11「価値関数やスコアを KAN にするのは意味がある？意味があるなら試してほしい。light-SVGD、GRU、PF の 3 つで」）。見立て: スコア網は低次元入力の局所的な場なので試す理由がある。ヘッドは 32 次元の学習表現が入力で期待は低い。
- **腕（campaign）**: `trysvgdlight-<task>-tf-svgd-kanscore-v1`（`model.energy_network_kind=kan`、light のスコア網）、`trysvgdlight-<task>-tf-svgd-kanhead-v1`、`trysvgd-<task>-gru-kanhead-v1`、`trysvgd-<task>-pf-v3-kanhead-v1`（`model.head_network_kind=kan`、方策 / 価値ヘッド）。対照は各腕の MLP 版。
- **課題**: mirror2（333 update）/ tagr（1293）× 5 seed × 4 腕 = 40 本。seed 10–12 は優先ブロックの末尾、13–14 は最後尾。
- **設定とジョブ**: `../config/trysvgdlight_<task>_tf-svgd-{kanscore,kanhead}.json`、`../config/trysvgd_<task>_{gru,pf_v3}-kanhead.json`、`../jobs/trysvgd_light/*-{kanscore,kanhead}_s*.sh`、`../jobs/trysvgd/*-kanhead_s*.sh`。kanscore は解析スコア（`score_implementation=analytic`）と併用不可なので、math では analytic の override を外した木（v23k）で走らせる。
- **状態**: 2026-11-12 投入。

## 結果の解説

- （未）

## 結論メモ

- （走行中）
