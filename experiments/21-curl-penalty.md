# 21 SVGD-light を soft な curl penalty で勾配場に近づける

- **問い**: スコアを直接出す light の局所的な非対称ヤコビアンを抑えると、方策 return と信念の質は改善するか。初めから一定の制約を掛ける場合と、PPO の進行に合わせて増やす場合に違いはあるか。
- **腕（campaign）**: `trysvgdlight-<task>-tf-svgd-curl-{mon,c001,c01,r01,r1}-v1`。mon は λ=0 + monitor、c001 / c01 は constant λ=0.01 / 0.1、r01 / r1 は ramp 0→0.1 / 1.0。GRU 対照は既存 `trysvgd-<task>-gru-ctl-v1`、素の light は既存 `trysvgdlight-<task>-tf-svgd-ctl-v1`。
- **課題**: mirror2（333 総更新、PPO 200）、tagr（1293 総更新、PPO 1000）、各 seed 10–14。2 課題 × 5 腕 × 5 seed = 50 本。
- **設定とジョブ**: `../config/trysvgdlight_<task>_tf-svgd-curl-<arm>.json`、`../jobs/curl/trysvgdlight_<task>_tf-svgd-curl-<arm>_s{10..14}.sh`。全設定・campaign 対応とスケジュールの定義は [jobs/curl/README.md](../jobs/curl/README.md)。実装 `src/sb_pomdp/score_curl.py`、テスト `src/tests/test_score_curl.py` / `test_curl_experiments.py`。
- **状態**: 2026-11-12、実装・CPU 検証・設定 10 個・ジョブ 50 本を準備。未投入。既定 off と mon の学習不変性、および変更前 HEAD との off 時の一致を確認。CPU の小設定では penalty のみの最適化で初期網・再帰網とも curl ratio が低下した。

## 結果の解説

学習済み対照 checkpoint の保存履歴から、環境・時刻・連鎖の段をランダム抽出して CPU 測定（各 128 組、1 組 16 粒子）。有界化と初期 −x を含む輸送スコアに対する厳密な値。

| task / seed | 平均 `||J-Jᵀ||²_F` | 平均 ratio | 最終 PPO 総損失の参考値 |
|---|---:|---:|---:|
| mirror2 / 10 | 0.711687 | 0.981564 | 0.107933 |
| mirror2 / 11 | 0.333245 | 0.776524 | 0.100599 |
| tagr / 10 | 4.077093 | 0.790814 | 0.030462 |
| tagr / 11 | 5.567718 | 1.456971 | 0.024862 |

ratio は `||J-Jᵀ|| / ||J||` なので、通常の反対称成分 `(J-Jᵀ)/2` の割合の 2 倍である。過去の 35–63% とそのまま比較しない。PPO 総損失は最終 metrics の policy + 0.5 value − 0.01 entropy で、curl 測定点と同一ミニバッチではない。tagr の λ=0.01 も PPO に対しては強い可能性があり、return と curl の両方で判断する。

curl は BPTT から切り離した座標・過去信念で評価する。1 ミニバッチ 8 組、D≤4 は exact、それ以上は Hutchinson 4 probes。ランダム抽出は seed と絶対 update 番号に基づく専用 generator。定常制約は予測 pretrain にも適用、増加制約は pretrain 中 0 で、その後は PPO 進行のみで増やす。

### 2026-11-13 の結果（seed 10〜12、mean_chain 評価 10 話）

| 課題 | seed | mon（λ = 0） | c001 | c01 | r01（0→0.1） | r1（0→1） |
|---|---|---:|---:|---:|---:|---:|
| mirror2 | 10 | −396.6 | −373.8 | −346.0 | −384.8 | −383.0 |
| mirror2 | 11 | −342.0 | −327.9 | −343.8 | −325.5 | −321.7 |
| mirror2 | 12 | −344.2 | −351.5 | −354.8 | −361.3 | −399.9 |
| mirror2 | 平均 | −360.9 | −351.1 | −348.2 | −357.2 | −368.2 |
| tagr | 10 | 2.4 | 2.8 | 4.0 | 3.0 | 1.9 |
| tagr | 11 | 2.2 | 2.4 | 2.4（1275） | 2.3 | 2.2 |
| tagr | 12（途中） | 2.6（850） | 3.0（900） | 2.5（850） | 3.1（600） | 3.2（550） |

括弧内は評価時点の update（無印は完走）。mon の seed 10 / 11 は素の light（`trysvgdlight-<task>-tf-svgd-ctl-v1`、mirror2 −396.6 / −342.0、tagr 2.4 / 2.2）と一致する（診断を足すだけで学習は不変）。反対称の割合（ratio）は罰則なしで 0.7〜1.2、罰則ありで 0.05〜0.4。

罰則は反対称成分を確かに下げるが、return は変わらない。seed 13〜14 と Car-Flag は後ろのキュー。

## 結論メモ

- soft な局所 curl 制約を PPO と予測フェーズの両方に実装できた。既定 off は既存学習と同一、mon は診断だけを追加する。
- 一定・増加のどちらでも ratio は 0.05〜0.4 まで下がるが、mirror2・tagr の return は素の light と同じ帯。スコアが勾配場でないことは、いまの頭打ちの原因ではない。
- 罰則の低下は局所的な勾配場への近さを示すだけで、正規化可能な分布・信念の正しさ・SVGD の収束を保証しない。
