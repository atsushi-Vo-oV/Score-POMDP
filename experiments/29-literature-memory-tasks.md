# 29 文献の記憶課題で、light は記憶を使えるようになるか

- **問い**: 方策勾配（PPO / A2C）+ 再帰型（GRU / LSTM）で記憶が働くと報告されていて、記憶なしでは解けないと分かっている課題を、文献に近い学習量で走らせる。同じ PPO の枠組みで (1) 我々の GRU は文献どおり記憶を学ぶか（陽性対照）、(2) light TF-SVGD は記憶を使えるようになるか（ユーザー 2026-11-16「一旦多峰性はなくていいので、方策勾配系での信念学習で記憶が必要な既存研究を持ってきて(GRUのやつが参考になるかな？)、参考文献の類似ステップで比較して記憶が機能するか試してほしい」）。
- **背景**: 評価時の記憶の切除（[analyses.md](analyses.md)、2026-11-16）で、light は mirror2 / tagr / tagrs / Car-Flag のどれでも「その話に固有の持ち越し」をほとんど使っていなかった。ただしこれらの課題は記憶の得が小さいか（tps、mirror2）、記憶なしでもかなり届く（tagrs の 9〜10）。記憶が決定的な課題で確かめる必要がある。

## 課題と出典

| 課題 | 中身 | 必要な記憶 | 出典 | 文献の学習量 | 文献の値（再帰型 / 記憶なし） |
|---|---|---|---|---|---|
| `cpv` | CartPole、速度を隠す（位置と角度だけ）、500 step、行動 2 値 | 直前 1〜2 step（速度の推定） | Stable-Baselines3 (SB3) の CartPoleNoVel-v1、POPGym の StatelessCartPole、Pleines et al. 2022 | SB3 10 万、Pleines 409,600、POPGym 1,500 万 step | SB3（10 万 step、公開 run を集計）: LSTM 中央値 493 / 記憶なし 44。POPGym: GRU 1.000 / MLP 0.72〜0.27（= 約 150 step で倒れる） |
| `pdv` | 振り子の振り上げ、角速度を隠す（cos θ, sin θ だけ）、200 step、連続トルク | 直前 1〜2 step | POPGym の StatelessPendulumEasy、SB3 の PendulumNoVel-v1 | POPGym 1,500 万（学習率 5e-5）、SB3 10〜20 万 | POPGym: GRU 0.913 / MLP 0.448（生の収益で −142 / −898）。SB3（10〜20 万 step）: LSTM 中央値 −481（−1447〜−185、未収束）/ 記憶なし −1095 |
| `pdvn` | `pdv` + 観測ノイズ 0.1（[−1, 1] に切る） | 数 step の平滑化 | POPGym の NoisyStatelessPendulumEasy | 1,500 万 | GRU 0.894 / MLP 0.387 / 4 フレーム重ね 0.832（生で −172 / −998 / −273） |
| `ml10` | bsuite の memory length N = 10: 最初の観測にだけ出る ±1 を、10 個目の行動で答える（正解 +1、不正解 −1、他は 0） | 9 step の保持 | Osband et al. 2020（bsuite `memory_len`） | 1 万話 = 10 万 step | A2C + RNN は N ≤ 30 で解け、N > 30 で当てずっぽう。再帰なしは N > 1 で学べない（記憶なしの期待値は厳密に 0） |
| `ml30` | 同、N = 30 | 29 step の保持 | 同 | 1 万話 = 30 万 step | 同（N = 30 は A2C + RNN が解ける上限） |

- POPGym の値は「最良 epoch の平均収益」を [−1, 1] に正規化したもの。振り子の生の収益への換算は (値 − 1) × 200 × 8.1368。
- SB3 の記憶なし / フレーム重ねの値は公表された表ではなく、公開 W&B project `sb3/no-vel-envs` の 77 run の最終評価を調査時に集計したもの。
- 環境の定義は元コードと照合した: 振り子は gymnasium の Pendulum（g = 10、dt = 0.05、最大トルク 2、開始 θ ~ U(−π, π)・θ̇ ~ U(−1, 1)）と同じ式で、既存の `MaskedPendulumEnv` がそのまま使える。POPGym のノイズ版は観測を [−1, 1] に切るので、選択肢 `observation_clip`（既定 false = 従来どおり）を足した。CartPole は既存の `MaskedCartPoleEnv`（観測は x / 2.4 と θ / 12°）。memory length は新しい `MemoryChainEnv`（課題名 `memory_chain`、長さ = `horizon`）で、論文の定義（各話 N step、c₁ ~ Unif{−1, +1}、r_N = +1 if a_N = c₁ else −1）に沿って実装した。bsuite の元コード `memory_chain.py` は取得できず未照合（観測の並びは記憶に基づく）。
- **外した候補**: MountainCarContinuousNoVel（SB3、30 万 step、記憶なし −99.9 / LSTM 92）は差が最も鮮明だが、疎な報酬で gSDE 探索に頼る設定なので、失敗しても記憶のせいか探索のせいか分からない。Mountain Hike（DVRL、DPFRL）は記憶なしの基準線が文献になく、DVRL と RNN の差も 3〜11 しかない。Car-Flag は下の「Car-Flag について」。

## 腕

| 腕 | campaign | 役割 |
|---|---|---|
| light v2 | `trysvgdlight2-<task>-tf-svgd-v1` | 既定 v2（一様開始・アンカーなし・非有界） |
| light v1 | `trysvgdlight-<task>-tf-svgd-v1` | 既定 v1（N(0, I) 開始・t = 0 の観測アンカー・有界） |
| light v1 prev-init | `trysvgdlight-<task>-tf-svgd-prev-init-v1` | 鎖を前時刻の粒子から始める。記憶の切除で唯一、持ち越しに依存していた腕（mirror2 で差し替え −133〜−416） |
| GRU | `trysvgd-<task>-gru-v1` | 陽性対照。文献の水準に届けば、この枠組みで記憶は学べる |
| PF v3 | `trysvgd-<task>-pf-v3-v1` | 粒子フィルタ（DPFRL 型） |
| MLP | `trysvgd-<task>-mlp-v1` | 記憶なしの下限（現在の観測だけ。前の行動も見ない。予測フェーズなし） |
| oracle | `trysvgd-<task>-oracle-v1` | 全状態を見る MLP = この PPO での上限（`cpv` / `pdv` / `pdvn` だけ。予測フェーズなし） |

GRU と PF v3 の幅は、課題ごとに light TF-SVGD の行動 1 step の FLOPs に合わせた（`cpv` 1.185 MFLOP、振り子 0.894、memory length 0.838）。MLP と oracle は合わせていない（約 1 万パラメータ）。

## 学習量

- `cpv` / `ml10` / `ml30`: いつもの 333 update（PPO 409,600 step = 16 環境 × 128 step × 200 update、+ 予測 133 update）。Pleines の 409,600 step と同じで、SB3 の 10 万・bsuite の 10〜30 万 step より多い。
- `pdv` / `pdvn`: 鬼ごっこと同じ 1293 update（PPO 2,048,000 step）。SB3 の 10〜20 万 step では LSTM が収束しておらず、POPGym の 1,500 万 step は学習率 5e-5（我々の 1/5）での値。環境 step は POPGym の約 1/7 だが、学習率 × 勾配 step 数は同程度（POPGym 5e-5 × 約 5.5 万、我々 2.5e-4 × 1.6 万）。GRU が文献の水準に届かなければ延長する。
- PPO の設定は他の実験と同じ（γ 0.99、λ 0.95、学習率 2.5e-4、4 epoch、minibatch 512）。POPGym も γ 0.99。

## 判定

1. **収益**: MLP（下限）と oracle / 文献の再帰型の値の間のどこに来るか。GRU が文献の水準に届くこと（陽性対照）を先に確かめる。
2. **記憶の切除**（`run1102/viewer/eval_memory.py`）: 持ち越す状態を別の話のものに差し替えたときの落ち幅。memory length では、記憶を使っていれば差し替えで正答率が 50 % に落ちる。
3. 読み方: GRU も届かない → 枠組み（PPO の設定、拡散方策、予測の交互学習）の問題。GRU は届き light だけ届かない → light の信念更新の問題で、短い記憶（`cpv` / `pdv`）と長い保持（`ml10` / `ml30`）のどちらで落ちるかが次の手掛かり。

## 設定とジョブ

- 設定 33 個: `../config/trysvgdlight{,2}_<task>_tf-svgd.json`、`../config/trysvgdlight_<task>_tf-svgd-prev-init.json`、`../config/trysvgd_<task>_{gru,pf_v3,mlp,oracle}.json`。
- ジョブ 165 本: `../jobs/memref/*_s{10..14}.sh`（5 課題 × 6〜7 腕 × 5 seed）。生成 `scripts/memref_generate.py`、テスト `src/tests/test_memory_reference_tasks.py`。
- **状態**: 2026-11-16、実装・設定・ジョブ・テストを準備。未投入（投入 OK 待ち）。

## Car-Flag について（調査で分かったこと）

- 文献の結果が割れている: Baisero & Amato 2022（離散行動、100 step）では再帰型 A2C は 400 万 step でも収益 0、Nguyen et al. 2022（CoSIL、連続行動、160 step）の付録では再帰型 PPO が約 25 万 step で成功率 1.0。
- CoSIL の元コード（`corl22` ブランチの `car_flag_continuous.py`）は、**青い旗の領域を一度通った話でだけ** 緑の旗に +1 を出す。我々の `CarFlagEnv` は通らなくても +1 を出すので、左へ一直線の当てずっぽうが期待値 0 になる（元コードでは −0.5）。
- 観測に速度が入っているので、「領域を出たあとは動いている向きにそのまま進む」という記憶なしの方策でも解ける。記憶の判定には向かない。

## 結果の解説

未取得。

## 結論メモ

- （未投入）
