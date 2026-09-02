# mountain_hike job set — DVRL の Mountain Hike を共通基盤として導入

Igl et al. 2018(DVRL)の Mountain Hike を公式実装(DeathValleyEnv +
mountainHike.yaml)から忠実に移植した `mountain_hike` タスク。belief 手法と
RNN の比較のために設計された課題で、方策勾配(A2C/PPO + GRU)が動作する
ことが文献で確認済み(DVRL, DPFRL)。能動的な情報収集は不要で、位置依存
でない大ノイズ(σ=3、盤面 [-10,10]²)下の filtering が要求される。

- 座標: 論文座標(内部箱 [-1,1]² × box_scale 10)。開始 N((-8.5,-8.5), I)、
  行動ノルム ≤ 0.5、遷移ノイズ std 0.25、観測ノイズ std 3.0(論文の
  σ_o ∈ {0, 1.5, 3} の最難)、75 step、報酬 = 地形 r(x,y) − 0.01‖a‖、
  箱外 −6、goal 報酬なし・終了なし(参照 config どおり)。
- 地形: 3 つの軸平行ガウス尾根の max → 1−(1−Z)^4 → ×4 − 4.5 + (x+y)/4。
  開始付近 ≈ −4.9/step、尾根上 ≈ −0.3/step。
- 参照値(scripted、scratchpad/hike_refs.py): README 末尾の表を参照。

## 手順(リポジトリ root、全 MIG、campaign = segmented resume 可)

```bash
pjsub jobs/mountain_hike/debug_hike.sh
pjsub jobs/mountain_hike/score_hike.sh
pjsub jobs/mountain_hike/alpha_hike.sh
pjsub jobs/mountain_hike/ds_hike.sh
pjsub jobs/mountain_hike/gru_hike.sh
pjsub jobs/mountain_hike/rnn_hike.sh
pjsub jobs/mountain_hike/pf_hike.sh
```

bulk index(method-major × 1 task × seeds [10,11,12]): 1 transformer,
4 alpha, 7 deepsets, 10 gru, 13 rnn, 16 pf(いずれも seed 10)。24h 壁で
止まったら同じファイルを再投入(latest.pt から厳密再開)。

## 参照値(scripted、200 ep、scratchpad/hike_refs.py)

| 方策 | σ=0 | σ=1.5 | σ=3.0 |
|---|---|---|---|
| 静止(開始点に留まる) | −362 | −360 | −360 |
| 真の位置で地形を登る(oracle 上限) | −122 | −122 | −122 |
| 生の観測で地形を登る(記憶なし) | −122 | −157 | **−189** |

σ=3 では「記憶なし」と「完全な filtering」の差が **67**(75 step)あり、
これが belief 学習で埋めるべき、勾配で到達可能な余地。学習方策の位置は
この 3 行の間で読む。
