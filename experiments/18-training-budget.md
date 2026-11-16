# 18 学習量を伸ばすと抜けるか（Car-Flag の長い学習）

- **問い**: Car-Flag は 333 update（PPO 41 万 step）で 7 腕 × seed 10–12 の全部が「priest を読まずに片側の旗へ一直線」の 50 % 当てずっぽう（約 −0.4）に止まった（[04](04-chain-start.md)）。原論文（COSIL、off-policy の再帰 SAC）は約 16 万 step・約 16 万回更新、Mountain Hike の DVRL コードは 5,000 万フレームが既定で、我々の予算は 1〜2 桁少ない。PPO の update を 10 倍（200 → 2000、PPO 410 万 step）にすると、light / GRU / PF v3 は局所解から抜けて priest を読むようになるか（ユーザー 2026-11-11「普通に学習不足な気がしてきた」「Car-Flag の長時間軸は投下 OK」）。
- **腕（campaign）**: `trysvgdlight-carflag-tf-svgd-long-v1`（light 素）、`trysvgd-carflag-gru-long-v1`、`trysvgd-carflag-pf-v3-long-v1`。設定は 333 update の腕と `ppo.total_steps`（409,600 → 4,096,000）だけが違う。総 2,498 update = 98 pretrain + 2,000 PPO + 400 予測（25 update ごとに 5）。評価 25 update ごと 10 話、checkpoint 25 update ごと。
- **課題**: Car-Flag × 5 seed × 3 腕 = 15 本。seed 10–12 を先に（Car-Flag ブロックの末尾）、13–14 は最後尾。1 run の見込み: light 8〜30 h（1 update 12〜48 s、旗に早く着く方策ほど速い）、GRU 5〜6 h、PF 7〜8 h。
- **設定とジョブ**: `../config/trysvgdlight_carflag_tf-svgd-long.json`、`../config/trysvgd_carflag_{gru,pf_v3}-long.json`、`../jobs/trysvgd_light/trysvgdlight_carflag_svgd-long_s*.sh`、`../jobs/trysvgd/trysvgd_carflag_{gru,pf_v3}-long_s*.sh`。生成は scratchpad gen_carflag_long.py。
- **判定の指標**: 平均報酬だけでなく、評価軌跡から「priest 帯（0.3〜0.7）を通る割合」「合図を見た step 数」「合図の前後で行動が変わるか」「選んだ旗が真の側に追従するか」を出す（scratchpad carflag_analyze.py、math run1102）。+0.4〜+0.8 に届けば合図を使えている。
- **状態**: 2026-11-11 投入（math、木 v22 に追加）。

## 結果の解説

- 333 update の結果（全腕 −0.4、合図を見ても行動が変わらない）: [04-chain-start.md](04-chain-start.md) の Car-Flag 行。
- 元論文の学習量の調べ: DVRL コード `default.yaml` の `num_frames: 50.0e06`、COSIL の `car_flag_continuous/rnn.yml`（1,000 iteration × 1 話）。

## 結論メモ

- **2026-11-16 22:30 時点（検証 100 話、± 0.10）**: 完走は GRU の seed 11 / 12 だけ（2,493 update = PPO 410 万 step）で、−0.47 / −0.47。10 倍に伸ばしても当てずっぽうの水準から動かない。途中で止めて後ろへ回したもの: GRU seed 10（update 419、−0.47）、PF v3 seed 10–12（863 / 869 / 1071、−0.47）、light seed 10–12（233 / 757 / 806、−0.28 / −0.30 / −0.27）、light + SIL seed 10–12（1797 / 827 / 330、−0.28 / −0.30 / −0.43）。どれも +0.4〜+0.8（合図を使えている水準）に届いていない。
- 文献調査（[29](29-literature-memory-tasks.md) の「Car-Flag について」）で、我々の `CarFlagEnv` は元コードと報酬が違う（元は青い領域を通った話でだけ +1）こと、観測に速度があるので記憶なしでも解ける課題であること、文献でも再帰型 A2C は 400 万 step で 0 のままという報告があることが分かった。学習量ではなく、課題の定義と探索の問題である可能性が高い（私の推察）。
