# PPO + head-level Self-Imitation

各系列 seed 10–14、ジョブ名末尾は `_s10.sh` ... `_s14.sh` です。

| 系列 | config | campaign | 総更新 |
|---|---|---|---:|
| light Tag/range, SIL ReLU | trysvgdlight_tagr_tf-svgd-sil.json | trysvgdlight-tagr-tf-svgd-sil-v1 | 1293 |
| light Tag/range, 指数重み | trysvgdlight_tagr_tf-svgd-silexp.json | trysvgdlight-tagr-tf-svgd-silexp-v1 | 1293 |
| GRU Tag/range, SIL ReLU | trysvgd_tagr_gru-sil.json | trysvgd-tagr-gru-sil-v1 | 1293 |
| light Car-Flag long, SIL ReLU | trysvgdlight_carflag_tf-svgd-long-sil.json | trysvgdlight-carflag-tf-svgd-long-sil-v1 | 2493 |
| light Tag/range, GAE λ=0.99 | trysvgdlight_tagr_tf-svgd-lam99.json | trysvgdlight-tagr-tf-svgd-lam99-v1 | 1293 |

**Car-Flag の依頼文にある2498更新は現コードと一致しません。** 元 long config は pretrain98 + PPO2000 + 途中のpredict395 = 2493です。スケジューラは最後の PPO ブロックの後には予測5更新を挿入しません。元対照の条件を保つため設定と実行上限を2493に揃えています。

## 仕組みと初期値

`ppo.self_imitation.enabled=false` が既定で、無効時にはバッファ・追加乱数・CSV列・checkpoint replay state はありません。有効時は各 PPO update 後に4回、CPUリングバッファの最大100,000遷移から512件を復元抽出してヘッドのみ更新します。完了エピソードの割引収益 R と収集時の detach 済み actor/critic 条件、行動、実際に生成した拡散鎖を保存します。未完了エピソードは更新境界を越えて保持し、checkpoint にも保存します。ランダム行動の事前学習は保存せず、予測フェーズでは現方策の経験を保存するだけで SIL 更新を行いません。

各 SIL step の直前に全バッファの `A=R-V` を現 critic で再計算し、`A>min_advantage` の遷移だけから `A^0.6` に比例して抽出します。重要度補正は行いません。ReLU 重みは A、指数重みは `min(exp(A/(temperature*std_buffer(A))), max_weight)` です。std は全 replay 遷移、母標準偏差、下限1e-6。`temperature=1`、`max_weight=20` とし exp の前に clip します。指数版も正の A だけを使う、AWR 型の制限付き変種です。

actor係数0.1、value係数0.01、priority exponent0.6、min advantage0が初期提案です。拡散鎖の全段 log-prob を合計して模倣するため、PPO の段平均とは係数の実効規模が異なります。symlog critic では重みと採用判定は raw の R−V、価値損失は正の `symlog(R)-v` の半二乗平均です。SIL 用の値更新でも既存 PPO と同じ target 座標を保ちます。

`c_t` は再計算しないので表現更新により古くなります。GRU の hidden/encoder も SIL では更新しません。gamma の時間減衰は残り、まだ出会っていない成功を replay から作り出すこともできません。これらは完全版 SIL/AWR の再現ではなく、この拡散 PPO の成功再利用仮説を調べるための opt-in 実装です。

有効時のみ `metrics.csv`/`arrays.npz` に `sil_policy_loss`, `sil_value_loss`, `sil_selected_transitions`, `sil_positive_transitions`, `sil_mean_weight`, `sil_buffer_size`, `sil_updates` が増えます。selected は復元抽出を含む学習使用件数、positive は最後の優先度更新時の候補数、損失は係数適用前の成功した SIL step の平均です。対象なし/予測フェーズは損失・採用数0です。

## math での実行

リポジトリルートで、必要な系列だけ選んで投入してください。ここでは一系列を例示します。

```bash
export PYTHONPATH="$PWD/src"
export PYTHONUTF8=1
for seed in 10 11 12 13 14; do
  pjsub "jobs/sil/trysvgdlight_tagr_tf-svgd-sil_s${seed}.sh"
done
```

全25本を投入する場合は `for job in jobs/sil/*.sh; do pjsub "$job"; done`。再開は同じジョブを再投入します。`SB_POMDP_SEGMENT_UPDATES` は各系列の全更新数で、既存の環境・RNG・optimizer と追加 replay/pending が同時に復元されます。

SIL 追加分の GPU 実測はありません。既存見積りでは light Tag/range 10.8–14.4時間/run、GRU 1.1–3.6時間/run。Car-Flag 2493更新に同じ30–40秒/updateを当てはめると20.8–27.7時間/runですが、環境差を無視した粗い参考値です。SIL は最大40万件分の価値再評価と2048件のヘッド学習/updateを追加します。CPU→GPU転送・checkpoint 増大も含め、まず各系列の初期5 PPO更新で時間・メモリを測ってください。
