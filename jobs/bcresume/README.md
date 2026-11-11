# BC actor からの新規 PPO campaign

`model.init_from_checkpoint` は重みだけを読む初期化です。optimizer、update、環境、RNG は新しい seed から開始します。通常の分割再開は既存の `--resume` で行い、その際は初期 checkpoint/BC asset を読み直さず、価値ヘッドも再初期化しません。

| 条件 | actor | belief/encoder | critic | 更新数 | campaign |
|---|---|---|---|---:|---|
| orig | 元 checkpoint | 凍結 | 元重み、warm-up なし | PPO 100 | bcresume-tagr-orig-v1 |
| bc | BC | 凍結 | 元重み、critic のみ10更新 | 10 + PPO 100 | bcresume-tagr-bc-v1 |
| bcreset | BC | 凍結 | 新規構築時の初期値、critic のみ10更新 | 10 + PPO 100 | bcresume-tagr-bcreset-v1 |

各条件に seed 10、11 の2設定・2ジョブがあります。

| seed | config | job | 出力 shard |
|---|---|---|---|
| 10 | `config/bcresume_tagr_s10_{orig,bc,bcreset}.json` | `jobs/bcresume/s10_{orig,bc,bcreset}.sh` | shard001 |
| 11 | `config/bcresume_tagr_s11_{orig,bc,bcreset}.json` | `jobs/bcresume/s11_{orig,bc,bcreset}.sh` | shard002 |

各 config は単一 seed のため `SB_POMDP_BULK_INDEX=1`。保存先を選ぶ `SB_POMDP_SHARD_INDEX` と独立にしました。従来ジョブは追加変数を指定しなければ従来どおりです。総環境 step は orig 204,800、BC 2条件225,280です。

予測フェーズはありません。`prediction_pretrain_steps=0`、`prediction_interval_updates=1000000`、`prediction_phase_updates=5` としています。大きい interval は元 checkpoint の predictor 構造を保持するためで、100 PPO 更新内には予測更新を一度も挿入しません。actor と V 以外の凍結パラメータは optimizer から除きます。warm-up 中は V ヘッドだけを更新します。

評価は PPO 10更新ごと、mean_chain/stochastic 各10話です。orig は総更新10,20,...,100、BC は総更新20,30,...,110です。warm-up 10更新には PPO 更新を含めません。

## math での実行

変更一式と `assets/bc_actors/` を反映した math のリポジトリルートで、初期 checkpoint の絶対パスが存在することを確認します。config 内のパスは依頼指定の v16-20261107 配下です。

```bash
export PYTHONPATH="$PWD/src"
export PYTHONUTF8=1
python3.11 -c 'from pathlib import Path; from sb_pomdp.config import load_config; [load_config(p) for p in Path("config").glob("bcresume_tagr_s*.json")]'
for seed in 10 11; do
  for condition in orig bc bcreset; do
    pjsub "jobs/bcresume/s${seed}_${condition}.sh"
  done
done
```

ジョブは `genkai_production.sh` に既存の segment/resume 処理を委譲します。途中終了時は同じジョブを再投入してください。完了済み campaign は再学習しません。ジョブの同時実行は既存の shard lock で防ぎます。

費用は過去の light 30–40秒/update を機械的に使うと、100更新0.83–1.11時間、110更新0.92–1.22時間、6本合計5.33–7.11 slot-hours。凍結による速度変化・評価・I/O・同居負荷は未計測であり、実行後の最初の5更新で見直してください。

## BC asset の規約

11月10日の `followup_bc.py` は保存前に `W_c <- W_c/xs`, `b <- b-W_c(original)@(xm/xs)` を実行しています。この履歴形式は `meta.normalization_folded` がなくても折り込み済みと解釈します。二重変換しません。新たに未折り込み重みを保存する場合に限り `meta.normalization_folded=false` を指定してください。統計の形・有限性・正の xs、ネットワーク構造、拡散 schedule/action bounds を検証します。
