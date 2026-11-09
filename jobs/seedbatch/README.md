# Seed バッチ学習（独立した実験経路）

`python -m sb_pomdp.seedbatch` は 1 課題・1 設定・複数 seed を 1 プロセスに積みます。
既存の `compare` / `train` は変更しません。既定の light mirror2 設定をそのまま使えます。
このディレクトリのスクリプトは雛形です。ジョブの投入は行っていません。

## 実行

リポジトリルートから、既存の PyTorch 2.3.1 環境で実行します。

```bash
export PYTHONPATH="$PWD/src"
export PYTHONUTF8=1
python3.11 -m sb_pomdp.seedbatch \
  --config config/trysvgdlight_mirror2_tf-svgd.json \
  --campaign-id seedbatch-mirror2 --seeds 10 11 12 13 14
```

出力は `experiment.output_dir/campaigns/<campaign>/shardNNN/full/score_transformer/<task>/seed_XXX/`。
shard 番号は元の設定の `compare --bulk-index` 行列で決まり、`--seeds` の部分選択や並べ替えでは変わりません。
`experiment.output_dir` の既定値は `results` です。campaign までを override すると二重になるため、基底の出力ディレクトリだけを指定します。
既存の manifest / summary / comparison と各 seed の CSV・評価軌跡・v4 checkpoint を出力します。

## math 用の 5 seed 例と分割実行

```bash
export SB_POMDP_CONFIG=config/trysvgdlight_mirror2_tf-svgd.json
export SB_POMDP_CAMPAIGN_ID=seedbatch-mirror2
export SB_POMDP_SEEDS="10 11 12 13 14"
export SB_POMDP_SEGMENT_UPDATES=25
export SB_POMDP_SKIP_MODULES=1  # math の既存 Python/CUDA 環境を利用
export SB_POMDP_PYTHON=python  # 必要に応じて venv の Python を指定
bash jobs/seedbatch/mirror2_5seeds.sh
```

これは 25 update の **1 セグメント**です。同じコマンドを既存のジョブ連鎖から繰り返します。
`--segment-updates` は pretrain / prediction を含む update 数です。PPO の予算は設定どおりです。
`--resume` は未作成の出力なら新規実行、既存なら checkpoint 再開です。完了後の再実行はファイルを変更しません。
既存の `run_shard.sh` は内部で `sb_pomdp.compare` を呼ぶため、そのまま seedbatch にはなりません。
呼び出すジョブをこの雛形に変更してください。既存ジョブ本体は変更していません。
`SB_POMDP_CONFIG` / `SB_POMDP_CAMPAIGN_ID` / `SB_POMDP_SEGMENT_UPDATES` と既存の環境設定を引き継ぎます。

seed ごとの checkpoint は原子的ですが、複数 seed 全体の保存は原子的ではありません。
強制終了で update がずれた場合は明確なエラーになります。遅れている seed を `--seeds` で単独選択し、
不足分の `--segment-updates` で追いつかせてから全 seed を再開してください。
バッチの構成変更では演算順序が変わり、完全な bit 再現は保証しません。
同じ campaign に既存 compare ジョブを同時実行しないでください。雛形の flock は seedbatch ジョブ間を排他します。
この CLI の resume は seedbatch が作成した campaign 用です。既存 compare の run directory の直接引き継ぎは未対応です。

## GPU ベンチ

```bash
export SB_POMDP_CONFIG=config/trysvgdlight_mirror2_tf-svgd.json
export SB_POMDP_PYTHON=python
bash scripts/seedbatch_bench.sh
```

N=1,2,5、各 3 回。予測フェーズを無効にし、1 PPO update の warmup 後に次の 1 update を測ります。
同じプロセス内で N seed を個別に順番に実行する時間も測ります。評価は測定に入りません。
測定には model 構築・checkpoint 読み込み/保存・CSV を含みます。GPU は測定前後に synchronize します。
JSON に全測定値・中央値・スループット比・解決済み設定を保存します。`--skip-single` で個別比較を省略できます。
初回は既定設定の必要メモリと数値を確認してください。GPU の高速化率は CPU の検証では判断できません。

## 対応範囲

full / score_transformer、direct、SVGD / ULA、diffusion、condition 予測、任意の粒子数・潜在次元、
noise / affine / uniform proposal、observation anchor、学習可能なステップ・温度・noise schedule、
数値 / null / log の score bound、initial score prior、GAE / symlog、Adam / AdamW に対応します。
direct の `score_implementation=analytic` も同じ直接スコア式で動きます。

energy、direct_flat / direct_joint、SMI、基準線、TD3/P3O、離散行動、tbptt_1、prior 初期信念、
補助 reward/observation predictor、prediction_only、policy_observation_input、action-value critic、
KAN/basis、network proposal、compile / CUDA Graphs、activation checkpoint はエラーで拒否します。
light 設定の `belief_activation_checkpoint=false` を使います。

学習時の信念・方策・PPO / prediction の全 forward と backward・再条件付けは seed 次元でまとめます。
環境処理、乱数生成、評価、truncation bootstrap と最終 bootstrap 値、最適化器の step / clipping、ファイル保存は seed ごとです。
CPU 上の数値比較は浮動小数点誤差を許容します。同じ seed バッチ構成での分割再開は bit 一致を検証します。
