# 実験監査で確認された懸念と修正依頼（2026-08-18）

本書は Belief-Score-Based-Model の全数監査（9領域 + 発見事項ごとの独立検証）で
**確認済み（CONFIRMED）** となった問題の修正依頼書である。各項目は実コードの
file:line を検証済みで、記載の行番号は 2026-08-18 時点の source に対応する。

## 最重要の前提制約（修正作業前に必ず読むこと）

1. **production campaign（80 shards, seeds 10–14）が Genkai に投入済みである。**
   `src/sb_pomdp/` 配下の **どの .py を1バイトでも変更すると source_sha256
   （`train.py` の `runtime_metadata` が `src/sb_pomdp/*.py` 全体から計算）が変わり、
   投入済み chained shard の resume が `compare.py:512-516` の
   「resumable run source code differs」ガードで恒久拒否される。**
   したがって修正は次のいずれかの運用を前提とする:
   - (a) 現 campaign の完了/中止後に適用する（推奨）、または
   - (b) Genkai 側の実行 tree には配布せず、ローカルの集計系
     （`aggregate.py` / `fetch_results.py` など学習に関与しないファイル）のみ先行適用する。
     ※ ただし aggregate.py も source hash 照合を行うため、照合対象の意味を変えないこと。
2. **科学設定（報酬・GAE・環境・config スキーマ・checkpoint format v4）を変える修正は、
   実行中 campaign の結果と混在させてはならない。** 該当項目（F-1, F-2）は
   「ユーザーの判断待ち」と明記した。Codex は勝手に規約を変更しないこと。
3. 既存テストが現行仕様を固定している箇所がある
   （例: `src/tests/test_buffer.py::test_done_transition_does_not_bootstrap`）。
   仕様変更を伴う修正ではテストの更新を同一 PR に含めること。
4. 各修正には回帰テストを追加すること。テストは `src/tests/` に置き、`uv run pytest` で
   全 suite（現在 236 passed / 2 skipped）が通ることを確認する。

---

## A. 修正を依頼する項目（campaign 運用の堅牢性 — 科学設定を変えない）

### A-1. [major] resume 時の CSV 重複行で chain shard が恒久 wedge する
- 場所: `src/sb_pomdp/train.py:1544`（episodes.csv append）、`:1673`（metrics.csv）、
  `:1730`（evaluation.csv）が **同 update の checkpoint 保存（`:1763-1767`）より先に fsync される。**
- 問題: kill がこの窓に入ると、resume 後に同一 update の評価行が二重追記される。
  final update（250）で起きると `compare.py:191-198` の
  「exactly one stochastic and one deterministic-readout row」チェックに全再投入が失敗し、
  さらに `train.py:1233-1237`（target > resume update 要求）により completed へ到達不能になる。
- 修正方針: resume 時に checkpoint の update 番号より新しい CSV 行を検出して
  切り詰める（metrics.csv には既に「missing/empty 時のみ再構築」`train.py:1332-1333` が
  あるので、これを「checkpoint update より先の行の削除」まで拡張する）。
  evaluation.csv / episodes.csv にも同じ切り詰めを適用する。
  atomic に行うこと（既存の temp-file replace ヘルパー `artifacts.py` を利用）。
- 受け入れ条件: 「eval 行追記後・checkpoint 保存前に kill → resume」を模したテストで、
  重複行が発生せず `_read_final_evaluations` が成功すること。

### A-2. [major] completed shard への余分な segment 投入が manifest を破壊する
- 場所: `src/sb_pomdp/compare.py:444-461`（metadata を initializing で上書き）、
  `:491-506`（manifest を initializing / runs=[] で上書き）。
  どちらも「残作業があるか」を確認する前に実行される。
- 問題: 全 250 updates 完了済み shard に segment を再投入すると、
  `train.py:1233-1237` の例外を経て except 側（`compare.py:596-635`）が
  summary/manifest/metadata を 'failed' で上書きし、completed 状態が恒久喪失する。
  README の pilot 手順（同一コマンド再投入）でも踏める。
- 修正方針: 上書き前に既存 manifest を読み、status=='completed' かつ resume 対象
  checkpoint が configured_updates に達している場合は
  **何も書き換えず「already completed」を正常終了（exit 0）で返す**。
  （chain の後続 segment が sd=ec!=0:all で削除されないよう、非0終了にしないこと。）
- 受け入れ条件: completed 済み run root に対して同一 segment コマンドを再実行しても
  manifest/summary が変化せず exit 0 となるテスト。

### A-3. [major] segment 早期失敗で source_sha256 が消え resume が恒久拒否される
- 場所: `src/sb_pomdp/compare.py:444-461` が source_sha256 を含まない dict で
  metadata を上書き → 成功時のみ `:510-517` で merge して `:529` で書き戻す。
  try 中の失敗（現実的には `train.py:116-117` の CUDA 不認識）で except 側
  （`:518-527`）が **fingerprint 欠落のまま** metadata を書く。
- 問題: 次の resume が previous_source=None を読み、`:512-516` の
  「source code differs」で以後全 segment が失敗し続ける（誤メッセージ付き）。
- 修正方針: metadata 上書き時に既存 metadata の source_sha256 を無条件で引き継ぐ
  （新規 root のときのみ後段で設定）。except 経路でも保持されることをテストで固定。
- 受け入れ条件: 「segment 1 成功 → segment 2 を人工的に device 失敗させる → segment 3」
  のシーケンスで segment 3 が正常 resume できること。

### A-4. [minor] GRU / tbptt_1 の 60 shards に resume 経路がない
- 場所: `src/sb_pomdp/compare.py:576-578`（baseline の segmented resume を明示拒否）、
  `src/sb_pomdp/train_baselines.py:1115-1126`（checkpoint に trainer_state なし）。
- 問題: 168h 単発ジョブが落ちるとゼロからやり直し。片腕だけ再実行コストが高い
  非対称な脆弱性。
- 修正方針（規模大・任意）: score trainer の checkpoint v4 と同等の
  trainer_state（env RNG・hidden/履歴・進行中 episode・全乱数状態）を
  train_baselines に実装し、`compare.py` の拒否を解除する。
  score 側の `train.py:644-705`（保存）/`:728-1133`（検証付き復元）を参照実装とする。
  実装しない場合でも、拒否メッセージに「再実行はゼロからになる」旨を明記すること。
- 受け入れ条件: score 側と同型の resume 等価性テスト
  （`src/tests/test_training_resume.py` 相当を baseline に追加）。

### A-5. [minor] fetch が partial shard を拒否し、README 記載の pilot 途中取得が不能
- 場所: `src/sb_pomdp/fetch_results.py:41` 付近（`_verify_staged_result` が
  status=='completed' 以外を拒否）。README:332 は pilot（partial）の取得を案内している。
- 修正方針: `--allow-partial`（または `--expected-status partial`）フラグを追加し、
  既定動作は現行のまま厳格に保つ。README の該当箇所と整合させる。

### A-6. [minor] baseline の summary status が無条件 'completed'
- 場所: `src/sb_pomdp/train_baselines.py:1142` 付近。
- 問題: score trainer の partial/completed 区別（README:311-315 の規約）と非対称。
  aggregate の「completed のみ集計」フィルタの意味が baseline 側で成立しない。
- 修正方針: score 側と同じ規則（全 configured updates + 最終評価完了時のみ
  'completed'）に合わせる。

### A-7. [minor] 集計・表示の整合
1. `comparison.csv` に paired 差の **分散（per-seed 差の std）** 列を追加する
   （現状 mean のみ: `compare.py:318-332`。README:352 の「必ず併記」と不整合）。
   paired 差の符号規約（score_transformer − gru）も列名かヘッダ文書に明記する。
2. `summary.csv` の `evaluation_return_mean`（実体は deterministic readout:
   `train.py:1742-1743, 1788-1790`）を `deterministic_return_mean` へ改名するか、
   README に「primary metric ではない」と明記する。
   ※ 列名変更は aggregate / 既存テストへ波及するため一括で行うこと。
3. `compare.py:309` 付近: 1 seed しか completed でない場合に std=0.0 と出力される。
   n<2 は空欄または NaN とし、集計行に n_seeds 列を追加する。
4. `compare.py:91` 付近の `_selection` が seeds を config と照合していない。
   methods/tasks と同様に検証を追加する。

### A-8. [minor] その他の小修正
1. `src/sb_pomdp/policies.py:439`: `ScoreBeliefActorCritic.sample_policy` の離散分岐が
   `generator` 引数を無視して global RNG から sample している
   （baseline 側 `baselines.py:308-315` は generator を尊重）。
   `torch.multinomial(probs, 1, generator=generator)` 方式へ揃える。
   ※ 現行結果への影響はない（eval は global RNG も同 seed で再現）が、
   将来の standalone 再評価で score 側だけ非再現になる罠。
2. `src/sb_pomdp/train.py:723` 付近: resume 時の非有限 belief 検出は fail-fast でよいが、
   エラーメッセージに復旧手順（どの checkpoint へ巻き戻すか）を含める。
3. `jobs/genkai_production.sh:66` 付近の flock による同一 shard 排他は
   Lustre の mount オプション（flock 対応）に依存する。lock 取得後に
   「PID/ホスト名を lock ファイルへ記録し、既存記録があれば警告」する
   best-effort 二重化を追加する。
4. `config/production.json:32`: MCC のみ `observation_noise_std=0.01`
   （他タスクは 0.05）。意図的なら README の該当表へ一行追記する。

---

## B. ユーザー判断待ち（Codex は変更しないこと — 科学設定の変更）

### F-1. [major] GAE が horizon 打ち切り（truncation）を真の終端として扱う
- 場所: `src/sb_pomdp/buffer.py:252-256`（nonterminal = 1 − done）、
  `train.py:231-232`（done = terminated OR truncated）、
  `train.py:236-239`（auto-reset が最終観測を破棄するため、現行データでは
  truncation 境界の V(s_T) bootstrap は不可能）。
- 事実関係: 両手法・両 mode で完全に共通（`train_baselines.py:873-874, 933-940`）
  なので **内部比較の公平性は保たれている**。ただし
  (a) 絶対 return / 価値学習は truncation-bootstrap 実装と比較不能、
  (b) GRU は hidden で経過時間を暗黙に数えて horizon 依存価値を fit できるが、
  score belief の条件（状態事後粒子）には時間チャネルがなく、
  この共有バイアスの負担が両表現で等しい保証はない、
  (c) README / docs/*.tex にこの規約が記載されていない。
- 選択肢: (i) 現規約を維持し README・tex に明記のみ（現 campaign と整合、推奨）、
  (ii) 最終観測の保持 + truncation 時 bootstrap を実装（Pardo et al. 方式。
  **全 80 runs の再実行が必要**、`test_buffer.py::test_done_transition_does_not_bootstrap`
  の更新も必要）。
- Codex への依頼範囲: ユーザーが (ii) を選ぶまでは **文書化のみ**
  （README「公平性と評価規約」節と tex の d_t 定義への追記）。

### F-2. [major] light_dark の報酬設計では情報収集行動が最適にならない
- 場所: `src/sb_pomdp/envs.py:784-791`。
  reward = −(0.5·‖x_t‖² + 0.5·‖u_t‖²)（PRE-transition state に課金）、goal bonus なし、
  ‖x'‖ ≤ 0.25 で終了（コスト打ち切りのみ）。
- 事実関係（検証エージェントの実シミュレーション済み）: 生成条件
  （N=5、light at x₀=5、初期 N(2·1, 0.5²I)、horizon 30）では
  dead-reckoning が平均 return −31.7、光側へ寄る情報収集方策が −77.0 で
  **迂回が厳密劣後**。力学が決定論的（x'=x+u）なので最適 belief は行動履歴の和で
  復元でき、このタスクは belief 品質をほぼ弁別しない。報酬定義は README/tex に未記載。
- 選択肢: (i) 現行のまま「解釈の限定事項」として文書化（現 campaign と整合）、
  (ii) goal 到達 bonus（例: +R·1{‖x'‖≤r}）や終了時残存不確実性ペナルティを導入して
  情報収集を最適化する再設計（**light_dark の 20 runs 再実行が必要**）。
- Codex への依頼範囲: ユーザー判断まで **README への報酬定義の明記のみ**。

### F-3. [critical/運用] full/score chain のポイント予算超過リスク（コード修正対象外）
- 根拠: `results/job-6439117/_job/shard003.log` 実測で update 1 が 87 分
  （full/score/MountainCar）。full mode の prefix 成長（horizon 999 で定常 ~8×）を
  考慮すると MountainCar full/score 1 shard ≈ 10,900 pt+、5 seeds で ~54,500 pt となり
  90,000 pt 予算（jobs/README.md:213-217）をほぼ確実に超過。
  segment が 168h を超えると `sd=ec!=0:all`（jobs/submit_production_chain.sh:101,121）で
  chain 残りが削除され shard は恒久 partial。
- これは投入済みジョブの運用判断（pjstat -E で実測確認 → 継続/縮小/中止）であり、
  コード修正では解決しない。関連する堅牢化は A-1〜A-3 が対応する。
  任意の追加堅牢化: genkai_production.sh に「PJM 上限前に update 境界で
  正常終了する watchdog」を追加する（README の短時間 DEBUG には既にある設計）。

### F-4. [major/運用] 現行ソースは Genkai 上で未検証のまま production 実行中
- repo 内の debug 成果物は全て旧 source（sha256 ab9d2f39…、旧 116k-param GRU）のもので、
  現行 tree（af5e38d9…）での matched-GRU diffusion / full replay 経路は実行実績ゼロ。
  `logs/genkai-debug/` は空。
- 対応は運用（短時間 DEBUG `jobs/genkai_debug_short.sh` の再実行）であり
  コード修正ではないが、Codex が A 群を修正した後の tree では
  **必ず DEBUG job を先に流してから** campaign を再開すること。

---

## 参考: 監査で健全と確認済みの領域（修正不要 — 変更しないこと)

ULA/score の符号と再parameterized replay、full/tbptt_1 の detach 意味論、
GAE 再帰・PPO ratio・diffusion chain log-prob・microbatch 勾配蓄積（有効 minibatch 保存）、
advantage 正規化の scope（rollout 全体で1回）、GRU との parameter 一致
（全8エントリ実測一致、最大差 0.116%）、head 共有、seed / budget / 評価プロトコルの対称性、
comparison.csv が final update 評価を使うこと（best checkpoint 非使用 = 選択バイアスなし）、
checkpoint v4 の RNG 完全保存による chain/一括実行の等価性。
これらの挙動を変える修正は本依頼の範囲外であり、意図せず変えた場合は regression とみなす。
