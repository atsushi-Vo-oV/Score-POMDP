# 凍結 light 表現上で学習した拡散 BC actor

提供された `data/followup_bc_disk_tag_score_transformer_{10,11}_L5_std0.05.pt` の内容を変更せずコピーしました。各64,278 bytesです。source は11月10日の `followup_bc.py`。5段、min_std=0.05、condition32次元、hidden96×96です。

保存形式は `state_dict`, `meta`, `xm`, `xs`。**state_dict は標準化を第1線形層へ折り込み済み**です。xm/xs を再適用しないでください。`initialization.load_bc_actor` がこの履歴形式と、明示的な `meta.normalization_folded=false` の未折り込み形式を扱います。
