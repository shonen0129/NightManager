# PR #57 統合時修正（2026-10-09）

CIのMypy失敗を再現した。close計画の `tuple[OrderRequest, ...]` を既存の `split_large_orders` の `list[OrderRequest]` 契約へ変換する。計画の不変性と注文の順序は保持する。修正後Mypyは152ファイルで成功し、close・分割注文・quote preflight・実口座risk・前処理の関連46件が成功した。

PR #54の在庫会計とVaR関数分割の競合を解消した。`_run_var_history_backtest` では各artifact区間を `open_inventory` で終了し、終端holdings/cash/mark date/target weightsを次区間へ渡す。監査・非有限値・fallbackの拒否、deadlineとsnapshot所有権を保持する。実VaR入口の在庫継続を含む関連23件が成功した。

全体テストとCIのその他の必須チェックは更新後のPRで実行する。Issue #45の範囲の統合であり、Phase全体の完了や実発注の受入とは扱わない。
