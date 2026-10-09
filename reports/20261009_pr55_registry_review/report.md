# PR #55 統合前の実registry点検（2026-10-09）

取得環境で未実施だった `var/experiments/registry.jsonl` の個別点検を所有環境で実施した。元120件中、保存DSRがある34件は探索履歴の完全性を確認できず、そのうち22件は数値入力も欠落または不正だった。

`research.study_history.review_registry` の標準の追記訂正を34件適用した。元の全バイトが変更後ファイルのprefixとして残ることと、再点検の訂正候補が0件になることを確認した。過去DSRを全件数学的に誤りとは扱わず、該当値を未確認化した。原本backupは所有環境の一時ディレクトリに保存し、raw registryはGitHubへ掲載しない。

集計・前後hash・冪等性は [verification.json](verification.json)。事前登録済みの完全な探索履歴を遡及的に作ったことにはならず、将来の採用判断にはPRで追加したstudy契約を使用する。Issue #41全体を自動closeせず、コードの統合と研究採否を区別する。

関連回帰57件成功。最新mainとの統合・全体検証はこのPRのGitHub CIで確認する。
