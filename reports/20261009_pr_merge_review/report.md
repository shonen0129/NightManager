# PR #53〜#59の統合レビュー（2026-10-09）

対象は認証情報境界、在庫会計、study/DSR契約、BLPX共有計算、日次/close/前処理/VaR分割、未使用層整理、運用文書契約の7件。依存順は #53 → #54 → #55 → #56 → #57 → #58 → #59。各PRは統合先を取り込んで必須 `quality-and-tests` の成功後にマージする。#59のbaseは#54の統合後にmainへ変更した。

## 修正と確認

- #53: RSAの想定例外/診断の非露出、実adapterからlog/注文summaryまでの合成secret回帰を確認した。
- #54: 受入日のgap/朝損益、前日NAV基準、値洗いしたcarry、終端決済と区間間の在庫継続を確認した。
- #55: 所有環境のregistry120件を点検し、34件を原本保持の追記訂正で未確認化した。[個別点検の集計](../20261009_pr55_registry_review/report.md)を参照する。架空trialを作らず、原本と過去成績を保持した。
- #56: finite/nonfinite/PSD/window/非対称計算の変更前fixtureとの一致、prior hookの継続と未来行の摂動不変性を確認した。
- #57: CIのtuple/list型不一致を再現して修正した。#54とのVaR競合では関数分割後も終端holdings/cash/date/target weightsの引継ぎを保持した。[修正記録](../20261009_pr57_merge_review/report.md)を参照する。
- #58: provider/ML wrapperの使用元撤去、convex optimizerの研究層移動、phase deadlineの新配置を確認した。文書の競合では#54/#56の完了した契約と残件の区別を保持した。
- #59: 継承解決した本番設定・実際の監査/close挙動を文書の契約表と照合した。文書検査と運用回帰は統合後も成功した。

## 統合コードの検証

Python 3.12のロック済み一時環境で **1,253 passed / 0 failed / 0 skipped**（baseline 1件 + 残り1,252件、4 workers）。全体期限はbaseline 300秒、残り1,800秒、終了猶予10秒。既存warningはDataFrame断片化と定数入力の2件。全体テストを通した統合treeと#59のコード/設定/test/CI treeは一致した。

compileall、本番/研究Ruff、Mypy（147 source files）、7 import契約、operational import境界2入口、ロック検査、現行文書246 path/symbol参照が成功した。新ADR/修正記録を含むリンク147件も成功した。clean wheelのsource manifest/研究除外、installed wheelのCLI・ML artifact roundtrip inference・ADR bundle・共有BLPX smokeも成功した。

コードtreeの識別値は [verification.json](verification.json)。GitHub側の必須チェックは各PRのActions記録を参照する。

## 範囲と残件

この結果は7件の差分と統合のレビューであり、本番の実発注受入や戦略採用判断を保証しない。269日版の完全な再評価、資格の失効/再発行等の所有者限定対応、過去探索履歴の完全性、live risk/市場capture/forward観測の残件を完了扱いしない。#33/#34/#41/#47を自動closeする表現は追加しない。

元のworkspaceのmain・未コミット文書・研究ファイルは保持し、一時checkoutでレビューと修正を実施した。研究registryのみ上記のappend-only訂正を行った。
