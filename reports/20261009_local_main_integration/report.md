# ローカルmain・未コミット変更の統合（2026-10-09）

## 対象

GitHub main `4e2b44884956dc9ec760a10314f560b6fb0814be` に、ローカルmainの
`9b950b74`（ML overlay研究専用化）と `c3a86c95`（adaptive overnight研究）を取り込んだ。
未コミットの個別株basket/sector alpha研究について、設定・再現スクリプト・レポート・不採用索引を追加した。
監査issue対応表のユーザーによる旧進捗節の削除を保持し、先頭の誤字だけを修正した。

ML overlayは本番設定で無効・artifact rootは空とし、別の研究設定でcandidateを選ぶ。
既存のV2・RuleD・risk stop・side leverage・監査・fallbackの契約は保持する。

## 統合時の修正

- 技術仕様書のML有効表示を継承解決した本番設定へ合わせた。既存の設定/文書照合テストは変更していない。
- 研究shadowのcandidate設定読込をshadow失敗処理の内側へ移した。通常のV2判断を止めず、shadow-onlyは失敗として終了する。broker境界をモックした回帰で双方を確認した。
- Overnight研究の銘柄別内訳は、正規 `simulate_daily_pnl` のNAV・終値在庫を読み、価格変動後の在庫と受入日へ損益/費用を帰属する。旧方式の別在庫状態機械は残していない。
- 研究replayの保有日数を受入区間へ合わせ、撤去済み価格抽出APIを `_compute_price_intervals` へ更新した。寄付gapと寄付→09:10を複利で結合し、当該entry→closeのtargetと対応させた。
- 長期研究入口のimportを正規research packageへ合わせた。4研究入口のimportを確認した。
- 個別株研究のDSR入力はdaily-v1の単位を明示し、探索履歴はunknownとした。未確認の履歴から確認済みDSRを作らない。

## 検証

Python 3.12のロック済み一時環境で **1,264 passed / 0 failed / 0 skipped**。
baseline 1件を直列、残り1,263件を4 workersで実行した。外側期限はそれぞれ300秒/1,800秒、終了猶予10秒。
対象の45回帰、価格変動/週末費用/受入日/終端決済の手計算回帰も成功した。既存warningは2件。

compileall、本番/研究Ruff、mypy（147 source files）、7 import契約、operational import境界2入口が成功した。
維持文書298参照と現行249 path/symbol参照、追加研究文書のリンクを検証した。
lock check、clean wheelのsource manifest/研究除外（152 files）、installed wheelのCLI・ML artifact roundtrip・ADR bundle・共有BLPX smokeが成功した。
GitHub必須CIは追加PRのActions記録を参照する。

## 保存・評価の範囲

元の未コミット内容はローカルのバックアップarchiveにも保持する。
市場cache・運用入力・ignored registryは今回のGit統合で書き換えない。
既存の研究CSV/JSONの数値は当時の結果を保持し、レポート先頭に現行会計との区別を追記した。
実データによる研究の全再評価・OOS採用判断・本番発注は行っていない。

元の研究記録:

- [個別株直接写像](../20261007_nightmanager_direct_stock_alpha/report.md)
- [ETFの株basket代替](../20261007_nightmanager_etf_stock_proxy/report.md)
- [Overnight初回](../20261008_adaptive_overnight_inventory/report.md)
- [Overnight長期](../20261009_adaptive_overnight_inventory_v2_long/report.md)
