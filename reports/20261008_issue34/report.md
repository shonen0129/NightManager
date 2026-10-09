# Issue #34: 認証情報の保存・例外境界とRSA復号fallback

対象: [Issue #34](https://github.com/shonen0129/NightManager/issues/34)。開始時HEAD `f08d1fd9117eb12594f7fde4ee78f795d9ee1000`。

## 実装

現行mainにあるbacktest設定のallowlist/hash、Tachibana HTTP/parse例外変換を維持し、CLI→全成果物/SQLite/cacheとbroker login/order/health→log/注文summaryの回帰を追加した。

RSA復号はSHA-256→SHA-1→v1.5を維持し、想定例外へ狭めた。固定messageのValueErrorと、安全なalgorithm/失敗段階の診断を正本のlogin diagnosticsへ記録する。許可された応答項目に入る任意値もそのまま保存せず、既知の機能ID・bounded code・flagだけを残す。

判断・運用手順: [ADR](../../docs/decisions/2026-10-08-credential-boundaries.md)。

## 検証

Python 3.12.11、`uv sync --locked --extra dev --extra ml-overlay` の隔離環境。各長時間プロセスに既存phase deadlineと終了猶予10秒を設定した。実資格・実API・実注文は使用せず、RSA keyと全secretは合成値。

- 対象回帰: 114件成功（26.42秒）。
- 固定V2基準回帰: 1件成功（1.66秒、停止期限300秒）。
- 全テスト: 残り1180件成功（126.04秒、4 worker、停止期限1800秒）。固定基準回帰と合わせて1181件を網羅し、skip/失敗なし。既存のDataFrame fragmentationとconstant correlationに関するwarningが2件あった。
- lockfileのoffline check、必須compileall、本番/研究Ruff、mypy（152 source files）、7 import契約、scheduled producer import境界: 全て成功。
- CI対象文書と新規ADR/本記録の参照検証: 成功。追加文書作成中の初回検証で未作成の本記録へのリンクを検出し、作成後に再検証した。
- clean wheel build、source manifest/研究package除外の検証、installed wheelとML推論smoke: 成功。ビルド時の既存setuptools license形式の非推奨warningはこの変更の対象外。

検証logとJUnit XMLはこの作業の隔離workspaceに保管した。すべて停止期限内に終了し、外側のdeadlineは全テストで `completed` / return code 0 / process-group cleanup `already_exited` を記録した。

## 残件

資格の実在性・現在の状態についての確認内容は公開記録へ含めず、失効・再発行と旧session無効化の証跡は所有者限定の運用記録で管理する。fresh checkoutに運用環境の既存DB・log・backupがなく、それらの残存状況・support提出先は未確認。値の再掲載・有効性試験・履歴書換え・本番反映は実施していない。このPRはコードの保存・例外・診断境界を修正するものであり、Issue全体の完了には運用側の証跡も必要となる。
