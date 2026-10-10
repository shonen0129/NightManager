# Issue #34: GitHub上の読み取り専用調査（2026-10-10）

対象: [Issue #34](https://github.com/shonen0129/NightManager/issues/34)。GitHub default branchの基準HEAD: `4c370755e2114c74a3599366e135d842eb78e1f5`。この記録は**公開可能なパス・実装構造の棚卸し**だけで、credential残存の検出結果ではない。credential値・URL query・secret-bearingデータは収集・掲載していない。

## 確認できたこと

| 管理ID | 調査起点（公開コード・文書） | 判定と次の所有者限定確認 |
| --- | --- | --- |
| NM34-GH-DB | `src/leadlag/execution/backtest.py` の `backtest_store.sqlite` 保存、`reports/full_backtest_20260815/report.md` の過去実行記録 | 旧DB生成経路は存在。旧版DB本体、run_info.config_json、WAL/SHM、snapshot、backup内の実資格該当性は**未確認** |
| NM34-GH-HTTP | `src/leadlag/broker/tachibana/api.py`、[credential境界ADR](../../docs/decisions/2026-10-08-credential-boundaries.md) | 旧HTTP例外やsummaryを保存していた可能性を調べる起点を確認。過去のraw logと提出先の現存状況は**未確認** |
| NM34-GH-SESSION | `src/leadlag/broker/tachibana/session_cache.py` は復号virtual URLとcounterの保存経路を持つ | session cacheの実所在・backupと提供元での旧session無効化は**未確認** |
| NM34-GH-CI | `reports/20260922_production_acceptance/prepare_ci_snapshot.py`、`reports/20260922_production_acceptance/fetch_ci_logs.py` のCI snapshot / log取得処理 | 過去CI artifactと提出物の調査起点。成果物の保存世代・共有範囲・資格情報の有無は**未確認** |
| NM34-GH-HISTORY | 2026-10-08 credential境界ADRは旧設定・Git history・clone/fork/CI artifact調査を要求 | 対象の複製・履歴の実調査は**未確認** |
| NM34-GH-EXTERNAL | Issue #34 が列挙する配布先・support提出先 | GitHub上の検索だけでは外部の受領先・保持状況は立証できず**未確認** |

GitHub code searchにより上記の**存在する保存経路・調査起点**を確認した。検索結果が存在しないことは「漏えいなし」の証明にならない。旧credentialの有効性確認、secret値探索、Git履歴の全量検査、CI archiveの全件精査、運用ホスト・backups・support提出先の実走査はしていない。

## PRの状態（調査時点）

- [PR #65](https://github.com/shonen0129/NightManager/pull/65): Open。保存失敗・認証・cacheログの境界補強。実credentialや旧DBの運用処理とは別。
- [PR #72](https://github.com/shonen0129/NightManager/pull/72): Open。この調査票と[運用手順](../../docs/operations/issue34-legacy-credential-remediation.md)を追加。PRがマージ可能と報告されてもCIテスト成功を意味しない。
- Issue #34: Open。履歴・データの処理やsession失効の完了証跡なしではcloseしない。

## 所有者による次のゲート

1. 非公開台帳にケースIDを採番し、各管理IDに対応する実データの範囲、保存期間、担当者、複製・外部受領先を確認。ここに具体的な秘密値や秘密を含むパスを転記しない。
2. 実資格該当性と運用影響を非公開で判断し、必要な失効・再発行・旧session無効化について操作ごとに**明示承認**を得る。
3. 証跡保持・隔離・削除を対象別に決定し、承認後の処理と検証を非公開台帳に記録。未調査のバックアップや受領先は未完として残す。
4. 公開issueに安全なケースID・カテゴリ別進捗のみ反映し、全ゲートの所有者限定証跡を確認してからcloseを検討。

## 検証・制限

GitHubのIssue、PR metadata、default branchの関連コード検索、既存ADR・作業規約を参照した読み取り専用確認。実行テスト、運用環境の走査、実API操作、資格情報の失効/再発行/旧session無効化、削除・隔離はしていない。
