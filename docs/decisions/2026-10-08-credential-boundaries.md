# 2026-10-08: artifact・Tachibana例外とRSA復号診断の秘密情報境界

Status: accepted（実装境界のみ。資格の状態・運用対応は所有者限定の記録で管理する）

対象: [Issue #34](https://github.com/shonen0129/NightManager/issues/34)。
開始時HEAD: `f08d1fd9117eb12594f7fde4ee78f795d9ee1000`。

## 保存と例外の正本

- backtest設定は既存の `src/leadlag/data/backtest_store.py::_safe_config` を正本とする。broker設定全体と未知のtop-level項目を除外し、strategy/risk/V2・非秘密の実行設定を保存する。hashは秘密を除いたcanonical JSONのSHA-256であり、broker資格の変更だけでは変わらず、保存対象の設定変更では変わる。
- 既存の `TachibanaClient._get_response` / `_parse_response` を全HTTP経路の例外境界として維持する。URL query、request/response、server error textを転記せず、固定messageの `TachibanaApiError` を送出し、unsafe causeを通常のtracebackへ出さない。
- login diagnosticsは項目名に加えて値も制限する。機能IDは `CLMAuthLoginRequest` / `CLMAuthLoginAck`、結果codeは1〜5桁のASCII数字、開示flagは0/1のみとし、任意の文字列・構造や長い数値は値を保存しない。欠落/null/空文字/型の区別は維持する。raw flagをmissing-URL例外へ補間しない。

## RSA復号の契約

`_decrypt_virtual_url` は OAEP SHA-256 → OAEP SHA-1 → PKCS1 v1.5 の順序を維持する。想定する復号例外は `ValueError` / `TypeError`、復号後のUTF-8不正は `UnicodeDecodeError` として分類する。想定外の実装エラーを「次方式へfallback」で隠さない。

各URLの既存 `last_login_diagnostics.virtual_urls` に、`decrypt_algorithm`（成功方式）、`decrypt_attempts`（固定algorithm名と `succeeded` / `decrypt_failed` / `invalid_utf8`）、`decrypt_failure_stage` を追加する。正常fallbackでwarningを出さない。診断の別正本や互換wrapperは作らない。

読込失敗は `key_read`、不正/公開鍵は `key_import`、base64入力失敗は `ciphertext_decode`、全方式失敗は `all_algorithms` に分類する。いずれも固定messageの `ValueError` とし、unsafe causeを抑止する。全方式失敗時は既存の明示message `Failed to decrypt virtual URL using all known RSA padding/hash algorithms.` を維持する。private key/path、暗号文、復号URL、認証payloadは例外・診断に入れない。

## 過去artifact・logの影響範囲と取扱い

この修正は過去に書かれたファイルやGit履歴を消去しない。影響範囲は以下を起点に、運用環境側で原本・複製・backup・support提出物を調べる。

| 保存経路 | 対象と確認方法 | 取扱い |
| --- | --- | --- |
| 旧backtest保存 | 結果directoryの `backtest_store.sqlite` / `run_info.config_json`、SQLite WAL/SHM、複製・backup。allowlist導入前のbroker設定を含むrow | 非秘密設定だけを新DBへexportし、秘密を含む原本を共有しない。rowをUPDATEするだけでは旧page/WAL/backupからの除去を保証しない |
| 旧broker HTTP失敗 | capture/decision/health/orderのlog・summary、raw request URLを含む例外/traceback、診断結果、提出済み添付 | credential-bearing queryを含むものを隔離し、共有する必要があれば安全な固定分類だけを再作成する |
| 旧設定とGit履歴 | `.env.example` の過去値、資格を含む設定、patch・archive・clone・fork・CI artifact | 現行ファイルの空欄化だけでは過去値を無効化できない。履歴整理は失効後、所有者が影響範囲を確認した上で別作業として実施する |
| session cache | 復号virtual URLを保存する既存session cacheとbackup | artifactとして配布しない。失効時は旧sessionも無効化し、運用側でcacheを更新する |

根拠は [9/29 ADR](2026-09-29-frozen-0910-account-risk-forward-eval.md)、[10/6 ADR](2026-10-06-audit-boundaries.md)、Issue #34。今回のfresh checkoutには運用側の結果DB・capture原本がなく、過去の全保存先・提出先の残存状況は確認できない。実資格の値を探索・転載したり、その有効性をAPIへ問い合わせたりしていない。

資格の実在性・現在の状態についての確認内容は公開記録へ含めない。所有者は証券会社側の失効・再発行、旧sessionの無効化と必要な設定更新の状況を所有者限定の運用記録で管理する。完了証跡は値を記録せず、失効日時・再発行完了・旧session無効化の状況で確認する。コードの回帰成功だけでIssue全体を完了扱いしない。

## 検証

合成key/ciphertextだけで3方式の成功、途中fallback、全方式失敗、UTF-8不正、鍵読込/import・base64失敗、診断の非露出を検証する。CLIの実設定読込→成果物/SQLite/cache、brokerの実login/order/health失敗→log/注文summaryを回帰で通す。数理モデル・prior・監査・cache fallback・durable reconciliationの契約は変更しない。本番反映・実発注・実資格操作は実行していない。

検証結果: [対応記録](../../reports/20261008_issue34/report.md)。
