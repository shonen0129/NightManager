# Issue #34: 過去の資格情報記録・旧sessionの所有者限定対応手順

対象: [Issue #34](https://github.com/shonen0129/NightManager/issues/34)。関連する実装境界は [2026-10-08 ADR](../decisions/2026-10-08-credential-boundaries.md) と [PR #65](https://github.com/shonen0129/NightManager/pull/65) を参照する。

> **公開可能なのはこの手順だけ。** 以下で作る実際の調査票・証跡・対象パス一覧は、GitHub issue、PR、公開repository、通常のCI artifactに保存しない。この文書の追加だけでは調査、失効、再発行、旧session無効化、削除の実施やIssue完了を意味しない。

## 1. 操作境界

- 調査は所有者の管理下にある端末・運用ストレージ・バックアップ管理画面・配布先・support提出先で行う。権限のない領域は「未確認」と記録する。
- 実credentialの値、token、復号URL、session、秘密を含むフルパス/URL、DB rowやraw tracebackを、公開記録・チャット・チケット・スクリーンショットに転載しない。資格の有効性をAPI/login/実注文で試さない。
- **失効・再発行、session無効化、データ削除・隔離、backup変更、共有先への連絡、履歴書換えはそれぞれの対象と影響を整理し、所有者の明示承認を得てから実施する。** 一括事前承認があると解釈しない。
- 原本を安易に編集しない。SQLiteのrow UPDATE/DELETEだけではWAL、freelist、snapshot、backupの残留を除去できない。証拠保全が必要な場合は先にアクセス制限・保存期限を決定する。
- 既存のsecret管理、incident対応、保持規程を優先する。独自の資格情報保管基盤やGit履歴の無断書換えを導入しない。

## 2. 所有者限定の調査台帳（非公開に作成）

既存のアクセス制御された運用記録へ、**ケースID**（例: `NM-SEC-34-YYYYMMDD-01`）を採番する。以下は項目の定義だけで、実データを書き込む公開テンプレートではない。

| 項目 | 記録内容（秘密値なし） |
| --- | --- |
| 対象ID / 種別 | 非秘密の管理用ID、DB / log / backup / session cache / clone / CI artifact / 配布先 / support提出先 |
| 管理者 / 権限 | 確認した担当者・対象へのアクセス権の有無 |
| 確認時刻 / 観測期間 | 日時・timezone、調査対象の期間、対象の版・保管世代 |
| 確認方法 / 結果 | 設定・metadata・schema・保管一覧等の**非秘密の**根拠、`affected` / `not_found` / `unknown` |
| 複製とアクセス | backup、WAL/SHM、archive、clone/fork、CI artifact、転送先、support添付の有無・閲覧可能範囲 |
| credential / session | 実資格該当の可能性、失効・再発行・旧session無効化の状態（`not_started` / `pending_approval` / `completed` / `not_applicable` / `unknown`） |
| 処理判断 | `retain_restricted` / `quarantine` / `recreate_safe_copy` / `delete_after_approval` / `pending`、保持根拠・期限 |
| 実行承認と証跡 | 操作ごとの承認者・承認時刻・対象ID・影響、実施者・実施時刻・確認方法・非秘密の証跡ID |
| 未了・再確認 | 未確認先、再配布・復元による再露出の可能性、次回確認担当・期限 |

調査で秘密を目視する必要がある場合も値を転記せず、閲覧履歴の残らない承認済み経路だけで取り扱う。出力コマンドのstdout、shell history、CI logへの露出を避け、ファイル名の一括公開検索や生の `grep` 結果貼付を行わない。`not_found` は確認したスコープ内のみの結論とし、全端末・全期間の安全宣言に使わない。

## 3. 調査順序と判定

1. **棚卸し**: 既存台帳・保管ポリシーから保存先、管理者、保持期間、共有・転送経路を列挙する。対象不明の経路は `unknown` とする。
2. **旧backtest保存**: `backtest_store.sqlite` の `run_info.config_json` と、付随するWAL/SHM、snapshot、export、backupを所有者限定で照合する。configの全文やrow内容は公開出力しない。
3. **旧HTTP失敗**: Tachibanaのlogin / health / decision / capture / orderに関するlog、例外・traceback、summary、support添付・配布済み成果物とその複製を確認する。tokenやURL queryを含み得る。
4. **設定・履歴・配布**: 旧設定、環境ファイル、Git history、clone/fork、CI artifact、archive、受領先を範囲に入れる。現行ファイルの修正は過去版の無害化ではない。
5. **session**: 復号virtual URLを含み得るcache、backup、旧端末の残存有無と失効手段を確認する。cacheファイル削除とサーバー側session無効化を同一視しない。
6. **相関づけ**: 同一credential/sessionの複製・提出先を管理用IDで紐づけ、承認が必要な操作と業務影響（再ログイン、予定実行停止、バックアップ復元時の再露出等）を整理する。

調査できなかった場所は、存在しないとは推定せず未確認として残す。実資格だったかの判断・有効性・個別アカウント情報は所有者限定記録のみで扱う。

## 4. 承認後の処理と検証

- **資格情報**: 実資格に該当する場合は提供元の正規手順で失効/再発行し、影響する設定・secret storeを更新する。関連する実行停止・再開計画を先に確認する。
- **session**: 提供元の正規手順で旧sessionを無効化できたことを確認し、必要なcache更新を行う。「ファイルを消した」だけで旧sessionの無効化完了と記録しない。提供元で失効状態が確認できない場合は `unknown` とする。
- **記録**: 保持が必要な監査証跡をアクセス制限下に保全し、共有用には非秘密の再生成物だけを使う。削除/隔離が必要な原本・複製・backup・提出物は個別承認、保持義務と復元リスクを確認して処理する。削除不能な受領先・snapshotは残存リスクとして追跡する。
- **確認**: 操作の完了と対象範囲を非秘密の証跡IDで相互確認し、未処理の複製・将来のbackup復元・配布先を再点検する。実資格による公開監査やライブAPI試験は行わない。

## 5. 公開issueへ記録する内容と終了判定

公開issue/PRには **ケースID、対象区分ごとの `not_started` / `in_progress` / `pending_approval` / `completed` / `unknown`、残件の件数・担当境界（非秘密）** のみ記載し、所有者限定台帳へのリンク・実ファイルパス・secret・具体的な事故対象値は載せない。

Issue #34を完了にするには、(a) DB/log/backup/配布先/support提出先のスコープと未確認領域、(b) 実資格該当時の失効・再発行、(c) 旧session無効化、(d) 監査証跡の保持と対象ごとの削除/隔離判断、(e) 各承認と完了証跡、が所有者限定記録で確認済みであることを要する。該当なしと判断した項目も根拠スコープを残す。公開文書やコード修正だけでチェックを完了にしない。
