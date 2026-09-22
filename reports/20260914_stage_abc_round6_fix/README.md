# 段階A〜C 第6回指摘の修正

前回レビューのW01/W02を修正しました。W01はgap bundleとV2利用側で共通の来歴検証を使い、日付aliasの矛盾・非scalar日付・非有限horizonを拒否します。W02はVaRのgap snapshot取得へ総期限を適用し、SQLiteロック待ちを期限内に停止します。BTがtimeoutした場合も、継続中のworkerがsnapshotを使い終えるまで所有者を保持し、終了後に回収します。

[修正内容と検証結果](./fix_report.md)に対象ファイル、受入条件、テスト結果、未完了の運用作業を記録します。
