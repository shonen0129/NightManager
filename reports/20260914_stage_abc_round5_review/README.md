# 段階A〜C 第5回・再発確認レビュー

2026-09-14の未commit作業ツリーを、元のF01〜F22と後続R/S/T/Uの指摘まで戻って確認しました。**判定はBLOCK、残存P2は4件**です。前回U02〜U04は再現解消、U01は通常decideで拒否できる一方、互換APIの利用側に問題が残ります。

| ID | 残存問題 |
|---|---|
| V01 | 不整合・来歴欠落の`.npy` bundleを互換loader経由で利用できる |
| V02 | 部分失効がpendingになり、訂正/取消要求の失敗を原注文の終了と扱う |
| V03 | common-input cacheが列名の訂正を識別せず旧計算を返す |
| V04 | VaRのcache key計算後にgapが更新されると、別版の系列を旧keyへ保存する |

[詳細レビュー](./review.md)に、発生条件、コード位置、再現結果、具体修正、回帰テストの受入条件、F01〜F22の確認表を記載しています。[機械可読一覧](./review_summary.json)も参照できます。

全`tests/`は[620 passed / 17 warnings](./pytest_full.log)。Ruff・mypy・compileall・import-linterも記載範囲で成功しました。検証済みの修正と、実本番ML artifactの再生成・本番/BT weights照合という運用未了を区別しています。

- [主な再現結果](./reproductions.json) / [再現コード](./probe_review.py)
- [設定・WAL・A7・pytest終了後の分離確認](./contracts.json) / [再現コード](./probe_contracts.py)
- [列名変更による実数値の差](./schema_cache.json) / [再現コード](./probe_schema_cache.py)
- [VaRのgap版不一致](./var_gap_version.json) / [再現コード](./probe_var_gap_version.py)
- [開始時hash](./source_snapshot.json) / [終了時manifest](./review_manifest.json) / [証跡検証](./validation.json)

本番ソース・設定・テストは変更していません。broker応答は偽物、一時DB・一時artifact・固定fixtureを使い、本番の発注・決済・口座照会はしていません。再現コマンドと検証範囲の限界は詳細レビュー末尾にあります。
