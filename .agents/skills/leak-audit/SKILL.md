---
name: leak-audit
description: 時系列計算・PIT・監査・フォールバックの変更検証、リーク疑い、監査失敗の調査に使う。
---

# リーク・制約監査

不変条件と本番の失敗時方針はルートの `AGENTS.md` を参照する。以下のパスはリポジトリルート基準。

## 実装と監査の対応

| 対象 | 入口・確認事項 |
|---|---|
| 汎用監査 | `src/leadlag/compliance/auditor.py::ComplianceAuditor.run_audit`。model の audit context と results の必須キーを確認する |
| V2 日付/PIT | `src/leadlag/compliance/v2_auditor.py::run_leakage_audit`。実際に用いた signal date / trade date / PIT 履歴日付を渡す |
| V2 数値 | 同ファイルの `run_numerical_audit`。有限性、net/gross、共分散の対称性・半正定値性（PSD）を確認する |
| 分布取得 | `src/leadlag/models/v2/fallback_policy.py` / `distribution_source.py`。cache / on-demand / flat の条件を追う |
| 結果処理 | `src/leadlag/models/v2/audit_comparator.py` と呼び出し元。監査失敗後のウェイトと発注可否を追う |

監査に存在しない関数名や検査項目を実施済みと書かない。日付の前後関係だけでは統計窓・残余化・データ公表時刻の非リークを証明できない。`pit_history_trade_dates=None` のようにチェックが実質省略される入力も確認する。

## 検証観点

対象の計算・失敗経路に当てはまる観点を選ぶ。参照だけの変更で全モデルの監査へ広げない。

1. 変更対象の入力がいつ確定するか、`df_exec` の当日ターゲットとどう分離されるかを整理する。
2. 相関・beta・PIT 閾値・macro surprise・copula・学習済み overlay の実際のデータ窓を追跡する。2010–2014 の事前分布と評価期間の混入を調べる。
3. 予測時点で未知の当日ターゲット・未来行を摂動しても当日出力が変わらないことを確認する。当日既知の US リターンや gap はこの摂動対象から区別する。
4. cache と on-demand の日付・モデル設定・銘柄順・入力期間を照合する。前日 cache の流用を検出し、on-demand の成功・無効・入力不足・例外を別々に検証する。
5. 数値監査失敗時のフラット化と、リーク監査失敗時の下流処理を別々に追う。現行 `_run_safety_audits` は数値 FAILED + `fallback_on_audit_failure=true` でフラット化する。リーク FAILED も同じ条件で自動フラット化すると仮定しない。
6. フラットになっても scores / 共分散の不正は残り得るため、再監査結果・alerts・実際の発注停止を確認する。`FLAT` は非稼働の状態であり、通常計算の全監査 PASS として数えない。
7. モデルウェイトと実効レバレッジの制約、gross − costs = net を照合する。汎用監査の許容誤差や既定値を本番リスク上限に置き換えない。

## 実行・報告

関連テストは `tests/integration/test_leakage_audit.py`、`tests/integration/test_production_residual_blpx.py` と変更箇所の unit test。期限と全体回帰は AGENTS.md に従う。

出力は実際の監査 artifacts と、対象経路・入力日付・失敗条件・再現方法を示す。汎用監査と V2 監査の対応、未確認範囲を明記する。レビューのみの依頼では修正案を提示し、修正も依頼されていれば必要な回帰を伴って直す。監査スキップや期待値の緩和で PASS にしない。
