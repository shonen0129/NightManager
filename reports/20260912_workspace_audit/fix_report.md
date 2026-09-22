# Workspace audit remediation

2026-09-13 に監査で再現した不具合を実装へ反映した。拒否・未約定・監査失敗・当日データ欠損は成功扱いせず、既存建玉の削減だけをリスク停止中に許可する。立花注文状態は `sOrderStatusCode` / `sOrderOrderSuryou` と約定数量で変換し、FILLED・PARTIALLY_FILLED を約定照会へ渡す（状態コードは[公式仕様](https://www.e-shiten.jp/e_api/mfds_json_api_ref_text.html)に準拠）。

設定は全入口で `__base__` を解決し、VaR cache は有効設定・上書き値・`df_exec`・モデルコード・gap bundle の fingerprint をキーに含める。9:10 現在値は始値 cache と分離し、日付一致を必須にした。米国リターンは日米共通日で先に切らず、休日後の最新米国セッションを次の日本営業日に対応付ける。相関・common-input cache は入力値、target、9:10 価格を含む fingerprint を使い、GapStore の μ・Ω・metadata は同一 SQLite transaction で読み書きする。

損益は前日からの持越し在庫を台帳化して決済 slippage を計上し、最大 DD は初期 wealth=1 を含める。バックテストと registry の主指標は flat fallback 日を含む全評価日を使い、主指標の Sharpe・AR・risk は日次へ統一した（月次は明示指定時だけ）。DSR は年率 Sharpe を日次へ変換してから計算する。標準 V2 backtest は有効な overlay を自動ロードし、学習期間・data/config hash・code revision のない既存 overlay artifact は安全のため拒否する。A7 は V1 の歴史診断であることを明示し、明示フラグなしの実行と本番証拠化を停止した。

regression は `tests/regression/baselines/df_exec_20260814.csv.gz` と 2026-08-14 行列を固定 bundle として参照する。検証結果は修正対象テスト 110 件、全体 `tests/` 583 件を実行し、全体では既存期待値の更新が必要な 2 件だけを検出した後に更新して PASS。`compileall`、変更対象ファイルの Ruff、`src/leadlag` の mypy、import-linter も PASS。`src/research` を含むリポジトリ全体の Ruff には今回の修正範囲外の既存指摘が残るため、全体 Ruff を PASS とは扱っていない。

既存の `models/ml_order_overlay/phase2_8` は学習期間・データ hash・設定 hash の provenance がない旧 artifact である。実行時に安全側へ拒否するよう変更したため、本番 overlay を再開する前に `tools/production/train_ml_order_overlay.py` で provenance 付き artifact を再学習する必要がある。旧 artifact を未知の情報で補完して検証済み扱いにはしていない。
