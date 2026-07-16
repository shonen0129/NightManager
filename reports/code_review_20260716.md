# 批判的レビュー報告書 — yfinanceタイムアウト重複解消 + v2_auditor実検証ロジック追加

本レビューは2026-07-16に行われたリファクタリング（yfinanceタイムアウトラッパー3箇所共通化 + `run_leakage_audit`実検証ロジック追加）の最終監査である。

## 1. 結論

**PASS** — 確認範囲では重大な問題がない。

## 2. レビュー対象と調査範囲

### 変更対象ファイル
- `src/leadlag/utils/__init__.py` — 新規（空パッケージ）
- `src/leadlag/utils/threading.py` — 新規（`run_with_timeout` 汎用ユーティリティ）
- `src/leadlag/data/fetcher.py` — daemon threadブロックを `run_with_timeout` に置換
- `src/leadlag/core/macro.py` — daemon threadブロックを `run_with_timeout` に置換
- `src/leadlag/broker/tachibana/client.py` — daemon threadブロックを `run_with_timeout` に置換
- `src/leadlag/compliance/v2_auditor.py` — `run_leakage_audit` に実検証ロジック追加
- `src/leadlag/models/production_v2.py` — `load_pit_ir_history` 戻り値変更、呼び出し側更新
- `tools/validation/run_daily_residual_blpx_shadow.py` — 呼び出し側更新
- `tests/unit/test_run_with_timeout.py` — 新規（3テスト）
- `tests/integration/test_leakage_audit.py` — テスト更新
- `tests/integration/test_production_v2.py` — テスト更新

### 確認した実行経路
- yfinance download経路: `fetcher.py` → `_yf_download_with_timeout` → `run_with_timeout`
- macro download経路: `macro.py` → `download_macro_prices` → `run_with_timeout`
- broker fallback経路: `client.py` → `fetch_us_etf_returns` → `run_with_timeout`
- 本番パイプライン: `production_v2.py` → `generate_v2_production_portfolio` → `run_leakage_audit`
- シャドーツール: `run_daily_residual_blpx_shadow.py` → `load_pit_ir_history`
- セルフテスト: `run_daily_production_v2.py` → `run_leakage_audit`

### 実行したコマンドと結果
- `python3 _check_syntax.py` — 12/12 OK
- `python3 -m py_compile` (変更ファイル9個) — 全てOK
- 並列テスト（7プロセス） — 406件全て合格（約9分）

### 確認できなかった範囲
- ruff/pyflakes未インストールのため静的解析未実行（py_compileで代替）
- yfinance実API呼び出しの統合テスト（ネットワーク依存）

## 3. 問題一覧

#### [P3-001] tachibana/client.py タイムアウト時のログメッセージ退化
- ファイルと行番号: `src/leadlag/broker/tachibana/client.py:244-258`
- 発生条件: yfinance Ticker.history() がタイムアウトした場合
- 影響: 元のコードは `logger.error("yfinance history() timed out for %s after %ds", ticker, _YF_TIMEOUT)` という専用メッセージを出力していた。新コードでは `TimeoutError` が外側の `except Exception as e` でキャッチされ、`logger.error("yfinance US return fetch failed for %s: %s", ticker, e)` となる。タイムアウト固有のメッセージが失われるが、機能的挙動（ticker を failed に追加してループ継続）は同一。
- 根拠: 元コードの `if th.is_alive(): logger.error(...); failed.append(ticker); continue` → 新コードの `except Exception as e: logger.error(...); failed.append(ticker)`
- なぜテストで防げないか: `test_fetch_us_etf_returns_fallback` は `yfinance.Ticker` をモックしており、タイムアウトパスをテストしていない
- 修正方針: 対応不要（ログメッセージの差異は運用上の実害がない）。必要なら `except TimeoutError as e:` ブロックを追加して専用メッセージを復元可能。
- 確信度: 高

#### [P3-002] macro.py タイムアウトエラーメッセージのフォーマット変化
- ファイルと行番号: `src/leadlag/core/macro.py:179-183` (経由 `src/leadlag/utils/threading.py:48`)
- 発生条件: yfinance download がタイムアウトした場合
- 影響: 元のメッセージ `yfinance download did not complete within 30.0s (tickers=[...], start=..., end=...)` が `yf.download(tickers=[...], start=..., end=...) exceeded 30.0s timeout` に変化。情報量は同等。
- 根拠: `run_with_timeout` の `raise TimeoutError(f"{label} exceeded {timeout}s timeout")`
- なぜテストで防げないか: `test_download_macro_prices_timeout` は `pytest.raises(TimeoutError)` のみ検証し、メッセージ内容を検証しない
- 修正方針: 対応不要
- 確信度: 高

#### [P3-003] fetcher.py の import が関数定義後に配置
- ファイルと行番号: `src/leadlag/data/fetcher.py:25,44-54`
- 発生条件: 常時（構造的問題）
- 影響: `from leadlag.utils.threading import run_with_timeout` が line 25 に、`from leadlag.data.cache import ...` が line 44 に配置されている。関数定義の後に import が続く構造は可読性を損なう。
- 根拠: 元のコードも同じ構造（`import threading` の後に import が続く）であり、本リファクタリングでは構造を維持
- なぜテストで防げないか: 構造的問題であり、テスト対象外
- 修正方針: import を全てファイル先頭に移動することが望ましいが、本リファクタリングのスコープ外
- 確信度: 高

## 4. 追加調査項目

なし — 全ての変更は確認済み。

## 5. カバレッジ表

| 観点 | 結果 |
|---|---|
| A1. ルックアヘッドリーク | 問題なし — `run_leakage_audit` の実検証ロジックが正しく機能。PIT履歴の日付フィルタリングは `load_pit_ir_history` で維持 |
| A2. ベースライン期間の分離 | 該当なし — 本変更で触れていない |
| A3. 市場中立制約 | 該当なし — 本変更で触れていない |
| A4. 数値安定性 | 問題なし — `run_with_timeout` の `result_box` パターンは元コードと同一 |
| A5. フォールバック挙動 | 問題なし — gapデータ欠損時のフラットポジション・フォールバックパスの `run_leakage_audit` 呼び出しは意図的にFAILEDを返す（正しい） |
| A6. コンプライアンス監査 | 問題なし — 監査項目の無効化なし。ハードコードTrueが実検証に置換された |
| B7. ロジックエラー | 問題なし |
| B8. 境界値・空値 | 問題なし — `pit_history_trade_dates` が空配列の場合 `pit_ok=True`（vacuously true） |
| B9. エラー処理 | 問題なし — P3-001のログメッセージ退化のみ |
| B10. 状態管理 | 問題なし — `run_with_timeout` はステートレス |
| B11. 並行処理 | 問題なし — daemon threadパターンは元コードと同一 |
| B12. 再試行・冪等性 | 該当なし |
| B13. 入力検証 | 問題なし |
| B14. 秘密情報 | 該当なし |
| B15. パフォーマンス | 問題なし — 余分なオーバーヘッドなし |
| B16. タイムアウト・ハング | 問題なし — タイムアウト機能は維持 |
| B17. 設定値 | 問題なし — タイムアウト値（60s/30s/30s）は全て維持 |
| B18. 依存関係 | 問題なし — 循環インポートなし（`leadlag.utils.threading` は外部パッケージのみ依存） |
| B19. テスト不足 | 問題なし — 新規3テスト + 既存テスト更新でカバー |
| B20. 重複実装 | 問題なし — 3箇所の重複を1箇所に集約完了 |
| B21. 変更誘発リスク | 問題なし — `run_with_timeout` のシグネチャが安定 |

## 6. 最終自己点検

- [x] 主要なエントリーポイントをすべて確認したか — `production_v2.py`, `run_daily_production_v2.py`, `run_daily_residual_blpx_shadow.py`
- [x] 正常系と異常系の両方を確認したか — タイムアウト・例外伝播・フォールバックパス
- [x] 処理経路を追跡したか — yfinance download経路3種・leakage audit経路2種
- [x] ドメイン固有リスクを確認したか — リーク・PIT・フォールバック
- [x] 2回目の反証レビューを実施したか — NaT入力・import順・後方互換性を確認
- [x] 指摘ごとに具体的な根拠があるか — 全てコード行番号付き
- [x] 未確認範囲を隠していないか — ruff未実行を明記
