# 網羅的バグ調査レポート（2026-08-22）

## 1. 結論

**BLOCK** — 本番影響のある問題が 3 件（P0 1 件・P1 2 件）見つかった。特に P0-001（マクロ因子の列入れ替わり）は本番の日次決定を現在進行形で劣化させており、即時修正を推奨する。P1-002（祝日テーブル誤り）は次の実害日 **2026-09-22（国民の休日）** までに修正が必要。

## 2. レビュー対象と調査範囲

- 対象: `src/leadlag/` 全般（models/v2, core, data, execution, broker, compliance, reporting, utils, domain, config）、`tools/research/compute_gap_adjusted_distribution.py`、`configs/`、`tests/`
- 確認した主要実行経路:
  - 本番決定: `cli.py decision` → `v2_bridge.run_v2_decision` → `ProductionRunner` → `ProductionV2Model.decide` → `FallbackPolicy`（file cache → on-demand → flat）→ `generate_v2_production_portfolio_from_distribution` → `_apply_pit_ruleD` → `_run_safety_audits` → `_apply_overlay`（ML overlay）→ `execute_post_decision_flow` → 発注
  - バックテスト: `cli.py backtest` → `BacktestEngine.run_v2_backtest` → 日次 `v2_model.decide` → `_simulate_daily_pnl`（コストモデル）
  - データ: `fetcher.download_data` → `preprocess_data` → `df_exec` → `PITDataLake`/`MarketSnapshot`
  - 前提データ: `compute_gap_adjusted_distribution.py`（Step 2, gap_store.sqlite）
- 実行した機械的検証:
  - `python3 -m compileall src/leadlag tests tools scripts src/research` → 全パス
  - `bash scripts/run_tests_parallel.sh` → **全 545 件 PASSED**（P1-P8 全ジョブ緑）
  - `ruff check --select E,F,W` → E402 のみ 6 件（`models/v2/__init__.py`、pyproject で E402 は意図的に ignore 済みのため非該当）
  - パターン走査: bare except / mutable デフォルト引数 / `== None` / グローバル可変状態 / `datetime.now()` 使用箇所 / `np.linalg.inv`
- 手法: 5 つの並行サブエージェント監査（V2 決定・コア数値・バックテスト/コスト・データ・実行/ブローカー）＋ 本体側での全指摘の個別検証（反証走査）。サブエージェント報告の約半数は検証の結果偽陽性と判定し排除した（§6 参照）
- 確認できなかった範囲: `kabu_auto_login/`（GUI 自動化、本番経路外）、`archive/`（凍結済み）、立花 API の実疎通（認証情報が必要）

## 3. 問題一覧

### P0 — 即時対応

#### [P0-001] マクロ因子の列ラベル入れ替わり（USDJPY ↔ 原油）— 本番稼働中

- ファイルと行番号: `src/leadlag/core/macro.py:198-203`
- 発生条件: `download_macro_prices` が実際の yfinance を呼ぶ全経路（本番決定・on-demand 計算・バックテスト）で macro_kappa_enabled / macro_direction_enabled = true のとき（本番 `configs/base.yaml` で両方 true）
- 影響: yfinance 1.5.2（uv.lock 固定）は複数ティッカーの `raw["Close"]` 列を**ソート順** `['CL=F', 'JPY=X', '^TNX']` で返すが、コードは要求順 `['JPY=X', 'CL=F', '^TNX']` を仮定して `close.columns = MACRO_NAMES` で位置代入する。結果として **"USDJPY" 列に原油先物（CL=F）、"CLF" 列にドル円（JPY=X）のデータが入る**（TNX は偶然正しい位置）。列数が一致するため例外も出ず完全にサイレント。
  - `compute_sigma_yy_inflation`: kappa=3.0（USDJPY 用の強い縮小）が**原油サプライズ**に、輸出セクター（自動車 1622.T 等）の USDJPY 感応度と組み合わされて適用される
  - `compute_macro_direction_adjustment`: 符号付き方向調整が誤った因子・誤ったセクターに適用される
  - バックテストも同一経路のため、マクロ機能の検証結果（採用判断を含む）がこのバグの影響下で計測されている
- 根拠（コード引用）:
  ```python
  # macro.py
  MACRO_TICKERS: list[str] = ["JPY=X", "CL=F", "^TNX"]
  MACRO_NAMES: list[str] = ["USDJPY", "CLF", "TNX"]
  ...
  if isinstance(raw.columns, pd.MultiIndex):
      close = raw["Close"]
  ...
  close.columns = MACRO_NAMES   # ← 位置代入。列順の検証なし
  ```
  実環境での実測（2026-08-22, yfinance 1.5.2）:
  ```
  raw['Close'].columns == ['CL=F', 'JPY=X', '^TNX']   # 要求順と異なる
  ```
- なぜテストで防げないか: `tests/unit/test_macro.py::_make_mock_yf_download` が `pd.MultiIndex.from_product([["Close"], close_data.columns])` で**要求順を保持するモック**を返すため、実際の yfinance のソート挙動との差が検出されない
- 修正方針:
  ```python
  close = raw["Close"].reindex(columns=MACRO_TICKERS)
  if close.columns.isna().any() or len(close.columns) != len(MACRO_TICKERS):
      raise RuntimeError("macro download missing tickers")
  close.columns = MACRO_NAMES
  ```
  併せて「実 yfinance の列順を再現するモック（ソート順）」の回帰テストを追加
- 確信度: 高（実環境で実測確認済み）

### P1 — 本番障害・主要機能の誤動作

#### [P1-002] 祝日静的テーブルの誤り + jpholiday 未導入 — 2026-01-05 に実取引日を静かにスキップ（済）、2026-09-22・12-31 が残存リスク

- ファイルと行番号: `src/leadlag/core/market_calendar.py:27-71`、呼び出し元 `src/leadlag/cli.py:413-429`、`src/leadlag/execution/v2_bridge.py:112-115`、`src/leadlag/models/v2/decision_engine.py:44`、`src/leadlag/execution/var_history.py:55-57`、`src/leadlag/data/preprocessor.py:259`、`src/leadlag/data/market_data_cache.py:78`
- 発生条件: `jpholiday` は pyproject で optional extra（`calendar`）だが現行 `.venv` に未導入のため、`_is_holiday_static` の静的テーブルが本番の権威になっている。そのテーブルに以下の誤りがある（JPX 公式カレンダーで照合済み）:
  - 2025: 誤って休場扱い → `1/6`（通常営業日）, `3/11`（「通常はなし」とコメントにありながら登録）, `5/2`（通常営業日）。欠落 → `3/20`（春分の日）, `12/31`（大納会翌日・休場）
  - 2026: 誤って休場扱い → `1/5`（年初営業日）。欠落 → `3/20`（春分の日）, `9/22`（国民の休日）, `12/31`
  - 両年とも `12/31` 欠落のため、`previous_trading_day` が年始に実休場日 12/31 を返す（例: `previous_trading_day(2026-01-05)` → 2025-12-31）
- 影響:
  1. `cli.py:419`: `decision`（trade_date 未指定）/`close`/`daily` が `is_market_closed(today)` で **終了コード 0 の静かなスキップ**。2026-01-05（実営業日）は既にスキップ済みの可能性。2026-09-22（実休場日）には逆に decision が走り、当日データなし → フラットまたはエラー
  2. `_derive_signal_date`（decision_engine.py:44）が誤った signal_date を算出し、`run_leakage_audit` の signal_date 検証の精度が劣化（`sig < trade` は通るため監査は FAIL しない）
  3. `var_history` / `market_data_cache` の staleness 営業日カウントが祝日前後で ±1 ずれる
  4. `preprocessor.py:259` のプレースホルダー翌営業日推定が誤った日付を作る（実休場日の provisional 行など）
- 根拠: `_STATIC_HOLIDAYS`（market_calendar.py:28-70）と JPX 公式休業日一覧の照合。`_is_holiday_jpholiday` は ImportError で None を返し静的テーブルにフォールバック（同 79-89）
- なぜテストで防げないか: テストは既知の祝日・週末の代表例のみで、テーブルの網羅的正しさ（実カレンダーとの照合）は検証していない
- 修正方針: (a) テーブル訂正（上記 7 日の修正＋12/31 追加、可能なら 2027 も） (b) jpholiday を主依存関係へ昇格するか、未導入時に起動時 WARNING を出す (c) 「テーブル↔JPX 公式」の照合テストを追加
- 確信度: 高

#### [P1-003] ポジション取得失敗時に差分計算がスキップされ全量新規発注（二重発注リスク）

- ファイルと行番号: `src/leadlag/execution/v2_bridge.py:403-408`、`src/leadlag/execution/broker_ops.py:294-343, 530-534`
- 発生条件: `--api-enable` 本番実行中に `fetch_current_positions(api_client)` が例外（ネットワーク断・セッション切れ等）
- 影響: 警告ログのみで `current_positions=None` のまま継続 → `_build_order_deltas` で `current={}` 扱い → **当日ターゲット全量が新規注文として発注される**。同日再実行（冪等性が前提の運用）や前日クローズ失敗で建玉が残っている場合に、ポジションが二重になる。日計り信用の戦略上、朝イチは通常フラットだが「再実行」「前日持ち越し残存」の 2 シナリオで実害がある
- 根拠:
  ```python
  # v2_bridge.py
  try:
      current_positions = fetch_current_positions(api_client)
  except Exception as e:
      logger.warning("Failed to fetch current positions: %s. Will submit full target.", e)
  ```
  ```python
  # broker_ops.py
  current = current_positions if current_positions is not None else {}
  ...
  delta = target_signed - current.get(ticker, 0)   # 常に delta = target
  ```
- なぜテストで防げないか: 例外経路と発注差分の結合テストがない
- 修正方針: `api_enable` 時はポジション取得失敗で**中止（fail-closed）**にするか、N 回リトライ後に中止。少なくとも「同日の発注済み証跡（api_execution_log）がある場合は中止」ガードを追加
- 確信度: 高

### P2 — 特定条件での不具合・運用上の実害

#### [P2-001] `--api-enable` なしの decision が 1000 円ダミー価格由来の gap を ML overlay 特徴量に流し、本番ウェイトファイルを上書き

- ファイルと行番号: `src/leadlag/execution/v2_bridge.py:273-274`（ダミー価格）、`320-336`（`jp_gap_api = 1000/prev_close - 1` で −50〜−90% の偽 gap）、`352`（`write_production_files` は dry_run 時のみ抑止）、`src/leadlag/models/ml_order_overlay.py:185-190`（snapshot gap を特徴量に使用）
- 発生条件: `decision` を `--api-enable` も `--dry-run` も付けずに実行
- 影響: file cache から読む mu_gap/Omega_gap 自体は健全だが、(1) on-demand フォールバック発動時は偽 gap で mu_gap を再計算、(2) ML overlay（本番有効）の特徴量が偽 gap で歪む、の 2 経路でウェイトが汚染され、それが `live/latest_weights.csv` 等に保存される。翌日の `--trade-date latest` 解決にも影響
- 増悪要因: `MarketSnapshot.validate()`（20% 閾値で偽 gap を検出可能）は `src/` 内のどこからも呼ばれておらず防御が未配線（`pit_lake.py:41-76` は定義のみ）
- なぜテストで防げないか: 統合テストは api_enable あり/なしの経路差を検査していない
- 修正方針: (a) api 無効時は `write_production_files` を抑止（dry-run 同等扱い） (b) `MarketSnapshot.validate()` を `run_v2_decision` の snapshot 構築後に配線し、失敗時はフラット/中止
- 確信度: 高（コード経路確認済み。実運用で非 api 実行をしない運用規約なら実害は限定）

#### [P2-002] バックテストのコストモデル: オーバーナイト保有分のスリッページが片道の半分しか計上されない

- ファイルと行番号: `src/leadlag/execution/backtester.py:172-175`
- 発生条件: `overnight_alpha_long/short > 0`（本番 config: 0.75/0.5 で常時発動）
- 影響: 日次決済分は `2.0 × |w_t|`（往復 = 片道×2、正しい）だが、保有分は `|Δw| / 2.0`。片道スリッページは「取引したノーショナル 1 単位ごとに slip」なので、保有分の当日取引量 `Σ alpha·|w_t − w_prev|` に対して正しくは `slip × Σ alpha·|Δw|`。現状はその半額。本番 config（保有比率が大きい）では net Sharpe が**過大評価方向**にバイアスされる。概算で日次 1〜3 bps 程度の過小計上になりうる（保有分の日次 churn 依存）。V1/V2 が同一 `_simulate_daily_pnl` を共有するため、過去の実験間の相対比較は保全されるが、絶対的な net 水準は割高
- 根拠:
  ```python
  slip_cost = side_leverage * slip * (
      2.0 * np.sum((1.0 - alpha_mask) * np.abs(w_t))        # 往復: 正しい
      + np.sum(alpha_mask * np.abs(w_t - w_prev) / 2.0)     # 保有分: 片道の半分（過小）
  )
  ```
- なぜテストで防げないか: コスト式の単位テストは「既知の実装値との一致」を見ており、経済的正当性（往復/片道の単位整合）は検証していない
- 修正方針: 保有分を `slip × Σ alpha_mask·|w_t − w_prev|` に修正し、修正前後で本番期間の net Sharpe 差分をレポート化（評価指標の約束事に基づく再計測）
- 確信度: 中（コード上の不整合は確実。影響量は churn 実測依存のため中）

#### [P2-003] バックテストで「日ごとに 2 年分のマクロ時系列」を yfinance から再ダウンロード

- ファイルと行番号: `src/leadlag/models/v2/fallback.py:50-55`（`macro_start = date_str − 2y`, `macro_end = date_str`）、`src/leadlag/core/macro.py:166-169`（キャッシュキー `(start, end, period)`）
- 発生条件: マクロ調整有効のまま全期間バックテストを実行（本番 config では有効）
- 影響: 日次で end 日が変わるためキャッシュキーが全日ユニーク → **各シミュレーション日に yfinance ダウンロード**（30 秒タイムアウト付き）。約 2800 営業日 × 1〜2 秒で、バックテスト全体の実行時間と外部 API 負荷を大きく増やす（n_jobs>1 の loky ワーカーはプロセス別キャッシュのためさらに増殖）。タイムアウト/レート制限時は `_repair_and_adjust` が例外を飲んでマクロなしで継続するため、**日によってマクロ調整の有無がばらつく**という一貫性問題もある
- 修正方針: バックテスト開始時に全期間分を 1 回ダウンロードし、日次は `close_prices.index < date_str` でスライスする方式に変更（PIT 制約は既に fallback.py:59 の `< date_str` で担保済み）
- 確信度: 高

### P3 — 変更時の不具合誘発・保守上の実害

#### [P3-001] マクロ調整の部分適用（Omega 膨張のみ反映され mu 調整が欠ける経路）
- `src/leadlag/models/v2/fallback.py:75-118`: kappa による Omega_gap 膨張（80 行目で代入済み）の後に direction 調整が例外を投げると、`except` で継続するため **Omega だけ膨張した不整合な分布**でウェイト計算が続行される。例外を kappa ブロックと direction ブロックで分割するか、例外時は両方とも適用前にロールバックすべき。確信度: 中

#### [P3-002] `_gap_alerts_fatal` がアラート文字列の文言（"shape" / "non-finite"）に依存
- `src/leadlag/models/v2/gap_io.py:93-101`: `validate_gap_matrices`（validation.py:262-275）のメッセージ文言を変えると fatal 判定が静かに無効化される。`validate_gap_matrices` が構造化した結果（fatal フラグ）を返すように改めるべき。確信度: 高

#### [P3-003] 監査失敗によるフラット化でも `fallback["gap_data_missing"]=True` が立ち、ログが "Gap data missing" と誤表示
- `src/leadlag/models/v2/audit_comparator.py:121-125` → `src/leadlag/execution/v2_bridge.py:354-356`。運用監視で原因を誤診する。`fallback` に `audit_failure` キーを分離すべき。確信度: 高

#### [P3-004] `STRATEGY_SLIPPAGE_BPS` 環境変数が設定しても効かない
- `src/leadlag/execution/config.py:204` で StrategyConfig.slippage_bps には注入されるが、`run_v2_backtest` の `_resolve_v2_backtest_cost_params`（backtester.py:398-400）は `v2.costs.slippage_bps_per_side` を優先し、config に同キーが存在する限り env 値は読まれない。確信度: 高

#### [P3-005] `_resolve_sim_dates` の end_date が非営業日のとき翌営業日まで含まれる
- `src/leadlag/execution/backtester.py:50`: `searchsorted(end_dt)` は left 挿入位置を返すため、end_date が休日・週末の場合にその翌営業日が範囲に混入する。`side="right"` なら 1 減らせる。`end_date="latest"`（デフォルト）では不発。確信度: 高

#### [P3-006] allocator のネットバランス・ループが 500 ステップで警告なしに打ち切り
- `src/leadlag/core/allocator.py:139-217`: 収束しなかった場合でも警告なく終了し、net 制約（±0.05）をわずかに超過した数量が通る可能性。ループ後に `abs(net) > net_limit` なら WARNING を出すべき。確信度: 中

#### [P3-007] ML overlay の `w_pre = w_final / mult` は mult=0 を許容する config スキーマ上の経路
- `src/leadlag/models/ml_order_overlay.py:658-660`、`schemas.py:629-631`（`mult_low: ge=0.0`）。mult=0 の日は `0/0` → `_safe` で全零 → "Overlay collapsed one side" で元のフラット結果に戻るため現状は事故らないが、脆い。mult=0 を「その日は取引しない」の表現として使うなら明示的なガードが望ましい。確信度: 中

#### [P3-008] on-demand フォールバック有効性の getattr デフォルト値が 2 経路で不一致
- `src/leadlag/models/v2/gap_io.py:274`（default True）vs `src/leadlag/models/v2/distribution_source.py:192`（default False）。`ProductionV2RunConfig` が常に属性を持つため現状は不発だが、属性欠落時に新旧経路で挙動が分かれる。確信度: 高（不発だが不整合）

#### [P3-009] 実行層の gross ゲートが実質不活性
- `risk_capital.auto_adjust_gross_exposure` は `max_gross_exposure=3.0`（base.yaml risk 節）とウェイトの gross（RuleD 後 ≤2.0、レバレッジ前）を比較するため、構造上発動しない。防御深度として意図的なら問題ないが、閾値の単位（レバレッジ前後）が混在している点は文書化すべき。確信度: 中

## 4. 追加調査項目（根拠不足・断定保留）

1. **キャッシュの updated_at タイムゾーン混在の可能性**: `market_data_cache.py:115-119` / `decision_cache.py:162-169` が `datetime.now(UTC).replace(tzinfo=None)` と他の naive 時刻を比較する経路。両側とも UTC-naive で統一されているか実データで要確認
2. **VaR/日次・月次損失ストップの入力系列**: `execution/var_history.py` がモデル由来の履歴リターンを使っており、**実現損益（約定ベース）ではない**。ストップの設計意図（モデル監視 vs 実損監視）を確認すべき
3. **`compute_entry_cost_bps` 系（microstructure）の fallback 15bps と本番 5bps の不整合**: 現行コードでは `CostCalculator` に `src/` 内の呼び出し元がなく休眠状態。将来配線する際に不整合が顕在化する
4. **`_ROLLING_CORR_CACHE` / `CacheManager` の `hash()` ベースキー**: プロセス内限定のため現状実害なし。SipHash 衝突確率は無視できるが、永続化する場合は要再検討
5. **`market_data_cache`/`gap_store` の SQLite 30 秒タイムアウト**: 高負荷時の挙動未検証
6. **`_compute_jp_target_returns_h`（h>1）で無効データ日のターゲットが 0.0 になる**（preprocessor.py:756）: h>1 の研究経路で 0% リターンとして混入しうる。現行本番は h=1 のみ使用のため影響限定

## 5. カバレッジ表

| 観点 | 結果 |
|---|---|
| A-1 ルックアヘッドリーク | 問題なし（窓・ベータ・PIT・マクロとも strictly historical を個別確認。偽陽性 2 件を反証で排除） |
| A-2 ベースライン期間分離 | 問題なし（2010-2014 固定・空期間は意図的に ValueError） |
| A-3 市場中立制約 | 概ね問題なし（構造で担保。P3-006/009 は限定的） |
| A-4 数値安定性 | 問題なし（PSD 修復・NaN 処理・特異時フォールバックは実装済み） |
| A-5 フォールバック挙動 | 問題あり（P1-003, P2-001, P3-003, P3-008） |
| A-6 コンプライアンス監査 | 問題あり（P1-002 由来の signal_date 精度劣化。監査の無効化はなし） |
| B-7 通常系ロジック | 問題あり（P0-001） |
| B-8 境界値・空・NaN | 問題なし（主張 4 件を検証し全て防御済みと確認） |
| B-9 例外処理 | 問題あり（P1-003, P3-001） |
| B-10 状態管理・キャッシュ | 問題あり（P2-003。グローバルキャッシュの汚染なし） |
| B-11 並行・競合 | 問題なし（loky プロセス分離、SQLite トランザクション） |
| B-12 冪等性 | 問題あり（P1-003） |
| B-13 入力検証・情報漏えい | 問題なし（秘密情報のログ出力なし） |
| B-14 秘密情報・ログ | 問題なし |
| B-15 パフォーマンス | 問題あり（P2-003） |
| B-16 タイムアウト・ハング | 問題なし（close.py の auto-close は max_wait_hours で bounded。yfinance/メール/API いずれもタイムアウト済み） |
| B-17 設定値・環境変数 | 問題あり（P3-004, P1-002 の jpholiday extra 化） |
| B-18 依存関係 | 問題あり（P0-001: yfinance 1.5.2 の列順挙動） |
| B-19 テスト不足 | 問題あり（P0-001 のモック乖離、P1-003 の例外経路） |
| B-20 重複・到達不能コード | 問題あり（`MarketSnapshot.validate()` 未配線、`CostCalculator` 休眠） |
| 日付・営業日カレンダー | 問題あり（P1-002） |

## 6. 反証走査で排除した主な偽陽性（二重検証防止のため記録）

| 主張 | 反証 |
|---|---|
| vol 調整 20 日窓に当日行が混入（signal.py:126, signal_computer.py:117, gap_adjustment.py:151） | いずれも `[current_index-20 : current_index]` で終端排他。**当日行は含まない**。PITMatrixView.historical_range も `end <= as_of` を強制し `[start:end]` 排他 |
| residualize / preprocessor の rolling beta が当日行を使用 | 1 次元経路は `.shift(1)`、多次元経路は `[t-window : t]` スライスで正しく除外。EWMA 経路も `[t-beta_window : t]` |
| build_weights_minvar が特異行列を未チェック | `_solve_minvar_sub` が NaN/Inf 置換・固有値 PSD 修復・solve 失敗時 raw フォールバックを実装済み |
| `_ROLLING_CORR_CACHE` の `hash(tobytes())` でキャッシュがヒットしない | コンテンツハッシュなので同一内容は正しくヒットする（誤） |
| compute_baseline_correlation が例外で停止 | AGENTS.md の不変条件 2 に基づく意図的フェイルファスト |
| close.py auto-close の無限待機 | `max_wait_hours` 事前ガード＋単調減少の remaining で bounded |
| オーバーナイト `gap_returns[i+1]` のインデックスずれ | 行規約（行 t の jp_gap は取引日 D_{t+1} の寄付）と整合し正しい |
| horizon>1 で p_start が空配列化し誤計算 | n < horizon 時は全行 NaN（先頭 horizon−1 行 NaN 化でカバー）で graceful |
| 中央値補間が NaN で失敗 | 50% 有効数チェックが先行し、非有限レコードは validate_exec_record で drop |
| PIT 履歴不足時に危険なビニング | `get_rolling_pit_bin` が ('Medium', nan, nan, 1.0) にフォールバック（docstring 明記の仕様） |
| gross_exp に side_leverage 未適用 | docstring 明記のレポート規約（raw weight 基準で報告） |
| `_sector_mapping_indices` 未初期化 | model.py:271-279 で初期化済み |
| マクロサプライズが当日値でリーク | `surprise_z[t]` は `vals[t]` と t−1 までの EWMA で計算。正しい |
| t-copula ν 非収束で破綻 | 初期値保持のフォールバックとして妥当（ログ不足のみ） |

## 7. 最終自己点検

- [x] 主要エントリーポイント（cli decision/backtest/close/daily、v2_bridge、ProductionRunner、BacktestEngine、compute_gap_adjusted_distribution）を確認した
- [x] 正常系・異常系の両方を確認した（フォールバック連鎖・例外経路を個別追跡）
- [x] ファイル単位ではなく処理経路を追跡した（df_exec 生成→スナップショット→分布解決→ウェイト→発注）
- [x] ドメイン固有リスク（リーク・中立・数値安定性・フォールバック）を確認した
- [x] 2 回目の反証レビューを実施した（サブエージェント全指摘を本体で再検証し約半数を排除）
- [x] 指摘ごとにファイル:行・発生条件・影響・根拠・修正方針を記載した
