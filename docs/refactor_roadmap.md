# リファクタリング・クリーンアップ ロードマップ

> **Note**: 本ファイルは 2026-08-10 の調査（プロジェクト全体の再設計案と、決定済み方針の未実行状況）を踏まえた **実行候補タスクのマスターリスト** です。  
> 実行時は AGENTS.md の不変条件（ルックアヘッド禁止・ベースライン期間分離・市場中立制約・ティッカー定義・前日 gap 行列使用禁止）を崩さないことを前提とする。  
> セクション 4 以降は **実装レベルの詳細設計**（実測値・コードスケッチ・移行手順・検証・ロールバック）を含む。

---

## 1. 背景・目的

### 1.1 現状の構造的負債

- **決定済み方針（ADR / Phase）の未実行が多数** 残っている。
- V1 系モデルの抽象層（`BaseModel` / `_BLPBase` / `SignalPipeline` 内の未使用 Combiner）が本番パッケージに残存。
- エントリポイント、設定、出力パスが複数系統に分裂。
- `var/` 移行は半端：ディレクトリ・シンボリックリンクはあるが、コード内のパスは未移行。
- テストが実ネットワーク / `market_data/etf_data.pkl` に依存し、クリーンクローンで再現困難。
- 日本語 + NFD/NFC Unicode 正規化が混在するリポジトリパスが移植性・自動化の妨げ。

### 1.2 目指す姿

- **本番 V2 のみ**: `ProductionV2Model` を中心とした最小・最も監査可能なコードベース。
- **1つの入口**: `python -m leadlag.cli` のみ。
- **1つの設定**: Pydantic `AppConfig` / `ProductionV2RunConfig`。
- **1つの出力木**: `var/` のみ。
- **完全気密のテスト**: 合成データのみ、マーカー駆動、2 分以内の unit。

---

## 2. ADR / Phase 未実行の完遂

以下はすでに「受け入れられた決定」だが、コードに完全に反映されていない項目。

| # | 元方針 | タスク | 未実行の実態 | 完了基準 |
|---|---|---|---|---|
| 2.1 | ADR-0002 | **Experiment Registry を稼働させる** | `leadlag.core.experiment_registry` がモジュール単独で、実験スクリプトから全く呼ばれていない。 | すべての `src/research/scripts/experiments/` 実行時に `ExperimentRecord` を JSONL へ追記。DSR 自動計算。 |
| 2.2 | ADR-0003 | **Data validation gates を strict に** | `preprocess_data` / `load_gap_matrices` に `strict` パラメータはあるがデフォルト `False`、本番ツール・`execution/` から `strict=True` が未呼出。 | 本番経路（`tools/production/`, `v2_bridge`）が `strict=True` を使い、無効データは即失敗 or flat position。 |
| 2.3 | ADR-0004 | **SqliteCacheStore への移行** | `data/cache_store.py` は作成済みだが、未使用。`data/cache.py` が pickle + fcntl のまま。 | `etf_data`・`decision_cache`・5分足キャッシュを SQLite 化 or 削除。pickle/fcntl 使用箇所ゼロ。 |
| 2.4 | ADR-0005 | **Frozen config helpers の採用** | `config/frozen.py` はテスト以外で未使用。研究スクリプトは `cfg.copy()` のまま。 | 全 `base_cfg.copy()` を `safe_config_copy` に置換。機械的 grep で根絶。 |
| 2.5 | ADR-0006 | **var/ への完全移行** | `var/` ディレクトリはあるが、コード内に `results/`, `artifacts/`, `logs/`, `live/` のハードコードが残る。`market_data/` はルート実体。 | `src/leadlag` / `tools/` 内でハードコードされた旧パスがゼロ。`market_data/` を `var/market_data/` へ移行。 |
| 2.6 | ADR-0008 | **テストマーカー徹底** | 57テスト中 2ファイルのみマーカー使用。`run_tests_unit_only.sh` は `--ignore` による除外で、マーカー未使用。 | `@pytest.mark.unit|integration|slow|property|leak` を全テストに付与。`run_tests_unit_only.sh` は `-m "not integration and not slow"` に。 |
| 2.7 | Phase 23 | **V1 抽象層の撤去** | `BaseModel`、`_BLPBase`、`sre.py` スタブ、`SignalPipeline` 内の SRE/BLP/RRR/Bayesian Combiner が残存。 | 上記を `git tag archive-2026-08` の `archive/legacy_src/` へ完全移設し `src/leadlag` から削除。削除後全テスト pass。 |
| 2.8 | Phase 24 | **helpers.py の削除** | `execution/helpers.py` が re-export shim として残存。 | `helpers.py` 自身を削除。既存 import を新モジュールへ機械的リダイレクト。 |
| 2.9 | Phase 25 | **全モデル・モジュールの Pydantic 一本化** | `sector_relative_ensemble_blp_enhanced.py` `__init__` 内 `_resolve_val` 61回。`ml_order_overlay.py` も `AppConfig \| dict` 受け入れ。 | 全モデル入口が `AppConfig` / `ProductionV2RunConfig` のみ。`_resolve_val` 削除。 |
| 2.10 | Phase 26 | **AppConfig 出力パスの一元化** | `config/schemas.py:191` と `execution/config.py:218` で `output_base_dir` デフォルト重複。`results/...` 直書き。 | 全出力ディレクトリが `AppConfig` 経由 or `var/` 決定。同じデフォルトが複数箇所に存在しない。 |
| 2.11 | Phase 27 | **研究スクリプト Pydantic 化** | `src/research/scripts/experiments/` 75本中14本のみ Pydantic 使用。`src/research/experiments/ml_order_decision/` は未移行。 | 全 `src/research/scripts/experiments/` と `src/research/experiments/` を Pydantic/AppConfig 化 or `src/research/` へ集約。 |
| 2.12 | Phase 28 | **SQLite gap / backtest store 一本化** | `GapStore` は `.npy` フォールバック付き；`BacktestResultStore` はテスト未使用。 | `.npy` フォールバック削除。`BacktestResultStore` を `BacktestEngine` で使用。 |
| 2.13 | Phase 29 | **daily CLI 一本化** | `daily` サブコマンドはあるが、`.sh` / `.plist` / `.bat` が個別に残る。 | `scripts/batch/run_decision*.sh`, `run_close_positions.sh`, 旧 plist を非推奨化・削除。 |
| 2.14 | Phase 30 | **テスト完全 hermetic 化** | `sample_df_exec` が yfinance ダウンロードに依存。 | `sample_df_exec` を合成 or 固定 fixture に置換。integration テスト以外はネットワークゼロ。 |

---

## 3. 新たな再設計提案

### 3.1 ディレクトリ構造

```
leadlag-fund/
├── README.md
├── pyproject.toml / uv.lock / .python-version
├── src/leadlag/
│   ├── cli.py                  # 唯一の入口
│   ├── config/
│   ├── domain/                 # 現 core/ を改名（I/O-free）
│   ├── data/
│   ├── models/v2/              # production_v2 を責務別に分割
│   ├── execution/
│   ├── compliance/
│   ├── reporting/
│   └── utils/
├── research/                   # 別パッケージ（本番 wheel から外す）
├── tests/ (unit / integration / e2e)
├── configs/ (base.yaml + production.yaml + experiments/*.yaml)
├── docs/ (adr / runbooks / specs)
└── var/                        # 唯一の実行時出力木
```

### 3.2 タスク一覧（サマリ）

#### 3.2.1 エントリポイント統一

- [ ] `scripts/run_v2_backtest.py` → `leadlag.cli backtest` 一本化
- [ ] `src/research/scripts/backtest/run_production_backtest.py` → `leadlag.cli backtest` へ吸収
- [ ] `tools/production/run_v2_decision.py` → `leadlag.cli decision` 一本化
- [ ] 実験用 V2 バックテスト4本を引数化（`--mode=exact|realistic|pessimistic|theoretical_5m`）
- [ ] `scripts/analyze_gap_bias.py` / `compare_gap_matrices.py` / `verify_gap_bias.py` を 1 本化
- [ ] `fix_mh_rankreversal.py` / `fix_pit_history.py` を `git tag archive-2026-08` の `archive/tools/` へ移動
- [ ] `run_research.py` エイリアスを `leadlag.cli research` サブコマンド or 削除
- [ ] `scripts/test/test_tachibana_*.py` を `tools/validation/` へ移設、pytest テストと区別
- [ ] Windows `.bat` / `.ps1` を非推奨化（macOS 運用一本化）

#### 3.2.2 設定 Pydantic 一本化

- [ ] `configs/base.yaml` 新設：全パラメータのデフォルト正本
- [ ] `configs/production.yaml` / `experiments/*.yaml` を base との差分のみに
- [ ] `StrategyConfig` / `AppConfig` / `ProductionV2RunConfig` の重複フィールド整理
- [ ] `config/frozen.py` の活用（shallow copy バグ根絶）
- [ ] `_resolve_val` 削除
- [ ] 出力パスデフォルトの重複解消

#### 3.2.3 データ・出力層

- [ ] `var/` 配下の canonical パスを `leadlag.config.paths` として定義
- [ ] 全スクリプト・コードの出力パスを `var/` 化
- [ ] `market_data/` → `var/market_data/` 移行
- [ ] `src/results/` 削除 or `var/results/` へ
- [ ] `GapStore` を gap 行列の第一正本に、`.npy` フォールバック廃止
- [ ] `BacktestResultStore` を `BacktestEngine` に統合
- [ ] `SqliteCacheStore` で pickle キャッシュを置換
- [ ] `live/` シンボリックリンク経由のパスを `var/live/` 直指定に

#### 3.2.4 本番コードの責務整理

- [ ] `core/pipeline.py` から V1 用 Combiner/Adapter（SRE/BLP/RRR/Bayesian）を削除
- [ ] `core/pipeline.py` を PCA/BLPX 用に分割・縮小
- [ ] `models/sector_relative_ensemble_blp_enhanced.py` を `research/` へ隔離 or 削除
- [ ] `models/base.py` `BaseModel` を `archive-2026-08` へ or 削除
- [ ] `models/blp_base.py` を V2 専用に縮小
- [ ] `sre.py` 内 `compute_jp_target_returns` を `data/` or `models/v2/` へ移動
- [ ] `core/experiment_registry.py` を `research/` へ移動 or 本番スクリプトに統合
- [ ] `monitoring/health_score.py` を `archive-2026-08` へ
- [ ] `reporting/sprint2c_lob_report.py` を `archive-2026-08` or `research/reports/` へ
- [ ] `data/cache.py` の `execution/` への逆依存を解消
- [ ] `data/cache.py` を `SqliteCacheStore` ベースに再実装 or 削除

#### 3.2.5 テスト改革

- [ ] 全テストに pytest marker 付与
- [ ] `sample_df_exec` の yfinance 依存を排除（合成 data へ）
- [ ] `tests/research/` の位置づけ明確化
- [ ] `run_tests_unit_only.sh` を `-m "not integration and not slow"` に修正
- [ ] unit テストの目標時間を 2 分以内に
- [ ] `test_pit_leak_property.py` の内容を全 signal 計算関数に展開

#### 3.2.6 ドキュメント・レポート

- [ ] ルート `README.md` 新設
- [ ] `docs/README.md`（旧 PCA-Ensemble 説明）を `git tag archive-2026-08` の `archive/docs/` へ
- [ ] `ARCHITECTURE.md` のツリー記述を現状に更新
- [ ] `運用方針書オリジナル.md` を `git tag archive-2026-08` の `archive/docs/` へ
- [ ] Phase 25〜30 を ADR 化
- [ ] AGENTS.md から「不採用実験の記録」セクションを `docs/experiment_graveyard.md` へ分離
- [ ] `reports/` の命名規則を統一
- [ ] `archive-2026-08` 実験資産を別 git repo or git tag へ移設

#### 3.2.7 環境・ツーリング

- [ ] `.venv` / `.venv-mac` / `.venv312` を一本化（`.python-version` + uv）
- [ ] Python バージョンを固定
- [ ] リポジトリ名を ASCII 化
- [ ] import-linter 導入（`data → domain → models → execution → cli`）
- [ ] `mypy --strict` 漸進的適用
- [ ] `_check_syntax.py` を `ruff check` への置換

---

## 4. タスク詳細（実装レベル）

### 4.1 P0（即座に着手）

---

#### T-P0-1. var/ ハードコードパス統一（ADR-0006 完遂）

**現状の実測（ハードコード一覧）**

| ファイル:行 | 値 | 種別 |
|---|---|---|
| `src/leadlag/cli.py:39` | `live/production_residual_blpx` | CLI デフォルト |
| `src/leadlag/execution/backtest.py:45` | `live/pipeline_data/gap_adjusted_distribution/latest` | gap dir フォールバック |
| `src/leadlag/execution/config.py:218-219` | `results/sector_relative_ensemble`, `live/sector_relative_ensemble` | デフォルト |
| `src/leadlag/config/schemas.py:191-192` | 同上（重複定義） | Pydantic デフォルト |
| `src/leadlag/execution/v2_bridge.py:43` | `live/production_residual_blpx` | デフォルト引数 |
| `src/leadlag/execution/microstructure/live_quote_logger.py:109` | `artifacts/sprint2c_lob_slippage/logs/quote_log.parquet` | sprint 固定パス |
| `src/leadlag/reporting/results_format.py` | docstring: 「`results/` が唯一の正本」 | 規約文書（要更新） |
| `src/leadlag/reporting/daily_pnl_report.py:486` | `results/...production_close_positions` | 参照パス |
| `tools/production/run_v2_decision.py:42,47,77` | `live/pipeline_data/...`, `live/production_residual_blpx`, `results` | デフォルト引数 |
| `tools/production/run_daily_production_v2.py:88` | `live/production_residual_blpx` | デフォルト引数 |
| `scripts/run_v2_backtest.py:24,28` | `live/pipeline_data/...`, **`src/results/v2_backtest`** | src 配下に出力（構造違反） |
| `src/research/scripts/backtest/run_production_backtest.py:35` | `results/production_backtest` | デフォルト引数 |

加えて物理実体: `market_data/`（ルート、約130MB の pkl/npz/csv）、`var/market_data/deprecated` のみ存在。ルートに symlink 5 本（`artifacts` `live` `logs` `results` `shadow_runs` → `var/…`、リンク先は NFD 表記 `/Users/shonen/日米ラグ/`）。

**設計**

新規 `src/leadlag/config/paths.py`:

```python
@dataclass(frozen=True)
class ProjectPaths:
    root: Path  # Path(__file__).resolve().parents[3] 基準、cwd 非依存

    @property
    def var(self) -> Path: return self.root / "var"
    def results(self, *parts: str) -> Path: return self.var / "results" / Path(*parts)
    def live(self, *parts: str) -> Path: ...
    def artifacts(self, *parts: str) -> Path: ...
    def logs(self) -> Path: ...
    def shadow_runs(self) -> Path: ...
    def market_data(self) -> Path: return self.var / "market_data"
    def gap_distribution_latest(self) -> Path: ...

def get_paths() -> ProjectPaths  # モジュールレベルシングルトン（テストで差替可）
```

**移行手順**

1. `paths.py` 作成 + unit test（パス解決が cwd 非依存であること）。
2. `schemas.py:191-192` と `execution/config.py:218-219` のデフォルトを `paths` 由来の `var/…` に変更（重複は `execution/config.py` 側を削除し `AppConfig` を参照）。
3. `cli.py:39` / `v2_bridge.py:43` / `backtest.py:45` / `live_quote_logger.py:109` を `paths` 経由に。
4. `tools/production/*.py` の argparse デフォルトを `paths` 経由に。
5. `data/cache.py::_data_dir()` の返り値を `var/market_data` に変更し、`market_data/` の pkl/npz を `var/market_data/` へ物理移動（`mv`、git 管理外のため安全）。
6. `src/results/` の中身を `var/results/` へ移動し、ディレクトリ削除。
7. ルート symlink 5 本は移行期間中残し、全スクリプト移行完了後に削除。

**検証**

- `bash scripts/run_tests_parallel.sh` 全 pass。
- `python3 -m leadlag.cli backtest --start-date 2026-08-01` 実行後、**ルートに新規ディレクトリが作成されない**こと、`var/results/` に成果物が出ること。
- 本番 dry-run（`run_daily_production_v2.py --dry-run true`）が `var/live/` に書くこと。

**ロールバック**: パス変更は git revert で戻せる。`market_data/` の物理移動は `mv` の逆操作で戻せる（symlink を残している間は新旧両パスで読める）。

**リスク/エッジケース**

- 本番運用中の `live/production_residual_blpx/latest_weights.csv` を `run_daily_production_v2.py:224` が trade_date 解決に読んでいる。**live データのパス変更は本番ジョブ停止中に実施**し、移行直後の初回実行で trade_date が正しく解決されることを確認。
- `sys.path.insert(0, str(ROOT / "src"))` + `ROOT / args.config` 形式のスクリプトは cwd に依存しないが、相対パス引数（`results/...`）は cwd に依存する。`paths.py` 化でこの揺れを排除。
- NFD/NFC: symlink 再生成時は `ln -s var/artifacts artifacts` のように**相対パス**で張り直す。

**依存**: なし（最初に実施可能）。T-P1-2（cache 移行）・T-P1-6（gap store）の前提。

---

#### T-P0-2. バックテスト入口 1 本化

**現状の実測（3 入口の差分）**

| 項目 | `scripts/run_v2_backtest.py` | `src/research/scripts/backtest/run_production_backtest.py` | `leadlag.cli backtest`（`execution/backtest.py::run_production`） |
|---|---|---|---|
| データ取得 | `load_df_exec_from_local_cache()`（キャッシュ必須） | `research.backtest_common.load_execution_data()`（cache or download） | `download_data()`（yfinance 毎回） |
| 引数 | `--config --gap-dir --start-date --end-date --output-dir --side-leverage --n-jobs` | `--config --start-date --output-dir --gap-dir --n-jobs` | `--config --start-date --slippage-bps` 等 |
| コスト指定 | config 経由 | config から個別変数を抜出して明示渡し | CLI 引数 + config |
| 出力 | 7 CSV（net/equity/DD/turnover/gross/costs/fallback）+ 独自 print | 12 CSV（+gross returns, cost 4 分解, weights）+ `calculate_metrics` | summary files（`save_summary_files`） |
| デフォルト出力先 | `src/results/v2_backtest` | `results/production_backtest` | `results/YYYYMMDD_HHMMSS_...` |

**設計**: `leadlag.cli backtest` に統一。追加引数:

- `--data-source {cache,download}`（デフォルト `cache` → yfinance ハング回避とも整合）
- `--end-date`（現行 CLI になし）
- `--side-leverage` / `--n-jobs`（`run_v2_backtest.py` 由来）
- `--output {minimal,detailed}`（7 CSV vs 12 CSV の出力セット切替、メトリクスは `calculate_metrics` に統一）

`execution/backtest.py::run_production` を拡張して上記を受け、`BacktestEngine.run_v2_backtest` 呼び出しはそのまま（エンジンは不変）。

**移行手順**

1. CLI/`run_production` に上記引数を追加。
2. `scripts/run_v2_backtest.py` を「CLI への thin wrapper + DeprecationWarning」に置換。
3. `run_production_backtest.py` も同様（`--detailed` 相当を CLI で再現できることを確認後）。
4. 1 リリース期間（または1週間）の並行運用後、2 ファイルを `git tag archive-2026-08` の `archive/tools/` へ git mv。
5. AGENTS.md「よく使うコマンド」更新。

**検証**: 同一 config・同一期間で 3 入口の `daily_net_returns.csv` を `np.allclose(rtol=0, atol=1e-12)` で比較。metrics（Sharpe/MDD/turnover）が一致。

**ロールバック**: wrapper 化の段階では旧スクリプトのロジックは残る。archive 移動後は git history から復元。

**リスク/エッジケース**

- `run_production` が毎回 yfinance ダウンロードする現行挙動はハング既知パターン（docs/スタック再発防止策.md）。`--data-source cache` をデフォルトにすることで CLI バックテストのハング頻度も下がる。
- `end_date="latest"` の解釈を 3 入口で統一（`load_df_exec_from_local_cache` の最終行）。
- `side_leverage` は Phase 20 で `AppConfig` 経由解決済み。CLI 引数は override として `model_copy` で反映。

**依存**: T-P0-1（出力先パス）を先にやると二重変更を避けられる。

---

#### T-P0-3. 死にコードの扱い決定と削除/移設

**現状の実測**

| モジュール | 行数 | 参照元 |
|---|---|---|
| `src/leadlag/core/experiment_registry.py` | 278 | `tests/unit/test_experiment_registry.py` のみ（archive 外の本番参照ゼロ） |
| `src/leadlag/monitoring/health_score.py` | 326 | `tests/unit/test_health_score.py` のみ |
| `src/leadlag/reporting/sprint2c_lob_report.py` | 130 | archive の sprint2c スクリプトのみ |

**注意（意思決定が先）**

- `experiment_registry` は **ADR-0002 で「稼働させる」と決定済み**。「削除」ではなく T-P1-1（稼働化）と統合し、`research/` 側へ移設した上で実験スクリプトから呼ぶ形にするのが矛盾しない。
- `health_score` は ARCHITECTURE.md Phase 9 で「記録・監視用のみ、サイズ調整には不採用」と決定済み。本番パスに無いので `archive-2026-08` か `research/monitoring/` へ。完全削除は「二度と使わない」判断が必要。
- `sprint2c_lob_report` は実験成果物 → `archive-2026-08` へ移設で確定。

**移行手順**

1. `sprint2c_lob_report.py` と `live_quote_logger.py`（sprint2c 固定パスを持つ）の本番パス使用を grep で最終確認 → `git tag archive-2026-08` の `archive/legacy_src/` へ git mv。
2. `health_score.py` + `test_health_score.py` を `archive-2026-08` or `research/` へ（判断後）。
3. `experiment_registry.py` + `test_experiment_registry.py` を T-P1-1 の実装先（`research/` または本番維持）へ移動。
4. `__init__.py` の export 整理、`grep -R "experiment_registry\|health_score\|sprint2c_lob" src/ tests/` が 0。

**検証**: 全テスト pass、CLI 動作、`ruff check` クリーン。

**リスク**: 参照ゼロの確認は文字列 grep だけでなく動的 import（`importlib`）も確認。

**依存**: T-P1-1（registry 稼働化）と順序調整が必要。

---

#### T-P0-4. 本番経路の strict validation 有効化（ADR-0003 完遂）

**現状の実測（strict パラメータの呼出状況）**

- `preprocess_data(..., strict_validation=False)` の現役呼出: `execution/backtest.py:98`、`data/cache.py:651`（`load_df_exec_from_local_cache` 内部の再構築）、`tests/conftest.py:37`、`src/research/scripts/experiments/compare_cumulative_method.py:56`。本番日次（`run_daily_production_v2.py`）は `preprocess_data` を直接呼ばず、`generate_v2_production_portfolio_with_overlay` 経由。
- `load_gap_matrices(..., strict=False)` の呼出: `models/production_v2.py`（re-export 経由）と `models/signal_enhancement.py`（multi-horizon / rank reversal）。
- `tools/`・`execution/` から `strict=True` の呼出は **ゼロ**。
- `validation.py:86-87` で target return（`jp_oc_*`）の NaN は許容済み（Phase 17 の 0 埋め挙動と整合）。つまり strict 化しても大引け前の当日行で誤爆しない設計になっている。

**設計（2 段ルール）**

- **欠損（missing）** → 現行通り flat position（w_final=0）。安全側の挙動を維持。
- **形状/有限性/PSD 異常（corrupt）** → `strict=True` で `DataValidationError` を raise。誤った行列で発注するリスクを遮断。
- `preprocess_data`: 本番経路では `strict_validation=True`。欠損行の silent skip をやめ、その日の実行を失敗させる（アラートで検知可能に）。

**移行手順**

1. `production_v2.py::_load_gap_or_flat` の `load_gap_matrices` 呼出に `strict` 相当の分岐を追加（missing→flat / corrupt→raise）。
2. `execution/backtest.py:98` と `data/cache.py:651` の `preprocess_data` を `strict_validation=True` に（cache 再構築で silent skip されて df_exec が途中切断される既知問題の検知が目的）。
3. `tools/production/run_daily_production_v2.py` / `compute_gap_adjusted_distribution.py` の呼出を確認し strict 化。
4. テスト: 欠損 gap → flat、破損 gap（非対称行列・非有限値）→ raise、preprocess の NaN 行 → raise、の 3 シナリオを unit test 追加。

**検証**: 既存の fallback 率（バックテストの `daily_fallback`）が変わらないこと（missing の挙動は不変のはず）。

**リスク/エッジケース**

- yfinance のティッカー別 NaN 欠損（AGENTS.md 既知の落とし穴）で本番が止まる頻度が上がる → データ修正フロー（pkl 検査）とセットで運用。strict 化で「気づけないまま flat」より「失敗して気づく」方が健全。
- shadow 検証期間: ADR-0003 本文の「move to strict=True after shadow validation」に倣い、まず `shadow_runs/` で 1-2 週間 strict の失敗率を観測してから本番反映。

**依存**: T-P0-1（パス）と独立。shadow 運用が前提。

---

### 4.2 P1（短期）

---

#### T-P1-1. Experiment Registry 稼働（ADR-0002 完遂）

**現状の実測**: `core/experiment_registry.py`（278 行）に `ExperimentRecord` / `ExperimentRegistry` / DSR 計算が実装済み。参照は `tests/unit/test_experiment_registry.py` のみ。AGENTS.md の「不採用実験の記録」は手書き。

**設計**

- 保存先: `var/experiments/registry.jsonl`（T-P0-1 の paths に乗せる）。
- 記録 API: デコレータまたはコンテキストマネージャ。

```python
with registry.record(
    hypothesis="PIT rolling window 756d improves Sharpe",
    config=app_config,           # model_dump() で差分保存
    tags=["pit", "walkforward"],
) as rec:
    results = run_backtest(...)
    rec.set_metrics(net_sharpe=..., max_dd=..., turnover=..., n_trials=12)
    rec.set_decision("rejected", reason="0/12 windows")
```

- DSR: registry 側で試行回数（同一仮説ファミリのレコード数）から自動計算。
- AGENTS.md「不採用実験の記録」は `docs/experiment_graveyard.md` に移し、各エントリに registry の record ID を付記。

**移行手順**

1. registry を `research/`（または本番維持）へ配置決定（T-P0-3 と連動）。
2. まず Pydantic 化済みの 14 スクリプト（`experiment_macro_kappa_v2.py` 等）に記録を組込み。
3. 新規実験テンプレート（`src/research/scripts/experiments/_template.py`）に必須化。
4. 残り 61 本は「触るタイミングで追加」の漸進ルール（一括改修はしない）。

**検証**: 1 実験実行で JSONL に 1 行、DSR が手計算と一致、registry から「同一仮説の試行一覧」が引ける。

**リスク**: 全スクリプト一括改修は差分が大きい → 漸進採用を明文化しないと形骸化する。

**依存**: T-P0-1（保存先パス）、T-P0-3（配置決定）。

---

#### T-P1-2. cache.py 分解と SqliteCacheStore 移行（ADR-0004 完遂）

**現状の実測**: `data/cache.py`（675 行、24 関数）の 4 責務:

1. ファイルロックプリミティブ（67-201 行: `_lock_with_timeout` / `file_lock` / `exclusive_lock` / `shared_lock`）
2. etf pkl I/O（`etf_pkl_path` / `load_raw_cache` / `save_raw_cache` / `is_pkl_cache_valid`）
3. 5分足キャッシュ（`load_intraday_cache` / `save_intraday_cache`）
4. 決定キャッシュ + VaR/ES 履歴再構成（303-636 行）

**逆依存の実体**: `cache.py:23` が `from leadlag.execution.config import build_app_config_from_dict`、`cache.py:373-374` が `from leadlag.execution.backtester import BacktestEngine`。`get_hist_returns_for_risk`（355-419 行）がキャッシュ無し時に **production.yaml を自前で読み V2 全バックテストを実行**している。

**設計**

- まず逆依存解消: `get_hist_returns_for_risk` を `execution/risk_capital.py` 側（または新 `execution/var_history.py`）へ移動。`data/` は純粋 I/O に。
- 次に保存形式: `etf_data` / `decision_cache` を `SqliteCacheStore`（`var/market_data/cache.sqlite`）へ。5分足は需要確認のうえ削除 or SQLite 化。
- pickle/fcntl は段階的に削除。移行期は「SQLite 優先・pkl フォールバック読み」→ 安定後 pkl 削除。

**移行手順**

1. `get_hist_returns_for_risk` の移動 + 呼出元（`risk_capital.py` 等）更新。
2. `load_df_exec_from_local_cache`（638-651 行、**56 ファイル**が使用）のインターフェースは維持したまま内部を SQLite 化。
3. pkl から SQLite への一度きり移行スクリプト（`tools/migrate_cache_to_sqlite.py`）。
4. fcntl ロック関数の使用箇所を洗い出し、SQLite 移行完了後に削除。

**検証**: `test_cache_store.py` 拡張 + 既存の cache 依存テスト全 pass。移行前後で `df_exec` が `pd.testing.assert_frame_equal` で一致。

**リスク/エッジケース**

- 56 ファイルが `load_df_exec_from_local_cache` を使う → **シグネチャは変えない**。
- pickle → SQLite の型再現性（DataFrame の index 型、tz）に注意。
- fcntl ハング（既知パターン）の解消効果が大きい一方、SQLite WAL でも NFS では問題が出る（本環境はローカルなので非該当）。

**依存**: T-P0-1（`var/market_data`）。

---

#### T-P1-3. Frozen config helpers の全面採用（ADR-0005 完遂）

**現状の実測**: `config/frozen.py`（`safe_config_copy` / `FrozenConfigDict` / `freeze_config_dict`）は `tests/unit/test_config_frozen.py` と `config/__init__.py` の export のみ。研究スクリプトは `cfg.copy()` / `deepcopy` 直書きが残る（AGENTS.md「config dict の shallow copy」の落とし穴が未根絶）。

**移行手順**

1. 対象列挙: `grep -R "\.copy()" scripts/ src/research/ src/research/experiments/` と `grep -R "deepcopy" src/ scripts/`。
2. dict 設定のコピーを `safe_config_copy()` に機械置換。比較実験の 2 モデル構成では必ず使用。
3. `blp_enhanced` 等の dict 受けモデルが残る間は、モデル入口で `freeze_config_dict()` を適用し書き換えを実行時エラー化。
4. 長期的には T-P1-4（Pydantic 一本化）で dict 自体を消し、`frozen.py` は research 用に縮小。

**検証**: 「2 モデルに異なる config を渡したとき互いに汚染しない」回帰テスト（Robust PCA 事故の再現テスト）を追加。

**リスク**: なし（防御的変更のみ）。

**依存**: なし。

---

#### T-P1-4. モデル層の Pydantic 一本化（Phase 25 完遂）

**現状の実測**

- `sector_relative_ensemble_blp_enhanced.py`: `__init__` が 62-222 行（160 行超）、`_resolve_val` を 61 回呼出。`config: dict | object` 受け入れ。ベンチマーク/実験専用（本番 V2 は `production_v2.py`）。
- `models/base.py`: `_resolve_val` / `_resolve_nested` / `_resolve_slippage_bps`（81-106 行等）が dict/Pydantic 両対応の複雑な解決ロジック。
- `ml_order_overlay.py:685`: `AppConfig | dict` 受け入れパターン。

**設計（分岐あり）**

- **案 A（推奨）**: `blp_enhanced` はベンチマーク専用なので `research/models/` へ移設（T-P2-1 と統合）し、本番 `models/` は V2 系のみ = Pydantic のみ。`_resolve_val` は research 側に残し、本番からは削除。
- **案 B**: `blp_enhanced` を本番に残して Pydantic 化 → 160 行の `__init__` を `BLPXConfig`（新規 Pydantic）に分解。工数大。

**移行手順（案 A）**

1. `blp_enhanced` + `blp_base` + `base.py` の研究利用を `research/` へ移動（T-P2-1 の一部を前倒し）。
2. 本番 `models/` に残るのは `production_v2.py` / `signal_enhancement.py` / `ml_order_overlay.py` / `sre.py`（`compute_jp_target_returns` のみ、移動先は T-P2-1）。
3. `ml_order_overlay.py` の `AppConfig | dict` を `AppConfig` のみに（境界で `model_validate`）。
4. `parse_run_config` は研究スクリプト用の境界変換として `research/` 側へ。

**検証**: 本番 V2 の `w_final` / `scores` / summary が before/after で一致（合成データ 7 シナリオ、Phase 22 B-1 と同じ検証手順）。

**リスク**: `blp_enhanced` は gap 分布計算（`compute_gap_adjusted_distribution.py`）のシグナル源でもあるはず → 移設時は tools/production からの import を確認。

**依存**: T-P2-1（V1 抽象層撤去）と一体。T-P1-3 完了後が望ましい。

---

#### T-P1-5. AppConfig 出力パス一元化（Phase 26 完遂）

**現状の実測**: `schemas.py:191-192`（`output_base_dir="results/sector_relative_ensemble"` / `output_live_dir="live/sector_relative_ensemble"`）と `execution/config.py:218-219`（dict 経由のフォールバック）で同一デフォルトが二重定義。`StrategyConfig` と `AppConfig` で役割が重複。

**移行手順**

1. T-P0-1 の `paths.py` 導入後、デフォルト値を `paths.results("sector_relative_ensemble")` 相当に一本化。
2. `execution/config.py` のフォールバック分岐を削除し、`AppConfig` 必須化。
3. `StrategyConfig` に残る実行系フィールド（start_date / costs / output）の `AppConfig` への格上げを検討（`AppConfig.strategy` の入れ子は維持）。

**検証**: `AppConfig()` を引数無しで構築したときの出力先が `var/...` である unit test。

**依存**: T-P0-1。

---

#### T-P1-6. GapStore / BacktestResultStore 一本化（Phase 28 完遂）

**現状の実測**

- `gap_matrix_io.py:100-105`: SQLite store は **パスが `.sqlite` / `.db` で終わる場合のみ** opt-in。本番 config の `gap_distribution.dir` は `live/pipeline_data/gap_adjusted_distribution/latest`（symlink → ディレクトリ）なので、**現行本番は常に .npy 経路**。
- `BacktestResultStore` は `tests/unit/test_backtest_store.py` のみ。
- `compute_gap_adjusted_distribution.py` が `mu_gap_{YYYYMMDD}.npy` / `omega_gap_{YYYYMMDD}.npy` を生成（マルチホライズン `mu_gap_h{h}_*.npy`、rank reversal 行列も）。

**設計**

- gap 行列の正本を `var/market_data/gap_matrices.sqlite`（`GapStore`）に。
- `compute_gap_adjusted_distribution.py` の出力を `GapStore.put()` 書き込みに変更（.npy は export オプション `--export-npy` で互換提供）。
- `load_gap_matrices` の .npy フォールバックは移行期間のみ残し、本番 config を `.sqlite` パスに切替後に削除。
- `BacktestEngine.run_v2_backtest` の `_assemble_v2_results` で `BacktestResultStore` に日次 PnL / weights を書き込み（CSV は従来通り出力）。

**移行手順**

1. 既存 .npy 群（`var/live/pipeline_data/gap_adjusted_distribution/` 配下の全日付分）を `GapStore` へ一括インポートするスクリプト。
2. 本番 config の `gap_distribution.dir` を sqlite パスに変更。
3. shadow 実行で .npy 経路と sqlite 経路の読み出し一致を検証。
4. .npy フォールバック削除。

**検証**: `test_gap_store.py` / `test_backtest_store.py` 拡張。`run_daily_production_v2.py --dry-run true` が sqlite から読み、同一 `w_final` を出す。

**リスク/エッジケース**

- 「当日行列不在 → flat」の不変条件（AGENTS.md #6）は sqlite 化後も維持（`store.get(date)` が None → flat）。**前日日付で検索するフォールバックは実装しない**。
- symlink（`latest`）経由の参照をやめると、日付解決を「store 内の最新 date_key」に変える必要がある。

**依存**: T-P0-1（パス）、T-P0-4（strict と整合）。

---

### 4.3 P2（中期）

---

#### T-P2-1. V1 抽象層の撤去（Phase 23 完遂 + 3.2.4）

**対象と処置（実測つき）**

| 対象 | 実測 | 処置 |
|---|---|---|
| `core/pipeline.py` の `SRECombiner` / `BLPCombiner` / `RRRCombiner` / `BayesianCombiner` + 各 OutputAdapter | 使用は `BLPXCombiner` のみ（`blp_enhanced.py:972-1221`） | アーカイブ済みモデル向けクラスを削除。`pipeline.py`（1561 行）を PCA/BLPX 用に縮小・分割（`pca_components.py` / `combiners.py` / `common_inputs.py`） |
| `models/base.py`（`BaseModel`） | V2 は未継承（手続き的）。研究側（`research/backtest_v1.py`）が使用 | `research/` へ移設 |
| `models/blp_base.py`（`_BLPBase`） | `blp_enhanced` のみが継承 | `blp_enhanced` とセットで `research/` へ |
| `models/sre.py` | `compute_jp_target_returns` のみ（71 行）。`backtester.py:25` / `blp_base.py:27` が使用 | 関数を `data/preprocessor.py` or `models/v2/target_returns.py` へ移動し `sre.py` 削除 |
| `models/sector_relative_ensemble_blp_enhanced.py` | 1243 行、ベンチマーク専用 | `research/models/` へ隔離（T-P1-4 案 A） |

**手順**: (1) `compute_jp_target_returns` の移動（本番参照あり、最優先で慎重に）→ (2) `blp_enhanced` 系を research へ → (3) `pipeline.py` 縮小 → (4) 各段階で全テスト。

**検証**: `ProductionV2Model` の出力一致（合成 7 シナリオ）。`gap 分布計算`（`compute_gap_adjusted_distribution.py`）が `blp_enhanced` を import している場合はその経路の動作確認。

**リスク**: `compute_jp_target_returns` は 9:10→大引けターゲットの正本。移動時にロジックを一行も変えない（移動のみ）。

---

#### T-P2-2. `execution/helpers.py` 削除（Phase 24 完遂）

**現状**: Phase 24 で `pricing.py` / `broker_ops.py` / `risk_capital.py` / `output_ops.py` / `post_decision.py` に分割済み、`helpers.py` は re-export shim。

**手順**: (1) `grep -R "execution.helpers\|execution import helpers" src/ tools/ tests/ scripts/` で全 import 元列挙 → (2) 新モジュールへの直接 import に機械置換 → (3) `helpers.py` 削除 → (4) 全テスト。

**検証**: `test_runner_helpers.py`（31 件）の更新と pass。

---

#### T-P2-3. 研究スクリプト集約（3.2.1 残り + Phase 27 完遂）

**現状の実測**: `src/research/scripts/experiments/` 75 本（Pydantic 化済み 14 本）、`src/research/experiments/ml_order_decision/` 3 本、`src/research/scripts/` 14 本。AGENTS.md（`src/research/scripts/experiments/` へ作れ）と Phase 14（`src/research/` へ統合）が併存。

**方針決定（先にやること）**: 実験の置き場を **`src/research/` に一本化**し、AGENTS.md の該当記述を更新。`src/research/scripts/experiments/` は新規作成禁止に。

**手順**: (1) 方針 ADR 化 → (2) `scripts/experiments/` を `src/research/scripts/experiments/` へ git mv（import パスは `research.` で済む）→ (3) `src/experiments/ml_order_decision/` を `src/research/experiments/` へ git mv → (4) `run_research.py` 削除、CLI `research` サブコマンド追加（任意）。

**検証**: 移動した代表スクリプト 3 本が動く。`src/research/scripts/experiments/` が空。

---

#### T-P2-4. ドキュメント・レポート整理（3.2.6）

**個別手順**

- ルート `README.md`: セットアップ（uv sync）/ 日次運用（`leadlag daily`）/ テスト / バックテストの 4 セクションで 5 分構成。
- `docs/README.md`（旧 PCA-Ensemble 説明）と `docs/運用方針書オリジナル.md` → `git tag archive-2026-08` の `archive/docs/` へ。
- `ARCHITECTURE.md` ツリー記述の更新: 削除済み `fast.py` / `decision.py::run_decision` / `sre.py` 第2フォールバック記述の除去、`daily` サブコマンド・`gap_store` の反映。Phase 履歴は残す。
- Phase 25-30 の ADR 化: 6 ファイル（`docs/decisions/2026-08-XX-*.md`）。
- AGENTS.md の「不採用実験の記録」→ `docs/experiment_graveyard.md` へ分離し、AGENTS.md は不変条件+コマンド+落とし穴に絞る（22KB → 半減目標）。
- `reports/`: 既存は動かさず、**今後の命名規則**のみ `reports/YYYYMMDD_<topic>/` に統一（過去分の一括リネームは git 履歴を汚すので任意）。

---

### 4.4 P3（長期）

---

#### T-P3-1. リポジトリ名 ASCII 化

**現状**: ワークスペース `/Users/shonen/日米ラグ` と symlink 先 `/Users/shonen/日米ラグ` は **同一 inode**（NFC/NFD 違い）。git・シェル・Python のパス比較で事故る種。

**手順**: (1) 新規 clone を `~/leadlag-fund` に作成 → (2) venv 再構築 → (3) var/ データの移動 or symlink → (4) cron/launchd のパス書換 → (5) 旧ディレクトリは移行確認後に削除。

**リスク**: 本番スケジューラ（plist）のパス書き換え漏れが最大リスク。切替は週末など市場閉場時に。

#### T-P3-2. `archive-2026-08` 完全分離

**現状**: 144 ファイルが git 管理下に同居（`git tag archive-2026-08` の `archive/legacy_src` 34 / `git tag archive-2026-08` の `archive/experiments` 101 等）。現役コードからの import ゼロ（確認済み）。

**手順**: (1) `archive-2026-08` を git subtree split or 別リポジトリ化 → (2) 本リポジトリから削除 → (3) 参照が必要なら git tag `archive-2026-08` に残す。

#### T-P3-3. `mypy --strict` 漸進適用

**手順**: (1) `pyproject.toml` に per-module strict 設定（新規ファイルは strict、既存は除外リスト）→ (2) 除外リストを週次で削減。ランタイム不変。

#### T-P3-4. 環境一本化・import-linter

**手順**: (1) `.python-version`（3.12 固定）+ `uv sync` に統一、`.venv-mac` / `.venv312` 削除 → (2) `import-linter` の contracts: `data → core → models → execution → cli` 一方向、`core` は I/O 禁止 → (3) CI/pre-commit に組込み。`_check_syntax.py` は `ruff check` で代替し削除。

---

## 5. 依存関係と実行順序

```
Wave 1（独立・即着手可）
  T-P0-1 var/ パス統一 ──────────────┐
  T-P0-3 死にコード処置決定           │
  T-P1-3 frozen config 採用           │

Wave 2（Wave 1 完了後）
  T-P0-2 バックテスト入口一本化（T-P0-1 の出力先に乗せる）
  T-P0-4 strict validation（shadow 検証を並走）
  T-P1-1 Experiment Registry 稼働（T-P0-1 の var/ 保存先、T-P0-3 の配置決定に依存）
  T-P1-2 cache 分解・SQLite 化（T-P0-1 の var/market_data に依存）

Wave 3
  T-P1-4 モデル層 Pydantic 一本化（T-P1-3 後、T-P2-1 と一体）
  T-P1-5 AppConfig 出力パス一元化（T-P0-1 後）
  T-P1-6 GapStore/BacktestResultStore 一本化（T-P0-1・T-P0-4 後）

Wave 4（P2）
  T-P2-1 V1 抽象層撤去 → T-P2-2 helpers 削除 → T-P2-3 研究スクリプト集約 → T-P2-4 ドキュメント整理

Wave 5（P3）
  T-P3-4 環境一本化 → T-P3-1 リポジトリ ASCII 化 → T-P3-2 archive 分離 → T-P3-3 mypy strict
```

各 Wave 内のタスクは並行可能。Wave 間は依存関係で順序付け。

---

## 6. 共通検証プロトコル

すべてのタスクで以下を実施:

```bash
# 1. 構文・lint・型
python3 _check_syntax.py
python3 -m ruff check src/leadlag/ --select E,F,W --ignore E501
python3 -m mypy --config-file pyproject.toml src/leadlag   # 新規エラーが増えないこと

# 2. 高速 unit（T-P0-4 等の対象変更時）
bash scripts/run_tests_unit_only.sh

# 3. 全体（リファクタ完了時）
bash scripts/run_tests_parallel.sh

# 4. 振る舞い一致（モデル/シグナルに触れる場合）
#    src/research/scripts/experiments/ に一時検証スクリプトを作成し、
#    before/after で predict_signals の signals / w_final を np.testing.assert_array_equal
```

**完了の定義**: 検証全 pass + レポート（`reports/<日付>_<task>/` に簡易記録）+ 該当する場合は ARCHITECTURE.md Phase 履歴へ追記。

---

## 7. 実行時の注意

- 各タスクは **1 変更 = 1 検証**。振る舞い一致テスト（例: `predict_signals` / `w_final` / `BacktestEngine` 出力）を必ず実施。
- リファクタリング中は `tests/unit/test_leakage_audit.py` および `tests/integration/test_production_residual_blpx.py` を高頻度で回す。
- AGENTS.md の不変条件を **崩さない**:
  1. ルックアヘッド禁止
  2. ベースライン期間 2010–2014 分離
  3. 全テスト pass
  4. 市場中立 net ±0.05 / gross ≤ 2.0
  5. `data/tickers.py` 単一正本
  6. 前日 gap 行列使用禁止
- 過学習ガード：新パラメータ追加時は `ExperimentRegistry` + DSR 記録を義務化。

---

## 8. 次のアクション

Wave 1 の 3 タスクから開始を推奨：

1. **T-P0-1 var/ パス統一**: `paths.py` 新設 → ハードコード 12 箇所置換 → `market_data/` 移動。
2. **T-P0-3 死にコード処置決定**: `sprint2c_lob_report` archive 移設は即実行可。`experiment_registry` / `health_score` は「稼働化 or archive」の意思決定が先。
3. **T-P1-3 frozen config 採用**: 機械的置換のみでリスク最小。

どのタスクから着手するか指示をください。
