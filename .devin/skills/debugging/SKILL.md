---
name: debugging
description: バグ・異常挙動の再現→最小化→計測→根本原因特定→修正→回帰テストのデバッグプロセスを実行する。フラットポジション発動・データパイプライン異常（NaN欠損・df_exec切断・キャッシュ陳腐化）・バックテスト指標異常・シグナルトレース・テスト失敗の症状別プレイブックを含む。予期しない挙動の調査・不具合修正時に必ず参照すること。
---

# Debugging スキル

## 目的

日米リードラグ戦略コードで発生したバグ・異常挙動を、再現→最小化→計測→根本原因特定→修正→回帰テストのループで解決する。**予防系スキル（code-review / edge-case-finder / leak-audit）が「起きる前」、本スキルは「起きた後」を担当する。**

## 他スキルとの棲み分け（症状→誘導先）

| 症状 | まず参照 |
|------|----------|
| プロセスがハング・スタック | `hang-prevention`（既知5パターンA-E） |
| 監査 FAIL・リーク疑い | `leak-audit` |
| 本番とバックテストの指標乖離・config不一致 | `prod-backtest-consistency` |
| 上記以外の全般（本スキル） | フラットポジション・データ欠損・異常値・テスト失敗等 |

重複する場合は本スキルのデバッグループを骨格にし、各専門スキルのチェックリストを計測・検証工程で使う。

## 実行規約（デバッグ固有・違反禁止）

- **`python3 -c "..."` のインライン実行は禁止**（AGENTS.md 実行規約）。検証用ワンライナーでも必ずスクリプトファイル化する。一時的なデバッグスクリプトは `scripts/experiments/` または `scratch/` に作成し、用後に削除する
- **長時間実行には必ずタイムアウト**。再現確認が30分コースの場合は対象を絞って再現時間を短縮してからループを回す
- **Python は `.venv` を使用**（`docs/スタック再発防止策.md` の `.venv-mac` 表記は旧環境）。CLI 経由の再現は `PYTHONPATH=src .venv/bin/python -m leadlag.cli ...`
- **デバッグ中に監査・テストを弱めない**。`ComplianceAuditor` の項目無効化、assert の緩和、テスト skip で「解決したことにする」のは禁止。根本原因を直す
- **config の比較は `copy.deepcopy`**。shallow copy の共有参照バグは「2系統の比較が完全一致する」という訳の分からない再現性として現れる（AGENTS.md 落とし穴）

## デバッグ手順（6ステップループ）

1. **再現**: 失敗を1コマンドで再現できる形にする。日次バッチ起因なら該当日のログ（下表）とコマンドを特定し、手動で同じ入口を叩く
2. **最小化**: 対象日・対象銘柄・対象関数まで絞る。バックテスト全体でしか出ない問題は、該当日だけの `run_live_aligned_v2_backtest.py --start-date X --end-date X` や単一テストに切り詰める
3. **仮説**: 症状別プレイブック（下記）の既知原因リストから仮説を立てる。該当しない場合は新パターンとして扱う
4. **計測**: `logger.info` や中間生成物のダンプで仮説を検証する。推測でコードを直さない。**コード修正前に「観測で原因を説明できる」状態にする**
5. **修正と回帰**: 根本原因を修正し、(a) 再現ケースが解消、(b) 既存テストが全パス、の両方を確認。可能なら先に失敗するテストを書いてから直す（`test-gen` 参照）
6. **事後処理**: 新パターンなら AGENTS.md「既知の落とし穴」または `docs/スタック再発防止策.md` に1行追記し、回帰テストを残す。再検証防止が目的

## 環境・パス早見表（2026-08-11 時点実測）

- ルートの `live/` `logs/` `results/` `shadow_runs/` `artifacts/` は **`var/` へのシンボリックリンク**。どちらのパス表記でも同じ実体を指す
- パス正本解決は `src/leadlag/config/paths.py`（`market_data()` → `var/market_data/`、legacy フォールバックあり）
- 市場データ: `var/market_data/`（`etf_data.pkl`, `etf_prices.sqlite`, `decision_cache.sqlite`, `df_exec.sqlite`, `etf_5m_data.pkl`）
- パイプライン前提データ: `live/pipeline_data/gap_adjusted_distribution/latest/matrices/mu_gap_YYYYMMDD.npy` / `omega_gap_YYYYMMDD.npy`、`distribution_diagnostics/latest/`
- PIT 状態: `live/production_residual_blpx/pit_binning.json`
- テスト並列ログ: `/tmp/pytest_parallel/`

### ログの対応表

| 処理 | ログ |
|------|------|
| V2 decision（9:10） | `logs/decision_YYYYMMDD.log` |
| gap 行列生成（6:30） | `logs/gap_distribution_YYYYMMDD.log` |
| Step1 行列生成（8:15） | `logs/distribution_diagnostics_YYYYMMDD.log` |
| 大引け決済（14:50） | `logs/close_positions_YYYYMMDD.log` |
| 市場データ更新（8:00） | `logs/update_market_data_YYYYMMDD.log` |
| launchd 由来 | `logs/*_launchd.log` |

## 症状別プレイブック

### 1. フラットポジション発動（w_final=0 が予期せず返る）

本番インシデントで最多。フォールバックは**仕様通りの安全側**なので「なぜ発動したか」の特定が先:

1. 当日の `logs/decision_*.log` で fallback / audit / gap のキーワードを検索
2. gap 行列の存在・日付確認: `ls live/pipeline_data/gap_adjusted_distribution/latest/matrices | grep $(date +%Y%m%d)`。不在なら `logs/gap_distribution_*.log` の鮮度チェック（`max trade_date != TODAY` で exit 1 → フラット化は正常動作）
3. **前日行列を当日日付でコピーして「復旧」してはならない**（不変条件6・2026-07-14 事故）
4. `live/production_residual_blpx/pit_binning.json` の `fallback_flag` / `history_count`（< 1000 や true なら正本 `full_history_diagnostics.csv` と `latest` シンボリックリンクを確認）
5. numerical audit 失敗が疑わしい場合は `run_numerical_audit` の FAILED 項目をログから特定（NaN/Inf・net 逸脱・非正定値のどれか）

### 2. df_exec 切断・NaN 欠損（バックテストが途中で終わる・日付が飛ぶ）

1. **第一容疑は yfinance ティッカー別 NaN**（IJR 等が特定日以降すべて NaN → `preprocess_data()` の NaN チェックで該当日の全レコードがスキップされ df_exec が切断。`preprocessor.py:264-272`）
2. `preprocess_data` 呼び出し前に `var/market_data/etf_data.pkl` を検査: 各ティッカーの最終有効日・NaN 列の有無を確認し、異常があれば `scripts/batch/update_market_data.sh`（`download_data(force=True)`）で再取得してから前処理
3. キャッシュ鮮度: `decision_cache.sqlite` は raw `etf_prices.sqlite` より mtime が古いと無効判定される（`decision_cache.py::is_decision_cache_valid`）。片方だけ手動更新すると**毎回リビルドが走る or 古い方が使われる**ので、両方の更新時刻を突き合わせる
4. `r_oc` NaN 行は0埋めで残す仕様（2026-07-14 修正）。行が消えていたら preprocessor の版を疑う

### 3. バックテスト指標の異常（Sharpe 激変・コスト異常・fallback 率スパイク）

1. 直近のコード/config 変更と `git diff` を突き合わせ、**再現時点の config 正本**（`configs/production/production.yaml`）で再実行
2. fallback 発動率 5% 超は要調査（edge-case-finder 観点4）。発動日を列挙してプレイブック1に誘導
3. コスト内訳（slippage / financing / borrow / reverse）を分解し、週末の暦日課金（`calendar_days` × `financing_daily`）が効いているか確認
4. 本番との乖離調査は `prod-backtest-consistency` の項目2・4・9（コード経路同一性・ウェイト再現性・実口座ズレ）へ
5. 「比較実験の両系統が完全一致」→ config shallow copy バグを疑う（`copy.deepcopy` で修正）

### 4. 特定日のシグナル/ウェイトトレース（なぜこのウェイトになったか）

段階ごとに中間値を観測する（すべて strictly historical、観測は当日行を含めない）:

```
US cc リターン → BLPX 構造化投影（signal.py）→ gap調整分布（mu_gap/Omega_gap）
→ mu_over_sigma スコア → MinVar weight（build_weights_minvar, α=0.8 ブレンド）
→ RuleD 動的グロス（PIT三分位）→ ML order overlay（phase2_8）→ w_final
```

- 本番同日再現: `python3 scripts/experiments/run_live_aligned_v2_backtest.py --start-date D --end-date D`（`sign_agreement=100%` かつ `weight_rmse < 0.02` が健全性基準）
- デバッグ用の一時ダンプは `logger.debug`/`logger.info` で追加し、解決後に除去するか level を DEBUG に下げる
- インスタンスキャッシュ（`self._production_signal_cache` 等）は `predict_signals` 開始時に clear される。トレース中にキャッシュ越しの値を見ていないか注意

### 5. テスト失敗

1. 失敗テストだけ再現: `python3 -m pytest tests/<path>::<test> -v --timeout=300`
2. 並列実行でのみ失敗 → ワーカー間の共有状態（グローバルキャッシュ・ファイルロック・`/tmp` 衝突）を疑う。直列で通るか確認
3. 直列でも失敗 → 直近変更との `git diff`、fixture（`tests/conftest.py`, `tests/fixtures/`）への影響を確認
4. 並列ログは `/tmp/pytest_parallel/`。全体回帰は `bash scripts/run_tests_parallel.sh`（約8分）
5. **テストを緩めて通さない**。テストが不変条件違反を検出している可能性を先に潰す（`leak-audit`）

### 6. ハング・スタック

`hang-prevention` スキルへ。既知5パターン（auto-close 無限待機 / yfinance / fcntl ロック / API バックオフ / フィル確認待ち）と復旧手順（プロセス kill → `results/.cache/*.lock` 削除 → ログ確認 → 手動再実行）を適用する。新パターンなら `docs/スタック再発防止策.md` に追記する。

## 計測・インスツルメンテーションの作法

- デバッグ print ではなく `logger` を使う（既存コードの作法）。長時間スリープの前後はハートビートログ（hang-prevention P5）
- 中間生成物をファイルに出す場合は `scratch/` か `outputs/experiments/<name>/` へ。`live/pipeline_data/` は運用データの正本なので**デバッグ出力を置かない**
- 再現スクリプトは `scripts/experiments/debug_<症状>_<日付>.py` の命名で作成し、解決後は回帰テストに昇格させるか削除する
- 数値比較は `np.testing.assert_allclose` / `assert_array_equal` を使い、`NaN != NaN` による false negative を避ける（NaN 含みは `equal_nan=True`）

## 完了条件

- [ ] 根本原因が観測データで説明できる（推測修正ではない）
- [ ] 再現ケースが解消し、全テストがパス（`bash scripts/run_tests_parallel.sh`）
- [ ] 回帰テストが追加された（または既存テストで担保済みと確認）
- [ ] 新パターンなら AGENTS.md 落とし穴 / 該当 docs に1行追記
- [ ] 一時デバッグコード・ダンプを除去（ログレベル低下 or 削除）
