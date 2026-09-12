---
name: leadlag-fund-improvement
description: 日米リードラグ戦略の改善・V2バックテスト・本番昇格で、設定と実行経路を特定する。
---

# 日米リードラグ・ファンド改善

共通規約・不変条件はリポジトリルートの `AGENTS.md` を参照する。本スキル内のパスも同ルート基準。

## 現行経路の確認

1. 対象の config と `__base__` 継承先を `load_config_from_yaml` で解決する。本番比較・昇格の基準は `configs/production/production.yaml`。旧 RuleD config や過去レポートの「最適値」で代用しない。
2. `src/leadlag/models/production_v2.py` を入口に、対象処理を `src/leadlag/models/v2/` まで追う。分布取得は `fallback_policy.py` / `distribution_source.py`、監査と要約は `audit_comparator.py` に分かれる。
3. データパスは `src/leadlag/config/paths.py` と有効設定から解決する。標準 gap store は `var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite`。旧 `live/` リンクや Git 管理の有無をバックアップ保証とみなさない。
4. cache / on-demand の整合検証時は `shadow_ondemand_validation` を確認する。`true` は cache 読込時に on-demand と比較する追加検証。必要なモデル・入力があることを確認する。現行比較は平均ベクトル/共分散の相対差が1%を超えると警告するが、警告だけで非リークや本番昇格の妥当性を保証しない。

## バックテストと日次経路の違い

- バックテストは `BacktestEngine.run_v2_backtest()` または CLI `backtest` を使う。対象期間の gap store の日付・銘柄順・生成設定を検査する。
- `ProductionV2Model` の on-demand 計算には BLPX モデルと必要な履歴・当日入力が必要。日次経路で利用可能でも、バックテストや単純ラッパーで同じ依存が注入されるとは限らない。cache 欠損のまま実行して全日フラットを戦略成績と誤認しない。
- gap 事前計算の入口は `tools/research/compute_gap_adjusted_distribution.py` と `scripts/batch/run_gap_distribution.sh`。必要な上流成果物・鮮度は実際の引数と読込処理で確認する。アーカイブ内の V1 スクリプトを日次の必須工程として復活させない。
- 日次バッチは `scripts/batch/run_decision_v2.sh` から CLI / V2 同期経路へ進む。開始時刻とデータの利用可能時刻を区別し、当日 gap の鮮度を検証する。

## 作業の分岐

- 性能を変える仮説・パラメータ比較は `experiment-design` を使う。単純な挙動維持の整理を新規実験扱いしない。
- 時系列計算・監査・フォールバックの挙動を変更・検証するときは `leak-audit`、具体的な境界リスクの分析には `edge-case-finder` を使う。
- バックテストを実行したら `backtest-report` で結果を保存する。未実施の OOS・監査・コスト内訳を PASS やゼロで埋めない。
- 本番昇格では実験結果だけでなく、同一データ・設定の shadow 整合を確認する。`tools/validation/monitor_residual_blpx_shadow_performance.py` の入力仕様を読み、cache / on-demand、ウェイト、コスト、フラット化の差を追う。実験採用だけを根拠に本番設定・スケジューラ・注文を変更しない。
