---
name: prod-backtest-consistency
description: 本番、バックテスト、運用文書の設定・入力・会計・監査契約を照合する。
---

# 本番・バックテストの整合確認

共通規約と不変条件は [AGENTS.md](../../../AGENTS.md) を正本とする。実行手順は [leadlag-fund-improvement](../../../.agents/skills/leadlag-fund-improvement/SKILL.md)、指標差の原因調査は [debugging](../../../.agents/skills/debugging/SKILL.md)、時系列・監査は [leak-audit](../../../.agents/skills/leak-audit/SKILL.md) に従う。短い文書更新だけで長期戦略実験を追加しない。

## 比較対象を固定する

- `configs/production/production.yaml` の `__base__` を `src/leadlag/execution/config.py::load_config_from_yaml` で解決する。旧設定とのraw YAML diffを現行有効設定の証拠にしない。
- 日次はCLI `decision` → `src/leadlag/execution/v2_bridge.py`、V2評価はCLI `backtest` → `src/leadlag/execution/backtester.py::BacktestEngine.run_v2_backtest`。共通factoryは `src/leadlag/runner/model_factory.py`、モデルは `src/leadlag/models/production_v2.py::ProductionV2Model`。
- config、model artifact version、ticker順、signal/trade date、当日gap bundle、run-owned 09:10/ADR/macro/PIT/rank-reversal snapshotのfingerprintを揃える。前日gapを改名して当日入力へ使わない。
- 2010–2014のpriorを固定し、評価開始は2015-01-05以降とする。要求/実評価/source期間は `evaluation_period` で確認する。

## 会計・機能・監査を別々に照合する

- net/gross、slippage/financing/borrow/reverse、calendar holding daysを同じ単位・期間で比較する。entry-mark-v2はcarryのprevious close→entryとentry→closeをつなぐ。旧成績との比較では会計versionを表示する。
- `daily_turnover` は実効opening+closing notional/2、`daily_target_weight_turnover` はmodel目標weight差L1/2。model grossとside leverage後のeffective grossを分ける。
- 全評価日を含める。flatの欠損labelはprice PnL0と確定できるが、active欠損は無効評価にする。指標から日を削除して成功扱いしない。
- 当日cache、on-demand、終端flat、PIT multiplier、ML applied/skipped/rejectedを区別する。ML enabledやaudit PASSだけでoverlay適用を認定しない。
- `src/leadlag/models/v2/audit_comparator.py` のnumerical/leakage失敗処理と、actual-liveのquote/account-risk/reconciliation gateを追う。model flatは口座建玉の全解消ではない。
- 通常closeは持越し率を維持する。緊急全解消を行う場合は承認範囲、専用の持越し率0設定、注文・約定・残建玉の照合が必要。診断のために `--api-enable` を追加しない。

## 実行と証跡

既存プロジェクト環境を使い、長時間実行には [hang-prevention](../../../.agents/skills/hang-prevention/SKILL.md) の全体deadlineを設定する。offlineの合成fixture・保存済み入力を優先し、入力を更新して差を隠さない。

コード変更は対象回帰後に `tests/` 全体とCI指定のcompileall/Ruff/mypy/import契約を通す。設定・指標比較は [experiment-design](../../../.agents/skills/experiment-design/SKILL.md) の事前仮説・OOS・試行数補正に従い、実行結果は [backtest-report](../../../.agents/skills/backtest-report/SKILL.md) に従って記録する。設計は `docs/decisions/`、結果は `reports/`、不採用索引は `docs/experiment_graveyard.md`。Skillへ実験成績・完了履歴を蓄積しない。

仕様の現行対応表は [モデル技術仕様書](../../../docs/モデル技術仕様書.md)、運用の持越し・停止・復旧は [日次運用手順書](../../../docs/日次運用手順書.md)。実費やactual-account損益が未照合なら、その限界と残件を明示して受入を保留する。
