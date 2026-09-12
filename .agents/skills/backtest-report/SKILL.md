---
name: backtest-report
description: V2バックテストの実行結果・既存成果物から、コストと検証範囲を追跡できるレポートを作る。
---

# バックテストレポート

共通の評価規約と保存先はルートの `AGENTS.md` に従う。以下のパスはリポジトリルート基準。既存結果の報告依頼だけなら、再実行は必要な場合に限る。

## 入力と集計

`BacktestEngine.run_v2_backtest()` の結果と有効設定を読み、実際のキー・日付インデックスを確認する。

| 内容 | 結果キー |
|---|---|
| net / gross リターン | `daily_returns` / `daily_returns_gross` |
| 総コスト | `daily_costs` |
| コスト4内訳 | `daily_slip_costs` / `daily_financing_costs` / `daily_borrow_costs` / `daily_reverse_costs` |
| ターンオーバー / グロス | `daily_turnover` / `daily_gross_exps` |
| フォールバック・日次要約 | `daily_fallback` / `v2_summaries` |

- 主評価の Sharpe・DD・turnover はフラット日を含む同一評価営業日で算出する。空系列、NaN/Inf、ゼロ分散は明記し、都合のよい日だけを除かない。
- 年率化係数、DD の資産曲線定義、turnover の片道/往復、コストの単位と日次平均/期間合計を明記する。`gross - costs = net` と内訳合計を日次で照合する。
- `daily_fallback` は実装が集計するフラット化理由を確認して使う。on-demand 成功率、終端フラット率（gap不足 / 監査失敗）、PIT multiplier 適用率は別物。利用経路のログがなければ「未取得」とする。
- 既存テンプレートや `src/research/experiment_utils.py` の自動値を無検証で転記しない。稼働日限定の成績は補助指標と明示する。

## レポートに含める内容

1. **再現情報**: 作成日、仮説、モデル/コード版、有効 config と差分、データ版・gap 生成条件、評価期間、実行コマンド、seed、成果物パス。
2. **結果表**: net/gross Sharpe、最大DD、turnover、フォールバック率。比較実験は baseline / experiment / 差分を同じ定義で並べる。
3. **コスト表**: slippage / financing / borrow / reverse と合計。overnight 保有・暦日課金・実効レバレッジも明記する。
4. **運用・シグナル診断**: モデル/実効 exposure、フラット理由、PIT 履歴不足、必要に応じ IC と実約定との差。
5. **監査結果**: 実際の出力に従い PASSED / FAILED / FLAT / 未実施を区別し、対象と失敗項目を残す。汎用 `ComplianceAuditor` の未実施を V2 監査の PASS で代替しない。
6. **過学習評価**: 比較実験では OOS 区間別成績、試行数、推定手法と不確実性。新パラメータ追加時の感度分析と DSR は必須。詳細手順は `experiment-design`。報告のためだけに新しいパラメータスイープを追加しない。
7. **判定・限界**: 採用 / 不採用 / 保留と根拠、未取得の情報、未実施の検証、本番反映の有無。

未計測値をゼロで埋めない。統計的非有意を「効果なしの証明」、bootstrap で差が正となる割合をそのまま「真の改善確率」と呼ばない。不採用もレポートと `docs/experiment_graveyard.md` に残す。
