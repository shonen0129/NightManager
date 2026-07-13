---
description: ウォークフォワード検証でOOS性能を確認し、過学習を防ぐ
---

# ウォークフォワード検証

モデル変更・パラメータ調整後の out-of-sample 性能確認。

## 手順

1. 全期間バックテストを実行（`run_production_backtest.py` に IS/OOS 分割フラグは**存在しない**。分割は結果の日次リターンを事後分割して行う）:

```
python3 src/research/scripts/backtest/run_production_backtest.py --config configs/production/production.yaml --start-date 2015-01-05 --output-dir results/walkforward
```

2. IS/OOS 分割分析スクリプトを `scripts/experiments/` に作成し、日次リターンを境界日（例: 2019-12-31 / 2020-01-01）で分割して IS / OOS それぞれの指標を算出する（先例: `reports/phase3_walkforward_validation_report.md`）。パラメータをIS期間のみで決めた場合を除き、事後分割は「参考値」であり真のOOSではないことをレポートに明記すること

3. 確認項目:
   - IS（In-Sample）と OOS（Out-of-Sample）の Sharpe 比較
   - OOS での最大DD・ターンオーバー
   - フォールバック発動率の安定性

4. 過学習ガード（必須）:
   - 新パラメータ追加時は **パラメータ±摂動の感度分析** を実施
   - **Deflated Sharpe**（試行回数補正）をレポートに含める
   - `archive/experiments/` 約30本の過去実験との重複に注意

5. 結果を `reports/<sprint名>/` に markdown で記録

## 注意事項

- このリポジトリには過去の実験config・スクリプトが大量にあり、同一ヒストリー上での反復選択が既に多い
- 「Sharpe改善なし」の結論も価値がある — 不採用実験も必ずレポート化して二重検証を防ぐ
- 不採用実験の記録は SKILL.md の「不採用実験の記録」セクションに追記すること
