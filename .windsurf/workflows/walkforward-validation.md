---
description: ウォークフォワード検証でOOS性能を確認し、過学習を防ぐ
---

# ウォークフォワード検証

モデル変更・パラメータ調整後の out-of-sample 性能確認。

## 手順

1. ウォークフォワード検証スクリプトを実行（先例: `reports/phase3_walkforward_validation_report.md` を参照）:

```
python3 src/research/scripts/backtest/run_production_backtest.py --config configs/production/production.yaml --start-date 2015-01-05 --train-end-date 2019-12-31 --oos-start-date 2020-01-01
```

2. 確認項目:
   - IS（In-Sample）と OOS（Out-of-Sample）の Sharpe 比較
   - OOS での最大DD・ターンオーバー
   - フォールバック発動率の安定性

3. 過学習ガード（必須）:
   - 新パラメータ追加時は **パラメータ±摂動の感度分析** を実施
   - **Deflated Sharpe**（試行回数補正）をレポートに含める
   - `archive/experiments/` 約30本の過去実験との重複に注意

4. 結果を `reports/<sprint名>/` に markdown で記録

## 注意事項

- このリポジトリには過去の実験config・スクリプトが大量にあり、同一ヒストリー上での反復選択が既に多い
- 「Sharpe改善なし」の結論も価値がある — 不採用実験も必ずレポート化して二重検証を防ぐ
- 不採用実験の記録は SKILL.md の「不採用実験の記録」セクションに追記すること
