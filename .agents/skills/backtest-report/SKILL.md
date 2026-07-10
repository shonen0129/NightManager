---
name: backtest-report
description: バックテスト結果をAGENTS.mdの約束事に沿った標準markdownレポートとして生成する。net Sharpe・最大DD・ターンオーバー・フォールバック発動率・コスト内訳を含む。reports/<sprint名>/ 配下に既存形式で保存する。バックテスト実行後に必ず参照すること。
---

# Backtest Report スキル

## 目的

バックテスト結果を `AGENTS.md` の評価指標約束事に沿った標準フォーマットでレポート化し、`reports/<sprint名>/` に保存する。

## 評価指標の約束事（AGENTS.mdより）

- **主指標**: net Sharpe（コスト後）、最大DD、ターンオーバー、フォールバック発動率
- **gross/net 両方を報告**: コスト内訳（slippage / financing / borrow / reverse）を分解
- **「Sharpe改善なし」の結論も価値**: 不採用の実験も必ずレポート化して二重検証を防ぐ

## レポートフォーマット

```markdown
# <Sprint名> レポート

> 作成日: YYYY-MM-DD
> モデル: <モデル名>
> Config: <configパス>
> 期間: <start_date> 〜 <end_date>

## 概要

<実験の目的・仮説を1-2文で>

## 結果サマリー

| 指標 | Baseline | Experiment | 差分 |
|------|----------|------------|------|
| Net Sharpe (annualized) | x.xxx | x.xxx | ±x.xxx |
| Gross Sharpe | x.xxx | x.xxx | ±x.xxx |
| Max Drawdown | x.x% | x.x% | ±x.x% |
| Turnover (daily avg) | x.xx | x.xx | ±x.xx |
| Fallback rate | x.x% | x.x% | ±x.x% |

## コスト内訳

| 項目 | Baseline | Experiment |
|------|----------|------------|
| Slippage (bps) | x.xx | x.xx |
| Financing | x.xx | x.xx |
| Borrow | x.xx | x.xx |
| Reverse | x.xx | x.xx |
| **Total cost** | **x.xx** | **x.xx** |

## 詳細分析

### シグナル品質
- IC (rank): x.xxx
- IC decay: <あれば>

### ポートフォリオ統計
- Average gross exposure: x.xx
- Average net exposure: x.xxx
- Long/Short balance: <特徴>

### フォールバック分析
- v2 発動率: x.x%
- v1 (Residual-BLPX) フォールバック率: x.x%
- PCA-Ensemble フォールバック率: x.x%
- フォールバック要因: <理由>

## 監査結果

- ComplianceAuditor: PASS / FAIL (<失敗項目があれば列挙>)
- Leakage audit: PASS / FAIL
- Numerical audit: PASS / FAIL

## 過学習ガード

- パラメータ数: <追加した場合は感度分析結果>
- Walk-forward OOS Sharpe: <実施した場合>
- Deflated Sharpe: <試行回数補正>

## 統計的有意性検定（比較実験時必須）

baseline と experiment の日次リターンを比較し、改善が統計的に有意か検証する。

### 検定項目

1. **Paired t-test**: 日次リターンの差の有意性
   - `scipy.stats.ttest_rel(exp_returns, base_returns)`
   - p < 0.05 で有意差あり
   - 勝率（experiment > baseline の日数割合）も報告

2. **Bootstrap Sharpe差の信頼区間**: Sharpe比の差の分布をブートストラップで推定
   - 5000回リサンプリング、2.5%–97.5%パーセンタイルでCI
   - CIが0を含まなければ有意
   - `P(delta > 0)` も報告

3. **パラメータスイープ**: 複数configで一括比較
   - `scripts/experiments/experiment_copula.py --sweep` を参考
   - 比較表: Label, Sharpe, ΔSharpe, AR%, ΔAR%, MDD%, ΔMDD%, Time

### 検定結果の記載形式

```markdown
### 統計検定

| 検定 | 結果 | 判定 |
|------|------|------|
| Paired t-test | t=x.xx, p=x.xxx | 有意/非有意 |
| Win days | xxxx/xxxx (xx.x%) | — |
| Bootstrap Sharpe差 | x.xxx, CI=[x.xxx, x.xxx] | 有意/非有意 |
| P(delta>0) | xx.x% | — |
```

### 解釈の注意

- **日次リターンのt-testが有意でもSharpe差のCIが0を含む場合がある**: リターン向上と同時にボラティリティも増加している可能性
- **copula単体で非有意でもMinVarと組み合わせで相乗効果が出る場合がある**: 単独効果と組み合わせ効果を別々に評価する

## 結論

- **採用 / 不採用**: <理由>
- **次ステップ**: <あれば>

## 付録

- バックテスト実行コマンド
- Config diff (baseline vs experiment)
```

## 実行手順

1. **バックテスト実行**: `BacktestEngine.run_backtest()` または CLI で結果を取得
2. **指標抽出**: 結果 dict から net/gross Sharpe、最大DD、ターンオーバー、フォールバック率を計算
3. **コスト分解**: `daily_costs` を slippage / financing / borrow / reverse に分解
4. **監査結果確認**: `ComplianceAuditor.run_audit()` の結果を記載
5. **レポート生成**: 上記フォーマットで markdown を生成
6. **保存**: `reports/<sprint名>/` 配下に保存（既存 sprint0–3b の形式に倣う）

## 比較実験時の注意

- **config deepcopy**: `copy.deepcopy(base_cfg)` を使用し、shallow copyによる設定汚染を防ぐ
- **同一期間**: baseline と experiment で同一の `start_date` / `end_date` を使用
- **同一データ**: `df_exec` が同一であることを確認

## 不採用実験の記録

不採用の場合も以下を記録し、二重検証を防ぐ:

- 仮説・実験内容
- 結果（Sharpe変化・IC変化等）
- 不採用理由
- 再検証防止用のタグ（例: **Robust PCA伝播行列** のように AGENTS.md / SKILL.md に追記）
