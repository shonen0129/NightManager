# Frobenius Norm Prior Scaling A/B Comparison Report

## 概要

仕様書 (`docs/モデル技術仕様書.md`) で規定されていた Frobenius ノルムスケーリングと、実装の raw priors（スケーリングなし）をバックテストで比較し、優位な方に統一した。

## 背景

- **仕様書**: `B_pca_scaled = B_pca × ||B_blp||_F / ||B_pca||_F` でスケール調整後、凸結合で B_struct を構成
- **実装**: Tikhonov 正則化アプローチで raw B_pca / M_sector を RHS に直接投入
- 仕様書と実装でアプローチが異なり、Frobenius スケーリングが未実装だった

## 変更内容

1. `_solve_tikhonov` に `B_blp` パラメータを追加し、`frobenius_scale_priors` フラグで Frobenius スケーリングを切り替え可能にした
2. `_solve_asymmetric_blp` にも `B_blp` を伝播
3. 比較バックテストスクリプト: `scripts/experiments/compare_frobenius_scaling.py`

## バックテスト結果

Config: `configs/production/production.yaml`, start_date=2015-01-05, 片道5bps + overnight costs

| Metric | A: Raw priors (current) | B: Frobenius scaled (spec) | Delta |
|--------|------------------------|---------------------------|-------|
| **Sharpe_net** | **8.3548** | 8.0316 | -0.3232 |
| Sharpe_gross | 10.3802 | 9.9788 | -0.4014 |
| AR_net | 141.98% | 141.74% | -0.24% |
| AR_gross | 176.36% | 176.07% | -0.29% |
| MDD | -6.89% | -8.12% | -1.23% |
| Turnover | 1.6392 | 1.6474 | +0.0083 |
| GrossExp | 2.0000 | 2.0000 | 0.0000 |

## 結論

**Raw priors（スケーリングなし）が優位**。Frobenius スケーリングは採用しない。

### 理由

- net Sharpe: 8.35 vs 8.03（-3.9%）
- MDD: -6.89% vs -8.12%（悪化）
- AR はほぼ同等だが、ボラティリティが増加

Tikhonov 定式化では事前分布は共分散構造を通じて正則化される。凸結合アプローチでは λ_pca / λ_sector の重みがノルムに依存するためスケール調整が必要だが、Tikhonov の RHS への加算では priors の自然なスケールが共分散行列のスケールと調和しており、無理なノルム正規化が逆効果となったと考えられる。

## 統一内容

- **仕様書更新**: `docs/モデル技術仕様書.md` および `docs/model_summary_for_improvement.md` を Tikhonov 定式化に合わせて更新。λ_sector も 0.40 → 0.60（実装値）に修正
- **実装**: `frobenius_scale_priors` フラグは `False`（デフォルト）に設定。将来的な再検証用にコードには残置
- **テスト**: 全403テスト合格

## 関連ファイル

- 実装: `src/leadlag/models/sector_relative_ensemble_blp_enhanced.py:503-543` (`_solve_tikhonov`)
- 比較スクリプト: `scripts/experiments/compare_frobenius_scaling.py`
- 仕様書: `docs/モデル技術仕様書.md:241-256`
- サマリー: `docs/model_summary_for_improvement.md:50-57`
