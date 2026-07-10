---
name: leadlag-fund-improvement
description: 日米セクター・リードラグ市場中立ファンド（Residual-BLPX-RA v2）の改善作業を行う際のガイド。モデル改善・バックテスト・検証・本番昇格のワークフロー、データ整合規約、リーク防止の不変条件、既知の落とし穴を記載。モデル変更・パラメータ調整・新シグナル追加・バックテスト実行時に必ず参照すること。
---

# 日米リードラグ・ファンド改善スキル

> **前提**: 戦略概要・データ整合規約・不変条件・改善ワークフロー・既知の落とし穴・評価指標の約束事は `AGENTS.md`（常にコンテキストにロード済み）を参照。本スキルはAGENTS.mdに記載のない追加情報のみを記載する。

## 2026-07追加機能（AGENTS.md未記載）

- **Copula相関ブレンド**: t-copulaで尾部依存を捕捉しPearson相関と動的ブレンド。ストレス期のみcopula重みを増加。`src/leadlag/core/correlation.py::estimate_t_copula`
- **MinVar weight最適化**: 予測共分散`Omega_gap`でロング/ショート各バスケット内の分散を最小化。シグナル比例weightとαブレンド（α=0.8が最適）。`src/leadlag/core/signal.py::build_weights_minvar`
- **Macro Confidence (Factor-Specific Kappa)**: USDJPY・原油・10年債利回りのサプライズでシグナル強度をスケーリング。`src/leadlag/core/macro.py`
- **実験用config**: `production_v2_primary_ruleD.yaml` にcopula・minvar・macro confidence設定を追加済み

## V2データパイプライン構造

V2日次実行に必要な前提データチェーン:

```
Step 1: distribution_diagnostics (omega_struct行列)
  スクリプト: archive/tools/compute_structured_prediction_covariance.py
  入力: df_exec, V1バックテストウェイト (daily_positions_Residual-BLPX_only.csv)
  出力: live/pipeline_data/distribution_diagnostics/<timestamp>/matrices/omega_struct_*.npy
  ※Copula有効時: 相関行列推定にt-copulaブレンドを使用（correlation.py::estimate_t_copula）
  ※Macro有効時: マクロサプライズスケールをシグナルに適用（macro.py::compute_factor_kappa_scale）

Step 2: gap_adjusted_distribution (mu_gap, Omega_gap行列)
  スクリプト: tools/production/compute_gap_adjusted_distribution.py
  入力: Step1のomega_struct, distribution_validation, vol_state_panel, V1ウェイト
  出力: live/pipeline_data/gap_adjusted_distribution/<timestamp>/matrices/mu_gap_*.npy
  日次バッチ: scripts/batch/run_gap_distribution.sh (6:30 JST)

V2決定: run_decision_v2.sh (9:05 JST)
  gap行列 → mu_over_sigmaスコア → MinVar weight構築(Omega_gap使用) → RuleDグロス調整 → ウェイト生成
  ※MinVar有効時: build_weights_minvarがOmega_gapで分散最小化（α=0.8でシグナル比例とブレンド）
```

### 運用データ保護（2026-07新設）

- **`live/pipeline_data/`**: パイプライン前提データの正本（Step1出力・gap分布・V1バックテスト・キャッシュ）
- **`results/`**: 実験・バックテスト結果（クリーンアップ対象、運用データではない）
- `.gitignore` で `!live/pipeline_data/` 例外を設定し保護
- **パス参照**: `compute_gap_adjusted_distribution.py`, `run_gap_distribution.sh`, `run_decision_v2.sh`, `production.yaml` のデフォルトパスは全て `live/pipeline_data/` 配下を参照済み

## 採用実験の記録（本番統合済み）

- **Copula相関ブレンド**（2026-07）: t-copulaで尾部依存を捕捉しPearson相関と動的ブレンド。単体ではSharpe +0.012（非有意）だが、MinVarと組み合わせで相乗効果あり。`correlation.py::estimate_t_copula`、`_t_copula_neg_loglik`（vector化済み）
- **MinVar weight最適化**（2026-07）: 予測共分散`Omega_gap`でバスケット内分散最小化。α=0.8でsignal比例とブレンド。Sharpe +0.098（paired t-test p<0.0001）、copula+minvarで+0.115。計算コストほぼゼロ（4s）。`signal.py::build_weights_minvar`
- **Macro Confidence**（2026-07）: USDJPY・原油・10年債利回りのサプライズでシグナルスケーリング。`macro.py::compute_factor_kappa_scale`。kappas=[3.0, 0.5, 0.5]
