# Time-Varying Gap Adjustment Coefficients via Kalman Filter

## Dangl & Halling (2012) — Walk-Forward Validation Report

**Date**: 2026-07-24
**Sprint**: tv_gap_coef
**Status**: NOT ADOPTED

---

## 1. Hypothesis

論文: Dangl & Halling (2012), "Predictive Regressions with Time-Varying Coefficients", JFE

現状の固定係数 `gap_open_coef=0.70` / `topix_beta_coef=0.60` を Kalman filter による時変係数推定に置換し、係数不安定性下での OOS パフォーマンス改善を検証する。

## 2. Configuration

- **Fixed baseline**: gap_open_coef=0.70, topix_beta_coef=0.60
- **Kalman prior mean**: [0.70, 0.60] (固定値を事前平均とするランダムウォーク)
- **Q grid (BMA)**: [1e-06, 1e-05, 1e-04, 1e-03, 1e-02]
- **R_diag (observation noise)**: 1e-3
- **Backtest start**: 2015-01-05
- **N_trials for DSR**: 55 (リポジトリ全体の過去実験数)
- **Observation model**: Y[t,j] = gap_coef * gap_vec[t,j] - beta_coef * gap_syst[t,j] + noise
  - Y = r_hat_jp_cc - r_jp_target (BLP予測誤差のgap成分への回帰)
  - 非線形 gap adjustment の一次近似 (daily returns < 2% で妥当)

## 3. BMA Model Probabilities

| Q (state noise var) | Log marginal lik | Posterior prob |
|---------------------|------------------|----------------|
| 1.0e-06             | 174887.23        | 0.0000         |
| 1.0e-05             | 174940.38        | 0.0000         |
| **1.0e-04**         | **174979.57**    | **1.0000**     |
| 1.0e-03             | 174954.33        | 0.0000         |
| 1.0e-02             | 174768.43        | 0.0000         |

BMAは Q=1e-4 に確率的に収束。中程度的な状態ノイズ分散が最適であり、係数は緩やかに変動するが完全固定でもないことを示唆。

## 4. Full-Period Metrics

| Config      | Sharpe (net) | AR (%) | Vol (%) | MDD (%) | Turnover | N days |
|-------------|-------------|--------|---------|---------|----------|--------|
| Fixed coef  | **8.6423**  | 138.15 | 15.99   | -5.97   | 1.6260   | 2741   |
| Kalman BMA  | 7.5181      | 118.74 | 15.79   | -6.27   | 1.6056   | 2741   |

Kalman は Sharpe -1.12、AR -19.4%劣化。ボラティリティは同等だが MDD も悪化。

## 5. Walk-Forward Yearly Sharpe

| Year | Fixed coef | Kalman BMA | Delta     |
|------|-----------|------------|-----------|
| 2015 | 14.16     | 11.15      | -3.01     |
| 2016 | 16.55     | 14.06      | -2.49     |
| 2017 | 10.15     | 8.74       | -1.42     |
| 2018 | 6.05      | 5.00       | -1.05     |
| 2019 | 7.43      | 6.28       | -1.15     |
| 2020 | 9.46      | 7.96       | -1.50     |
| 2021 | 6.60      | 6.85       | **+0.25** |
| 2022 | 8.35      | 6.71       | -1.64     |
| 2023 | 4.48      | 3.72       | -0.76     |
| 2024 | 7.11      | 6.31       | -0.80     |
| 2025 | 7.70      | 7.36       | -0.34     |
| 2026 | 8.51      | 6.86       | -1.66     |

## 6. Walk-Forward Consistency

- Kalman positive periods: **12/12** (全期間で正のSharpe)
- Kalman beats fixed: **1/12** periods (2021年のみ)
- Kalman mean Sharpe: 7.58
- Fixed mean Sharpe: 8.88
- Mean delta (Kalman - Fixed): **-1.30**

## 7. Deflated Sharpe Ratio

| Config      | Sharpe  | DSR    | skew  | kurt  |
|-------------|---------|--------|-------|-------|
| Fixed coef  | 8.6423  | 1.0000 | 1.204 | 8.850 |
| Kalman BMA  | 7.5181  | 1.0000 | 1.095 | 8.099 |

両者とも DSR >= 0.95 を満たすが、Kalman は Sharpe が低く統計的優位性もない。

## 8. Sensitivity Analysis (±20% perturbation of Q grid and R_diag)

| Config           | Sharpe (net) | AR (%) | MDD (%) | Turnover |
|------------------|-------------|--------|---------|----------|
| baseline_q_grid  | 7.5181      | 118.74 | -6.27   | 1.6056   |
| q_grid_x0.8      | 7.5066      | 118.40 | -6.27   | 1.6056   |
| q_grid_x1.2      | 7.5088      | 118.92 | -6.50   | 1.6050   |
| r_diag_x0.8      | 7.5174      | 119.08 | -6.53   | 1.6050   |
| r_diag_x1.2      | 7.5116      | 118.55 | -6.29   | 1.6055   |
| q_grid_x0.5      | 7.5376      | 119.93 | -7.12   | 1.6049   |
| q_grid_x2.0      | 7.5234      | 119.66 | -7.03   | 1.6050   |
| fixed_coef       | 8.6423      | 138.15 | -5.97   | 1.6260   |

- **Sensitivity range (Kalman): 0.4%** — 極めて安定。パラメータフラジリティなし。

## 9. Kalman Coefficient Statistics (post-warmup, 2015-2026)

| Coefficient       | Mean  | Std   | Min    | Max   |
|-------------------|-------|-------|--------|-------|
| gap_open_coef     | 0.246 | 0.131 | -0.030 | 0.766 |
| topix_beta_coef   | 0.239 | 0.128 | -0.027 | 0.699 |

Kalman は固定値 (0.70, 0.60) よりも大幅に低い係数を推定。gap調整の最適強度は固定値より弱いことが示唆されるが、ポートフォリオパフォーマンスとしては固定値の方が優れている。

## 10. Analysis: なぜKalmanが劣化したか

1. **目的関数のミスマッチ**: Kalman filter は予測誤差 (r_hat - r_target) の二乗誤差を最小化するが、実際のパフォーマンス指標は cross-sectional ranking -> weight construction -> net return の多段階プロセス。予測誤差の最小化がポートフォリオSharpeの最大化と同値ではない

2. **正規化の影響**: gap adjustment の後に cross-sectional normalization (zscore) が入る。係数の絶対値が変化すると normalization 後の ranking も変化し、予測誤差では捕捉できない効果が生じる

3. **固定係数の最適性**: gap_open_coef=0.70 は過去のバックテストで広範なグリッドサーチにより選択された値。Kalman の事前平均として使いつつも、データがより低い係数を好む場合でも、ポートフォリオ構築には高い係数の方が適している可能性

4. **2021年の例外**: Kalman が固定を上回った唯一の年 (6.85 vs 6.60) は、コロナ後の異常ボラティリティ環境。係数の適応的調整が特定のレジームで有効な可能性を示唆するが、全体では一貫した改善なし

## 11. Verdict

**NOT ADOPTED**: Kalman filter による時変gap係数は Sharpe -1.12 の劣化。固定係数が引き続き最適。

- Walk-forward 一貫性: 1/12 periods only (8.3%)
- DSR: 両者合格だが Kalman の Sharpe が低い
- 感度分析: パラメータフラジリティなし (0.4% range)
- 係数推定値: 固定値より低い (0.25 vs 0.70) が、ポートフォリオパフォーマンスとしては固定値が優位

### 今後の方向性

- **レジーム依存型調整**: Kalman の2021年での優位性は、ボラティリティレジームに応じた係数切り替えの可能性を示唆。ただし追加パラメータの過学習リスクに注意
- **目的関数の修正**: 予測誤差最小化ではなく、直接 Sharpe や IC を目的関数に組み込む手法（もしくは2段階最適化）の検討
- **固定係数の維持**: 現状の gap_open_coef=0.70 / topix_beta_coef=0.60 は引き続き正本とする
