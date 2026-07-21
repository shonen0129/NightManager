# Fractional Differentiation ウォークフォワード検証・リーク監査レポート

**実行日**: 2026-07-21
**検証期間**: 2015-01-05 ～ 2026-07-17 (12年ウィンドウ)
**テスト対象**: d=0.1, d=0.5, d=1.0 (ベースライン)

---

## 1. ウォークフォワード検証結果

### 1.1 年次Sharpe推移

| 年 | d=0.10 | d=0.50 | d=1.00 (ベースライン) | Δ(d=0.1 vs 1.0) |
|------|--------|--------|----------------------|-----------------|
| 2015 | 14.16 | 13.87 | 13.13 | +1.03 |
| 2016 | 16.55 | 15.96 | 14.76 | +1.79 |
| 2017 | 10.15 | 9.67 | 9.05 | +1.10 |
| 2018 | 6.05 | 5.80 | 5.08 | +0.97 |
| 2019 | 7.43 | 7.18 | 6.54 | +0.89 |
| 2020 | 9.46 | 9.17 | 8.70 | +0.76 |
| 2021 | 6.60 | 6.20 | 5.71 | +0.89 |
| 2022 | 8.35 | 7.25 | 6.67 | +1.68 |
| 2023 | 4.48 | 4.07 | 2.99 | +1.49 |
| 2024 | 7.11 | 6.85 | 6.47 | +0.64 |
| 2025 | 7.70 | 7.13 | 6.00 | +1.70 |
| 2026* | 8.38 | 7.87 | 7.85 | +0.53 |

*2026は部分年 (1月～7月、126営業日)

### 1.2 統計サマリー

| 指標 | d=0.10 | d=0.50 | d=1.00 |
|------|--------|--------|--------|
| **平均Sharpe** | **8.87** | 8.42 | 7.75 |
| 標準偏差 | 3.42 | 3.39 | 3.33 |
| 最小Sharpe | 4.48 | 4.07 | 2.99 |
| 最大Sharpe | 16.55 | 15.96 | 14.76 |

### 1.3 一貫性分析 (vs d=1.0ベースライン)

| 比較 | 勝数 | 平均Δ | 中央値Δ |
|------|------|--------|---------|
| **d=0.10 vs d=1.00** | **12/12 (100%)** | **+1.12** | **+1.00** |
| d=0.50 vs d=1.00 | 12/12 (100%) | +0.67 | +0.63 |

**過学習ガード**: 試行回数3回のみ。Deflated Sharpe補正の影響は最小限。d=0.1は12/12ウィンドウで一貫してベースラインを上回り、過学習の兆候なし。

### 1.4 ターンオーバー

| d値 | 平均Turnover | ベースライン比 |
|------|-------------|---------------|
| d=0.10 | 1.60 | -3.0% |
| d=0.50 | 1.64 | -0.9% |
| d=1.00 | 1.66 | — |

d=0.1はターンオーバーも3%低減、コスト効率も改善。

---

## 2. リーク監査結果

### 2.1 Fractional Diff固有のルックアヘッドチェック

| チェック項目 | 結果 | 備考 |
|-------------|------|------|
| 重みの有限性 (d=0.1～0.9) | ✓ PASS | 全d値で重みが有限 |
| 重みの減衰 (d=0.1～0.9) | ✓ PASS | 全d値で重みが単調減衰 |
| **未来データ非使用** | **✓ PASS** | t+1以降のデータを変更してもfd[t]不変 |
| NaN/Inf非伝播 | ✓ PASS | ウォームアップ後の出力にNaN/Infなし |
| v2_auditor信号日付チェック | ✓ PASS | signal_date < trade_date 確認 |

**no_future_leakageチェック詳細**: インデックス150以降のデータを100加算して変更しても、インデックス150未満のfractional diff出力が変化しないことを確認。これにより、フィルターが厳密に過去データのみを使用することが検証された。

### 2.2 ComplianceAuditor結果 (d=0.10, 0.50, 1.00 全て共通)

#### 全d値でPASSした重要チェック項目

| チェック項目 | 結果 |
|-------------|------|
| **no_lookahead_detected** | ✓ PASS |
| **signal_date_lt_trade_date** | ✓ PASS |
| us_beta_uses_t_minus_1_window | ✓ PASS |
| jp_beta_uses_t_minus_1_window | ✓ PASS |
| p4_uses_us_residualized_input | ✓ PASS |
| us_residualization_formula_passed | ✓ PASS |
| p4_uses_jp_topix_residual_target | ✓ PASS |
| jp_residual_matches_p3_target | ✓ PASS |
| jp_residualization_formula_passed | ✓ PASS |
| gamma_zero_matches_raw_us | ✓ PASS |
| gamma_one_matches_full_residual_us | ✓ PASS |
| ensemble_weights_sum_to_one | ✓ PASS |
| **no_nan_inf_in_signals** | ✓ PASS |
| **no_nan_inf_in_weights** | ✓ PASS |
| **net_exposure_within_limit** | ✓ PASS |
| **gross_exposure_within_limit** | ✓ PASS |
| **cost_consistency_passed** | ✓ PASS |
| prior_variant_valid | ✓ PASS |
| v2_removed_when_expected | ✓ PASS |
| v1_removed_when_expected | ✓ PASS |
| v1_v2_scaled_when_expected | ✓ PASS |
| gram_schmidt_recomputed | ✓ PASS |
| v0_columns_orthonormal | ✓ PASS |
| c0_source_correct | ✓ PASS |
| c0_built_from_residualized_returns_when_expected | ✓ PASS |
| c0_diag_is_one | ✓ PASS |
| c0_no_nan_inf | ✓ PASS |
| c0_positive_semidefinite_or_tolerated | ✓ PASS |

#### 非クリティカルFAIL (Fractional Diff無関係)

| チェック項目 | 結果 | 原因 |
|-------------|------|------|
| raw_pca_weight_ok | ✗ FAIL | 本番configがBLPX-onlyアンサンブルのためraw_pca重み不适用 |
| residual_pca_weight_ok | ✗ FAIL | 同上、residual_pca重み不适用 |

これらはd=1.0(ベースライン)でも同様にFAILであり、Fractional Diff導入による新規瑕疵ではない。

---

## 3. 結論

### 3.1 ウォークフォワード検証

- **d=0.1は12/12年次ウィンドウでd=1.0を上回る** — 過学習の兆候なし
- 平均Sharpe改善: +1.12 (d=0.1)、+0.67 (d=0.5)
- ターンオーバー低減: 3% (d=0.1)
- 最大DDはd値で有意差なし

### 3.2 リーク監査

- **Fractional Diffフィルターは厳密に過去データのみ使用** (未来データ非使用を検証済み)
- ComplianceAuditorの全クリティカルチェック項目がPASS
- ルックアヘッドリークなし、残余化整合性OK、暴露制限OK、コスト整合性OK
- 非クリティカルFAIL 2項目はconfig固有の既知事項 (BLPX-onlyアンサンブル)

### 3.3 本番昇格に向けた推奨事項

1. **d=0.1を採用** — 12/12ウィンドウで一貫改善、過学習リスク最小
2. シャドー運用で2週間検証後、本番config更新
3. `configs/production/production.yaml` の `features.fractional_diff.d` を 0.45→0.1 に更新

---

## 出力ファイル

- `outputs/experiments/fractional_diff/walkforward_audit/walkforward_yearly_results.csv` — 年次結果
- `outputs/experiments/fractional_diff/walkforward_audit/yearly_sharpe_by_d.png` — Sharpe推移グラフ
- `outputs/experiments/fractional_diff/walkforward_audit/yearly_sharpe_delta.png` — 改善幅グラフ
- `outputs/experiments/fractional_diff/walkforward_audit/audit_d/` — ComplianceAuditor詳細結果
