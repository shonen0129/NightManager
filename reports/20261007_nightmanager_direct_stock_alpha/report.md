# NightManager sector alpha: direct mapping to high-liquidity stocks

2026-10-09統合時の注記: 以下は保存ウェイトを使った当時のopen→close proxy再価格の記録で、inventory-v3による現行本番PnLの再評価ではない。旧DSR値は探索履歴未確認として扱う。再現コードでは現行daily-v1入力契約とunknownの探索履歴状態を明示し、確認済みDSRを生成しない。研究の再実行・採用・本番反映は行っていない。

- 作成日: 2026-10-07（Asia/Tokyo）
- 実験コード: src/research/scripts/experiments/experiment_nightmanager_direct_stock_alpha_20261007.py
- 設定: configs/research/nightmanager_direct_stock_alpha_20261007.yaml
- 結果: var/results/20261007_nightmanager_direct_stock_alpha/
- production code/config: 未変更

## 仮説と今回の問い

既存NightManagerのTOPIX-17 sector signalを同業種内の流動性が高くbetaの安定した個別株へ写像し、ETFを直接使う場合と費用控除後PnLを比較した。株の選択・配分はETFリターンへの追随精度を最適化していない。

各個別株の予測リターンは「trade-dateのmu_gap × 前日までに推定した親ETF beta」。ロングの各sector exposureはこのbeta-equivalent exposureを維持するようにstockへ配分し、3銘柄×上限で届かない分はETFに残した。shortは貸株データがないためETFを維持した。

| Variant | 方法 |
|---|---|
| etf_baseline | 現行TOPIX-17 ETF portfolio |
| single_stock_hybrid | 各long sectorで流動性上位かつbeta安定条件を満たす1銘柄。shortはETF |
| stock3_hybrid | 同業種の上位3銘柄へbeta-equivalent exposure配分。shortはETF |
| cost_aware_stock3_hybrid | 上位K候補から最大3銘柄を列挙し、推定売買費用＋turnover penaltyを最小化。shortはETF |

前提: candidate K=10、beta 252日/安定性比較126日、beta範囲0.50〜1.50、2期間beta差0.25以内、ADV 1億円以上、個別株上限はNAVの10%、Cost-aware turnover penalty 5bp。感度は各連続値±20%、K=8/12を1因子ずつ試した。beta/流動性filterと銘柄選択は常にtrade dateより前に確定した情報に限る。

## 結果

対象期間は2020-01-06〜2026-08-05の1549日。全候補の日次PnLを同じ日付でbaselineと対にして比較した。side_leverageは現行production設定から解決した1.30を適用した。

| Variant | Gross Sharpe | Net Sharpe | Gross total return | Net total return | Gross MDD | Net MDD | One-way turnover/day | Cost bp NAV/day | Long net contribution | Short net contribution | ΔNet bp/day vs ETF | Paired block CI95 bp/day | DSR | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| etf_baseline | 10.236 | 7.813 | 8232573.92% | 536434.50% | -5.68% | -6.36% | 1.775 | 17.75 | 3.084 | 5.624 | 0.00 | [0.00, 0.00] | 1.000 | baseline |
| single_stock_hybrid | 9.239 | 6.557 | 3229355.44% | 151851.40% | -8.32% | -9.22% | 1.823 | 19.85 | 1.820 | 5.624 | -8.16 | [-10.03, -6.33] | 1.000 | not_passed |
| stock3_hybrid | 7.292 | 4.272 | 620830.32% | 15848.07% | -7.59% | -12.34% | 1.928 | 23.74 | -0.428 | 5.624 | -22.67 | [-26.66, -18.71] | 1.000 | not_passed |
| cost_aware_stock3_hybrid | 8.850 | 6.412 | 1961524.78% | 123011.41% | -6.74% | -7.81% | 1.788 | 17.97 | 1.608 | 5.624 | -9.53 | [-12.14, -6.99] | 1.000 | not_passed |
| costaware_beta202 | 8.793 | 6.375 | 1945526.98% | 122611.83% | -7.21% | -8.26% | 1.785 | 17.94 | 1.605 | 5.624 | -9.55 | [-11.99, -7.11] | 1.000 | not_passed |
| costaware_beta302 | 8.828 | 6.399 | 1989226.64% | 124401.21% | -6.76% | -7.29% | 1.790 | 17.99 | 1.620 | 5.624 | -9.46 | [-11.96, -7.02] | 1.000 | not_passed |
| costaware_stability20 | 8.808 | 6.366 | 1878677.39% | 116911.36% | -6.74% | -7.81% | 1.791 | 18.02 | 1.557 | 5.624 | -9.86 | [-12.33, -7.44] | 1.000 | not_passed |
| costaware_stability30 | 8.858 | 6.424 | 1972675.69% | 124274.38% | -6.72% | -7.78% | 1.786 | 17.94 | 1.618 | 5.624 | -9.47 | [-12.11, -6.89] | 1.000 | not_passed |
| costaware_beta_band_60_140 | 8.939 | 6.485 | 2112728.48% | 131042.44% | -6.67% | -7.81% | 1.785 | 18.05 | 1.671 | 5.624 | -9.13 | [-11.77, -6.47] | 1.000 | not_passed |
| costaware_beta_band_40_160 | 8.863 | 6.425 | 1976318.09% | 123927.14% | -7.63% | -8.69% | 1.789 | 17.98 | 1.615 | 5.624 | -9.49 | [-12.00, -6.98] | 1.000 | not_passed |
| costaware_adv80m | 8.835 | 6.398 | 1947843.03% | 121888.27% | -6.96% | -8.03% | 1.788 | 17.99 | 1.599 | 5.624 | -9.59 | [-12.15, -7.09] | 1.000 | not_passed |
| costaware_adv120m | 8.852 | 6.415 | 1967736.40% | 123386.54% | -6.74% | -7.81% | 1.788 | 17.98 | 1.611 | 5.624 | -9.51 | [-12.09, -7.10] | 1.000 | not_passed |
| costaware_k8 | 8.883 | 6.445 | 2056916.90% | 128337.93% | -7.86% | -8.92% | 1.791 | 18.01 | 1.650 | 5.624 | -9.26 | [-11.69, -6.86] | 1.000 | not_passed |
| costaware_k12 | 8.826 | 6.392 | 1894908.03% | 119364.87% | -6.67% | -7.42% | 1.785 | 17.94 | 1.577 | 5.624 | -9.73 | [-12.28, -7.26] | 1.000 | not_passed |
| costaware_cap8pct | 9.212 | 6.745 | 2506179.68% | 158475.59% | -6.40% | -7.46% | 1.784 | 17.92 | 1.859 | 5.624 | -7.91 | [-9.99, -5.85] | 1.000 | not_passed |
| costaware_cap12pct | 8.523 | 6.114 | 1572252.94% | 97726.16% | -7.12% | -8.19% | 1.791 | 18.03 | 1.380 | 5.624 | -11.00 | [-13.90, -8.06] | 1.000 | not_passed |
| costaware_turnlambda4 | 8.856 | 6.420 | 1978963.85% | 124121.69% | -6.74% | -7.81% | 1.788 | 17.97 | 1.617 | 5.624 | -9.48 | [-12.02, -7.02] | 1.000 | not_passed |
| costaware_turnlambda6 | 8.847 | 6.411 | 1962646.85% | 123034.71% | -6.74% | -7.81% | 1.787 | 17.98 | 1.608 | 5.624 | -9.53 | [-12.07, -7.01] | 1.000 | not_passed |

費用はポートフォリオの目標weight変化に片道単価を掛けて控除した。ETFは5bp/side、株は過去OHLCから推定したhalf-spread＋同じ5bp/side。取引turnoverはweight差分の0.5×L1。コスト計算は実際の変更weight全量へper-side rateを適用。欠損した選択株の当日open-close returnはゼロとして扱い、該当weightを別途出力した。

ETF baseline: net Sharpe 7.813, net MDD -6.36%, 平均turnover 1.775, 日次平均推定費用 17.75bp。

## Sector exposure・Long/Short

sector beta exposure errorは各日の「ETF weight＋stock weight×rolling beta」とNightManager target sector weightとの差。Beta-equivalent exposureは配分式で意図的に維持するが、stockの名目weight合計はbeta次第でずれるため両方を別に計測した。日次・ETF別データはsector_metrics.csvとdaily_portfolio_pnl_and_paired_differences.csv。

| Variant | ETF/sector | Net PnL Δ vs ETF (bp NAV) | ΔNet bp/day | Paired block CI95 bp/day | 判定 | Cost total (return) | Turnover/day | Mean abs beta exposure gap | Nominal sector gap | Long stock beta exposure share |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| cost_aware_stock3_hybrid | 1617.T | -800.2 | -0.52 | [-0.99, -0.01] | deterioration | 0.1722 | 0.111 | 0.000000 | 0.004639 | 29.4% |
| cost_aware_stock3_hybrid | 1618.T | -791.9 | -0.51 | [-0.80, -0.23] | deterioration | 0.1566 | 0.101 | 0.000000 | 0.004192 | 28.5% |
| cost_aware_stock3_hybrid | 1619.T | -872.3 | -0.56 | [-1.16, 0.01] | inconclusive | 0.1610 | 0.103 | 0.000000 | 0.004525 | 38.4% |
| cost_aware_stock3_hybrid | 1620.T | -577.3 | -0.37 | [-0.69, -0.06] | deterioration | 0.1667 | 0.107 | 0.000000 | 0.002500 | 33.5% |
| cost_aware_stock3_hybrid | 1621.T | -900.1 | -0.58 | [-0.87, -0.30] | deterioration | 0.1719 | 0.110 | 0.000000 | 0.004319 | 36.0% |
| cost_aware_stock3_hybrid | 1622.T | -1518.4 | -0.98 | [-1.43, -0.56] | deterioration | 0.1604 | 0.103 | 0.000000 | 0.002443 | 39.9% |
| cost_aware_stock3_hybrid | 1623.T | -961.4 | -0.62 | [-1.09, -0.15] | deterioration | 0.1401 | 0.090 | 0.000000 | 0.003424 | 44.1% |
| cost_aware_stock3_hybrid | 1624.T | -1454.7 | -0.94 | [-1.43, -0.49] | deterioration | 0.1683 | 0.107 | 0.000000 | 0.005564 | 40.1% |
| cost_aware_stock3_hybrid | 1625.T | -805.0 | -0.52 | [-1.17, 0.13] | inconclusive | 0.1879 | 0.121 | 0.000000 | 0.004298 | 44.8% |
| cost_aware_stock3_hybrid | 1626.T | -1175.0 | -0.76 | [-1.47, -0.09] | deterioration | 0.1911 | 0.123 | 0.000000 | 0.006259 | 54.0% |
| cost_aware_stock3_hybrid | 1627.T | -146.4 | -0.09 | [-0.54, 0.37] | inconclusive | 0.1668 | 0.107 | 0.000000 | 0.004073 | 33.7% |
| cost_aware_stock3_hybrid | 1628.T | -332.4 | -0.21 | [-0.61, 0.17] | inconclusive | 0.1233 | 0.079 | 0.000000 | 0.004746 | 38.9% |
| cost_aware_stock3_hybrid | 1629.T | -768.4 | -0.50 | [-0.87, -0.15] | deterioration | 0.1380 | 0.089 | 0.000000 | 0.002127 | 51.3% |
| cost_aware_stock3_hybrid | 1630.T | -1306.0 | -0.84 | [-1.29, -0.38] | deterioration | 0.1892 | 0.122 | 0.000000 | 0.004460 | 30.8% |
| cost_aware_stock3_hybrid | 1631.T | -529.0 | -0.34 | [-0.58, -0.12] | deterioration | 0.1466 | 0.094 | 0.000000 | 0.002899 | 38.1% |
| cost_aware_stock3_hybrid | 1632.T | -941.0 | -0.61 | [-1.02, -0.20] | deterioration | 0.1711 | 0.110 | 0.000000 | 0.004759 | 34.5% |
| cost_aware_stock3_hybrid | 1633.T | -887.8 | -0.57 | [-1.13, 0.05] | inconclusive | 0.1730 | 0.111 | 0.000000 | 0.004359 | 36.6% |
| etf_baseline | 1617.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1667 | 0.108 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1618.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1517 | 0.098 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1619.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1588 | 0.102 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1620.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1644 | 0.106 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1621.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1704 | 0.110 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1622.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1599 | 0.103 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1623.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1407 | 0.091 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1624.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1660 | 0.107 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1625.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1904 | 0.123 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1626.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1944 | 0.125 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1627.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1629 | 0.105 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1628.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1213 | 0.078 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1629.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1376 | 0.089 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1630.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1861 | 0.120 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1631.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1438 | 0.093 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1632.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1661 | 0.107 | 0.000000 | 0.000000 | 0.0% |
| etf_baseline | 1633.T | 0.0 | 0.00 | [0.00, 0.00] | baseline | 0.1680 | 0.108 | 0.000000 | 0.000000 | 0.0% |
| single_stock_hybrid | 1617.T | -826.0 | -0.53 | [-0.86, -0.20] | deterioration | 0.1919 | 0.113 | 0.000000 | 0.007242 | 25.3% |
| single_stock_hybrid | 1618.T | -729.6 | -0.47 | [-0.74, -0.22] | deterioration | 0.1602 | 0.102 | 0.000000 | 0.005463 | 26.7% |
| single_stock_hybrid | 1619.T | -870.5 | -0.56 | [-1.09, -0.03] | deterioration | 0.1852 | 0.109 | 0.000000 | 0.010713 | 24.4% |
| single_stock_hybrid | 1620.T | -338.9 | -0.22 | [-0.48, 0.04] | inconclusive | 0.1709 | 0.109 | 0.000000 | 0.004071 | 29.8% |
| single_stock_hybrid | 1621.T | -416.8 | -0.27 | [-0.50, -0.04] | deterioration | 0.1865 | 0.114 | 0.000000 | 0.006676 | 28.1% |
| single_stock_hybrid | 1622.T | -655.8 | -0.42 | [-0.70, -0.15] | deterioration | 0.1697 | 0.106 | 0.000000 | 0.004663 | 29.8% |
| single_stock_hybrid | 1623.T | -541.1 | -0.35 | [-0.71, -0.01] | deterioration | 0.1529 | 0.091 | 0.000000 | 0.002810 | 35.9% |
| single_stock_hybrid | 1624.T | -1346.9 | -0.87 | [-1.36, -0.39] | deterioration | 0.1990 | 0.110 | 0.000000 | 0.005286 | 30.9% |
| single_stock_hybrid | 1625.T | -1023.4 | -0.66 | [-1.23, -0.11] | deterioration | 0.2029 | 0.123 | 0.000000 | 0.006127 | 33.3% |
| single_stock_hybrid | 1626.T | -584.1 | -0.38 | [-0.78, 0.03] | inconclusive | 0.2039 | 0.125 | 0.000000 | 0.006512 | 37.4% |
| single_stock_hybrid | 1627.T | -715.6 | -0.46 | [-1.00, 0.08] | inconclusive | 0.2114 | 0.106 | 0.000000 | 0.005936 | 36.7% |
| single_stock_hybrid | 1628.T | -822.7 | -0.53 | [-0.91, -0.18] | deterioration | 0.1445 | 0.080 | 0.000000 | 0.005594 | 34.2% |
| single_stock_hybrid | 1629.T | -536.1 | -0.35 | [-0.57, -0.13] | deterioration | 0.1549 | 0.092 | 0.000000 | 0.003776 | 36.8% |
| single_stock_hybrid | 1630.T | -970.7 | -0.63 | [-0.98, -0.26] | deterioration | 0.2075 | 0.124 | 0.000000 | 0.006292 | 25.2% |
| single_stock_hybrid | 1631.T | -315.8 | -0.20 | [-0.39, 0.01] | inconclusive | 0.1583 | 0.095 | 0.000000 | 0.003705 | 33.9% |
| single_stock_hybrid | 1632.T | -810.5 | -0.52 | [-0.82, -0.23] | deterioration | 0.1780 | 0.113 | 0.000000 | 0.007902 | 28.0% |
| single_stock_hybrid | 1633.T | -1136.1 | -0.73 | [-1.07, -0.42] | deterioration | 0.1967 | 0.112 | 0.000000 | 0.006482 | 30.3% |
| stock3_hybrid | 1617.T | -2397.3 | -1.55 | [-2.18, -0.95] | deterioration | 0.2501 | 0.122 | 0.000000 | 0.019384 | 63.1% |
| stock3_hybrid | 1618.T | -2019.6 | -1.30 | [-1.83, -0.78] | deterioration | 0.1834 | 0.109 | 0.000000 | 0.014609 | 59.4% |
| stock3_hybrid | 1619.T | -3090.4 | -2.00 | [-2.75, -1.26] | deterioration | 0.2229 | 0.117 | 0.000000 | 0.022212 | 67.1% |
| stock3_hybrid | 1620.T | -1682.0 | -1.09 | [-1.66, -0.51] | deterioration | 0.2060 | 0.116 | 0.000000 | 0.013445 | 67.6% |
| stock3_hybrid | 1621.T | -1449.4 | -0.94 | [-1.36, -0.51] | deterioration | 0.2142 | 0.117 | 0.000000 | 0.008277 | 70.8% |
| stock3_hybrid | 1622.T | -1803.6 | -1.16 | [-1.79, -0.59] | deterioration | 0.1971 | 0.112 | 0.000000 | 0.012204 | 69.5% |
| stock3_hybrid | 1623.T | -1977.7 | -1.28 | [-1.94, -0.62] | deterioration | 0.1723 | 0.094 | 0.000000 | 0.005625 | 73.4% |
| stock3_hybrid | 1624.T | -3223.6 | -2.08 | [-2.93, -1.29] | deterioration | 0.2320 | 0.116 | 0.000000 | 0.013029 | 70.2% |
| stock3_hybrid | 1625.T | -2702.1 | -1.74 | [-2.68, -0.87] | deterioration | 0.2271 | 0.128 | 0.000000 | 0.010265 | 71.0% |
| stock3_hybrid | 1626.T | -1635.5 | -1.06 | [-1.80, -0.33] | deterioration | 0.2537 | 0.131 | 0.000000 | 0.009008 | 75.6% |
| stock3_hybrid | 1627.T | -1678.0 | -1.08 | [-1.78, -0.44] | deterioration | 0.2651 | 0.112 | 0.000000 | 0.009033 | 73.2% |
| stock3_hybrid | 1628.T | -2172.8 | -1.40 | [-2.20, -0.65] | deterioration | 0.1930 | 0.085 | 0.000000 | 0.011768 | 70.9% |
| stock3_hybrid | 1629.T | -1265.1 | -0.82 | [-1.31, -0.36] | deterioration | 0.1701 | 0.094 | 0.000000 | 0.006992 | 85.5% |
| stock3_hybrid | 1630.T | -2055.2 | -1.33 | [-2.15, -0.38] | deterioration | 0.2593 | 0.136 | 0.000000 | 0.020259 | 59.4% |
| stock3_hybrid | 1631.T | -802.9 | -0.52 | [-0.89, -0.15] | deterioration | 0.1729 | 0.102 | 0.000000 | 0.012430 | 74.6% |
| stock3_hybrid | 1632.T | -2315.5 | -1.49 | [-2.13, -0.89] | deterioration | 0.2101 | 0.121 | 0.000000 | 0.018828 | 71.6% |
| stock3_hybrid | 1633.T | -2848.3 | -1.84 | [-2.60, -1.06] | deterioration | 0.2490 | 0.119 | 0.000000 | 0.015770 | 71.1% |

ETF別判定数:
- single_stock_hybrid: 改善 0、悪化 13、区間が0を含む/保留 4。
- stock3_hybrid: 改善 0、悪化 17、区間が0を含む/保留 0。
- cost_aware_stock3_hybrid: 改善 0、悪化 12、区間が0を含む/保留 5。

## 年別OOS安定性・不確実性

この期間はrolling historical simulationであり、過去に閲覧された日次データを含むretrospective pseudo-OOSである。未使用forward holdoutではない。日次Net差の95%区間は20営業日circular moving-block bootstrap、5,000 resamples、seed=20261007。DSR trial familyは事前固定した4比較と14個の1因子感度条件、計18系列。DSRはportfolioの絶対Net Sharpeに対する選択補正であり、ETF比の改善を示す値ではない。未知の未登録試行を含む完全な補正でもない。

年次指標はannual_portfolio_metrics.csvに、日次対応PnLとNet差はdaily_portfolio_pnl_and_paired_differences.csvに保存した。ETF別の成績を同じ時間順で確認し、特定の17セクター一律置換は想定しない。

## Data / PIT / 実行条件

- 個別株は1620銘柄、現行分類mapping configs/research/subsector_mapping_expanded.yaml。分類metadataは{'jpx_master_file': 'var/research/subsector/jpx_master/jpx_master.csv', 'mapped_tickers': 1620, 'taxonomy_file': 'configs/research/taxonomy_expanded_topix.yaml', 'total_taxonomy_tickers': 1624, 'unmapped_tickers': 4}。
- NightManager target weights: var/results/v2_backtest_exact_production/daily_weights.csv。保存summary上のside leverageは1.50だが、この再価格実験ではactive production configの1.30を再適用した。weight生成そのものは再実行していない。
- signal source: var/live/pipeline_data/gap_adjusted_distribution/20260731_024303/matrices。mu_gap vectors were checked against var/live/pipeline_data/gap_adjusted_distribution/20260731_024303/gap_adjusted_distribution_long.csv; max absolute difference=9.997699283031958e-17。利用可能予測日1544/1549。欠損日は株式化せずETF baseline weightsを使用。realized_target_return columnは読み込んでいない。
- 個別株の09:10価格はローカルcacheにないため、株・ETF双方のprimary PnLはopen→close proxy。モデルsignal自体はNightManagerの9:10→close予測だが株側の執行時間を厳密に再現しない。
- survivorship: expanded mappingは2026年現行1620銘柄で上場廃止銘柄の全履歴を網羅していない。classification PIT: 現行親TOPIX-17を全履歴に固定し、歴史的セクター変更を再構成していない。
- cost limitation: ETF quote spreadは未取得で、ETFは5bp/sideだけ。stockは日足OHLC推定spread＋5bp/side。委託手数料、税、impact、borrow、financing、逆日歩は含めない。shortはETFであるため個別株貸株availabilityを仮定しない。
- baseline saved weightsはproduction exact artifactを再価格したもの。gross/net returnの絶対値は既存BacktestEngine保存系列の完全な再現ではなく、同じopen-close proxy上のpaired comparison。

## 前回実験・JPX33/Subsectorとの差

前回のTOPIX-17 ETF stock-proxyはETF close returnとの相関/TEを最小化するETF replication testで、全ETFを一括またはtracking quality gateで置換する設計だった。今回の目的関数にETF tracking errorはなく、元の17次元NightManager signalとportfolio weightsを保持し、銘柄へβで移す。ShortはETFへ残し、cost/turnoverを下げる候補選択を評価した。

Subsector/JPX33実験は予測空間・ラベルを17次元から79/33次元へ拡張し、細分予測の質や17-sectorへの集約を検証した。今回の予測モデル・信号次元は変更せず、既存17-sectorの執行instrumentだけを個別株とETFのhybridにする。したがって既存実験の再実行ではない。

## 判定

採用条件は、ETFより推定costが減るだけでなく、paired cost-adjusted PnL差の95%区間とDD/sector exposureを確認し、株式化した方が改善するETF/sectorに限定すること。retrospective proxy、ETF spread欠落、survivorship/PIT制約があるため、結果はresearch判断に限りproduction変更には使わない。

## 再現

1. repository rootで既存.venvを有効化。
2. 実行: timeout -k 20 1800 .venv/bin/python src/research/scripts/experiments/experiment_nightmanager_direct_stock_alpha_20261007.py
3. report: reports/20261007_nightmanager_direct_stock_alpha/report.md。tables: var/results/20261007_nightmanager_direct_stock_alpha/。
4. strategy/one-factor trial summaries are appended to var/experiments/registry.jsonl.
