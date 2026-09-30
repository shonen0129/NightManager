# JPX33を個別予測してTOPIX-17へ集約する長期診断

## 判定

**33業種案は予測指標が分かれ、総合的な精度向上は確認できなかった。** 主比較ではRank ICが−0.0147（95% CI [−0.0217, −0.0081]）、方向正解率も1.5pt低下した一方、MAEは1.39bp、RMSEは1.47bp改善した。33業種・構成銘柄のPIT履歴と9:10業種価格がないため、これらは代理データによる遡及診断であり採用判断には使わない。

## 事前固定した比較

- 期間: 2015-01-06〜2026-08-21、2759日。学習・相関窓は当日を含めず、c_fullは2010〜2014年の完全行のみ。TOPIXベータは60営業日ローリングOLSで当日を除外した。
- Baseline: 現行17ETF系列・現行17銘柄ラベルで直接予測。比較対象A: 33次元で予測し現行の17銘柄感応度を各子業種へ複製。比較対象B: 上位モデル提示の33業種w3〜w6を適用。Aが次元拡張の効果を切り分ける主比較、Bは提示ラベルを使った別試行。
- 33→17写像は予測平均 μ17=W_t μ33、共分散 Ω17=W_t Ω33 W_t'。W_tは親TOPIX-17業種内で正規化した前年JPX年末業種総時価総額。各年の第1取引日は前々年末値を使い、2015年の第1取引日は2013年値がなく除外した。浮動株調整ETF保有比率ではない。
- 評価targetは17ETFの始値→大引けリターンからTOPIX始値→大引け成分をローリング除去した残差。候補33系列も同じ時間帯の個別株バスケットproxyから作成。実9:10 target・実約定・ポートフォリオ損益は未評価。
- 対象期間の17ETF基準日2779日に対し、33業種proxyの有効な最後の日は2026-08-21。業種proxy欠損で19日（うち2026-08-21以降19日）を除外し、2015-01-05は前年ウェイト取得不能で除外した。
- 指標は営業日ごとの17ETF横断Rank IC、MAE、RMSE、方向正解率。差の95%区間は対応日を保つ非循環20営業日moving-block bootstrap 5,000回。

## 予測精度

| モデル | paired日数 | 平均Rank IC | ΔRank IC vs 17 (95% CI) | MAE (bp) | ΔMAE vs 17 (95% CI, bp) | RMSE (bp) | 方向正解率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct17 | 2759 | +0.0964 | — | 64.16 | — | 84.29 | 0.532 |
| direct33 / parent labels | 2759 | +0.0817 | -0.0147 [-0.0217, -0.0081] | 62.77 | -1.39 [-1.67, -1.11] | 82.82 | 0.518 |
| direct33 / proposed labels | 2759 | +0.0823 | -0.0141 [-0.0212, -0.0073] | 62.76 | -1.40 [-1.68, -1.12] | 82.81 | 0.517 |

| 年 | 日数 | 17 Rank IC | 33親ラベル Rank IC | 差 | 17 MAE (bp) | 33親ラベル MAE (bp) | 差 (bp) | 33提案ラベル Rank IC | 差 | 33提案ラベル MAE (bp) | 差 (bp) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 235 | +0.1113 | +0.1115 | +0.0002 | 71.29 | 70.31 | -0.98 | +0.1114 | +0.0001 | 70.30 | -1.00 |
| 2016 | 237 | +0.1199 | +0.1143 | -0.0055 | 81.85 | 80.62 | -1.23 | +0.1172 | -0.0026 | 80.60 | -1.25 |
| 2017 | 244 | +0.0786 | +0.0855 | +0.0069 | 44.90 | 41.59 | -3.32 | +0.0894 | +0.0107 | 41.58 | -3.32 |
| 2018 | 251 | +0.0757 | +0.0434 | -0.0324 | 47.44 | 45.40 | -2.04 | +0.0441 | -0.0316 | 45.39 | -2.06 |
| 2019 | 233 | +0.1336 | +0.1080 | -0.0256 | 41.52 | 41.50 | -0.02 | +0.1091 | -0.0246 | 41.50 | -0.02 |
| 2020 | 234 | +0.1066 | +0.0833 | -0.0233 | 76.25 | 74.39 | -1.85 | +0.0831 | -0.0235 | 74.37 | -1.88 |
| 2021 | 237 | +0.0788 | +0.0626 | -0.0162 | 59.59 | 58.59 | -1.00 | +0.0648 | -0.0140 | 58.58 | -1.01 |
| 2022 | 235 | +0.1043 | +0.0751 | -0.0292 | 64.52 | 63.59 | -0.92 | +0.0713 | -0.0330 | 63.58 | -0.93 |
| 2023 | 238 | +0.0593 | +0.0563 | -0.0030 | 54.73 | 53.60 | -1.13 | +0.0571 | -0.0022 | 53.59 | -1.14 |
| 2024 | 236 | +0.1025 | +0.0820 | -0.0206 | 68.93 | 68.03 | -0.90 | +0.0822 | -0.0204 | 68.03 | -0.90 |
| 2025 | 230 | +0.1132 | +0.0805 | -0.0326 | 70.36 | 69.35 | -1.01 | +0.0801 | -0.0330 | 69.36 | -1.01 |
| 2026 | 149 | +0.0648 | +0.0799 | +0.0151 | 105.52 | 103.02 | -2.50 | +0.0802 | +0.0154 | 102.99 | -2.53 |
## データ品質と写像確認

- 33業種basketは現在分類の494銘柄を使用。業種ごとの選定銘柄数は1〜52。2025年末のJPX業種総時価総額に対する選定銘柄capカバー率は中央値85.5%、最低54.2%。
- 年次capウェイトで集約した33業種proxyと実ETF open-to-closeの相関は中央値0.689、最低0.537。17 ETF中9本が0.70未満。対象別は `mapped_proxy_quality.csv`。
- 分類表は2026-07-31の1時点のみ。上場廃止銘柄・過去のTOPIX採用入替・過去の業種変更を含むPITメンバー履歴ではない。選定494銘柄の過去リターンはサバイバーシップを含む。株数は最新時点から逆算した分割調整近似で、増資・自社株買いを反映しない。
- Ωの17次元写像は両33候補で全2759日を計算。PSD違反数は親ラベル=0, 提示ラベル=0。最小固有値はそれぞれ7.638e-06, 7.630e-06。

## 解釈・採否

探索的精度ゲートはRank IC差の95%区間下限>0かつ平均MAE差≤0と定義した。過去期間はこの作業以前にも閲覧済みで、ラベル案も事前に提示されたため、基準を満たしても未使用OOSや本番採用の証拠にはならない。加えて、入力バスケットのPIT分類・完全な構成銘柄・float-adjusted W_t・9:10価格は揃っていない。今回の結果だけでproductionモデルは変更しない。
33業種提案ラベル版と17親ラベル複製版の差は、感応度値と33次元 prior subspace の両方を変える。主比較のparent-label版が粒度変更の中心的診断だが、33業種株バスケット自体がPIT再現でないため、正式な粒度仮説の結論にはならない。
コスト、turnover、最大DD、V2ポジション制約の損益効果、Deflated Sharpeは未評価。ここで報告するのは予測精度診断であり、戦略バックテストではない。

## 再現情報

- 設定: `configs/research/jpx33_direct_prediction_2015_2026.yaml`。スクリプト: `src/research/scripts/experiments/experiment_jpx33_direct_prediction_20260924.py`。成果物: `var/results/20260924_jpx33_direct_prediction/`。
- df_exec fingerprint: `edff587e754a20af73fedea51a500e4a9275306904b5e54b76422c56c6ed43cb`。33業種proxy fingerprint: `da7db68604bf354e23df1cc7921b6749e8bda523aac57158c2a108c9d671f397`。実験設定SHA-256: `904984d75f0576b7ec6836c3d339daffae05eb594222fc909785ac6ec8ec3df7`。
- production BLPX parameters: `{"alpha_xx": 0.2, "alpha_yx": 0.15, "alpha_yy": 0.5, "asymmetry_delta": 0.3, "asymmetry_mode": "scalar", "asymmetry_post_gap_delta": 0.0, "asymmetry_post_gap_mode": "signal_split", "beta_conf": 0.25, "beta_floor": 0.0, "beta_window": 60, "blp_ewma_halflife": 120.0, "blp_window": 504, "copula_blend_weight": 1.0, "copula_dynamic_blend": true, "copula_enabled": true, "copula_marginal_method": "empirical", "copula_nu_init": 5.0, "copula_stress_threshold": 1.5, "corr_min_periods": 60, "corr_window": 60, "costs": null, "ewma_halflife": 120, "exec_adjustment": null, "execution_target_cost_adjustment": "none", "frac_diff_d": 0.1, "frac_diff_enabled": true, "frac_diff_normalize": null, "frac_diff_threshold": 1e-05, "frac_diff_window": 100, "frobenius_scale_priors": false, "gap_open_coef": 0.7, "gap_open_coef_neg": 0.6, "include_v4_prior": true, "k": 6, "lambda_lw": 0.5, "lambda_pca": 0.1, "lambda_reg": 0.75, "lambda_sector": 0.6, "lw_target": "equicorrelation", "macro_confidence_enabled": true, "macro_direction_enabled": true, "macro_kappa_enabled": true, "macro_kappas": [3.0, 0.5, 0.5], "macro_sigma_yy_inflation_enabled": false, "macro_surprise_halflife_mean": 20.0, "macro_surprise_halflife_vol": 60.0, "min_raw_weight": 0.0, "minvar_alpha": 0.5, "minvar_enabled": false, "model_name": "ProductionBLPXModel", "n_j": 17, "n_u": 15, "normalization": "zscore", "p4_weight": 0.0, "p5_weight": null, "p5p3_weight": null, "param_set": "phase1d_optimized_residual_blpx_production", "prior_variant": null, "q": 0.3, "rank": "full", "raw_blpx_weight": 0.0, "raw_pca_weight": 0.0, "residual_blpx_weight": 1.0, "residual_pca_weight": 0.0, "rho": 0.01, "sector_eta": 0.5, "sector_gamma": 4.0, "signal_components": null, "slippage_bps": 5.0, "target": "topix_residual", "topix_beta_coef": 0.6, "topix_beta_coef_neg": 0.6, "us_res_beta_window": 252, "us_res_enabled": false, "us_res_gamma": 0.5, "use_raw_target": false, "vol_adjusted_target": false, "weight_mode": "signal", "winsor_sigma": 3.0}`。本番コード・設定は変更していない。
- 関連する既登録JPX33/サブセクター試行: 9件。今回の候補比較は2件。主な既存参考は `reports/20260924_sensitivity_pipeline_audit/report.md` および `reports/subsector_refinement/phase2_blpx/subsector_ic_report_expanded.md`。
- 再実行: `timeout -k 20 14400 .venv/bin/python -u src/research/scripts/experiments/experiment_jpx33_direct_prediction_20260924.py`。
