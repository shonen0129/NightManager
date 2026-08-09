# 不採用実験の記録

このファイルには、本戦略で検証されたが採用されなかった実験の記録を残す。
詳細は個別レポートを参照。

- **Robust PCA伝播行列**（2026-07）: B_struct を低ランク+スパース分解（L+S）で置換する方針を検証。セクター事前知識（M_sector）とPCA事前分布（B_pca）の統合が失われ、confidence weighting の inv_A_tikh も単位行列フォールバックになった結果、Sharpe -35%、IC -32%と大幅劣化。チューニングでは埋められない構造的欠陥が原因。コードは全て破棄済み。
- **V1フォールバック廃止**（2026-07）: gapデータ欠損時のV1ウェイトフォールバックを廃止しフラットポジション化。理由: (1) `production_v2_writer.py` がV2実行のたびに `v1_baseline_weights.csv` を `w_v1` で上書きする循環参照があり、V1ウェイトが新規計算されず凍結化していた (2) 一度ゼロになると永久にゼロになる（実例あり） (3) データパイプライン障害時に古いシグナルで取引するより取引を見送る方がリスク管理として健全。
- **前日gap行列フォールバック廃止**（2026-07-14）: `run_gap_distribution.sh` の前日行列コピー機能を廃止。当日のgap行列が存在しない場合、前日行列を当日日付でコピーして使用していたが、これにより誤ったポジションで発注するリスクがあった。廃止後は当日行列不在時にフラットポジションを返すのが正しい挙動。
- **Fractional Differentiation採用**（2026-07-21）: US ETFリターンにLópez de Prado (2018)の分数階差分（d=0.1）を適用。ウォークフォワード検証でd=0.1がd=1.0ベースラインを12/12ウィンドウで上回る（平均Sharpe 8.87 vs 7.75）。
- **Transfer Entropy統合**（2026-07-22）: TEによる共分散行列調整を検証。6窓中5窓で最適alpha=0.0、TE-StaticはSharpe 7.02 vs Base 7.66と劣化。本番未統合。コードは実験用として保持、レポートは `reports/te_walkforward_experiment.md`。
- **非線形シュリンケージ不採用**（2026-07-24）: Ledoit & Wolf (2020)の解析的非線形シュリンケージで `regularize_correlation` Stage 1を置換する案を検証。NL-MPは平均Sharpe 8.04 vs 8.87、Δ=-0.83と劣化。本番未統合。レポートは `reports/nonlinear_shrinkage_walkforward_report.md`。
- **時変gap係数（Kalman filter）不採用**（2026-07-24）: 固定gap_open_coef/topix_beta_coefをKalman+BMA時変係数に置換。平均Sharpe 7.58 vs 8.88、Δ=-1.30と劣化。本番未統合。レポートは `reports/tv_gap_coef/tv_gap_coef_walkforward_report.md`。
- **USセクター横断面ランク入力不採用**（2026-07-29）: BLPX投影前にUSリターンを横断面ランクに変換。1/9窓のみ勝利、プールSharpe 6.45 vs 7.20と劣化。本番未統合。レポートは `reports/us_cs_rank_blpx/us_cs_rank_blpx_report.md`。
