# サブセクター細分化リファインメント：最終報告

**Date**: 2026-08-23
**Scope**: TOPIX-17 セクターETF → 79 サブセクターへの予測空間細分化の検証
**Approach**: 第1版（non-expanded, 79 サブセクター / 494 銘柄, 時価総額加重）

## 結論

**H0「サブセクター細分化は単なる次元増大で予測精度を改善しない」が支持されたため、本戦略への本番統合は見送る。**

Phase 0-2 まで実施し、データ基盤・集約行列・BLPX 信号・IC 評価を構築した結果、サブセクター BLPX の集約信号は direct 17 次元 BLPX と比較して **大幅に劣化し、符号も反転** した。

## Phase 0：データ・マッピング基盤構築

### 実施内容
- `configs/research/subsector_mapping_generated.yaml`（494 銘柄）を canonical mapping として使用。
- `src/research/scripts/experiments/build_subsector_market_cap.py` で PIT 時価総額パネルを構築。
- `src/research/scripts/experiments/build_subsector_aggregation.py` で 17×79 の時価総額加重集約行列 A を構築。
- `configs/research/subsector_aggregation_vw.yaml` に A 行列・coverage を保存。

### Phase 0 ゲート
| Gate | 内容 | 結果 | 判定 |
|---|---|---|---|
| (1) | canonical mapping の未マップ数 = 0 | 4 銘柄失敗（4530.T, 6201.T, 6406.T, 9719.T）、残り 494/498 マップ済 | 注意 |
| (2) | 17 セクター全てに ≥1 サブセクター | True | PASS |
| (3) | バスケット vs ETF 相関 ρ ≥ 0.7 | min=0.27, median=0.51, 74/79 未満（1620 拡張でも min=0.36, median=0.54, 74/79 未満） | **FAIL** |
| (4) | セクターカバレッジ計測・記録 | True | PASS |
| (5) | ベースライン (2010-2014) に ≥126 有効日のサブセクター数 | 78 | PASS |

ゲート (3) の未達は、サブセクター細分化がノイズを増大させる可能性を示唆。計画に従い未満 74 サブセクターを `reports/subsector_refinement/phase0_data/quality_report_vw.md` に記録。

## Phase 1：サブセクターパネル構築と品質検証

- `var/research/subsector/panel_subsector_oc_vw.parquet`（79 サブセクター open-to-close）を使用。
- 79 サブセクターバスケット vs 対応 TOPIX-17 ETF の全期間相関を計算。
- 全体的に低相関（median 0.51, min 0.27）。品質ゲート (3) は未達。

## Phase 2：BLPX 推定モデル構築と統合

### 実施内容
- `src/research/scripts/experiments/build_subsector_df_exec.py` で 79 次元 `df_exec` を構築。
- `src/research/scripts/experiments/run_subsector_ic.py` で `ProductionBLPXModel` を 79 次元（n_j=79）で実行。
- 79 次元 residual BLPX 信号を集約行列 A（17×79）で 17 次元に集約。
- Direct 17 次元 residual BLPX と IC を比較。

### IC 評価結果
| Metric | Aggregated Subsector BLPX | Direct 17-dim BLPX |
|---|---|---|
| mean rank IC (per-ETF) | **-0.025** | +0.159 |
| median rank IC (per-ETF) | **-0.020** | +0.162 |
| mean daily IC (cross-sectional) | **-0.070** | n/a |

### Appendix: 1620-stock expanded mapping

After the initial rejection, the mapping was extended from 494 to 1620 TOPIX constituent stocks (`configs/research/subsector_mapping_expanded.yaml`) and the full Phase 0-2 pipeline was re-run.

| Metric | Aggregated 1620-Stock BLPX | Direct 17-dim BLPX |
|---|---|---|
| mean rank IC (per-ETF) | **-0.096** | +0.159 |
| median rank IC (per-ETF) | **-0.093** | +0.162 |
| mean daily IC (cross-sectional) | **-0.058** | n/a |

The expanded universe made the result substantially worse: all 17 ETFs now show negative per-ETF IC, with the worst at 1618.T (-0.163), 1621.T (-0.160), and 1623.T (-0.138).

全 17 ETF で aggregated subsector BLPX の IC は direct 17 次元より劣化。特に 1626.T (-0.068), 1631.T (-0.058), 1633.T (-0.044) で大きく悪化。唯一 1630.T のみ +0.018 だったが、依然として direct (+0.154) を大きく下回る。

### 原因分析
1. **相関ゲート未達**: 74/79 サブセクターの自己バスケット vs ETF 相関が 0.7 未満。1620 銘柄拡張でも 74/79 の未達は変わらず、median は 0.51→0.54 にわずか改善したが min は 0.27→0.36 に留まり、ゲート達成には程遠い。
2. **ノイズ増大**: 79 次元への拡大により、サンプル共分散行列の推定誤差が増大。1620 銘柄版ではデータ品質のばらつき（薄商い・欠損 gap）がさらに増え、BLPX 信号がさらに劣化。
3. **集約損失**: A 行列は市場重心を正しく捉えるが、サブセクター信号のノイズも同時に集約される。銘柄数を増やしても ETF ターゲットに対する予測力は回復しない。

## Phase 3：バックテスト・感度分析

**Phase 2 の IC が負であり、direct 17 次元を統計的に有意に下回ることが確認されたため、Phase 3 の本番バックテストは追加価値がないと判断し実施しない。**

H0 が強く支持されるため、バックテストによるさらなる反証は不要。

## 不採用決定と記録

- `docs/experiment_graveyard.md` に本実験を不採用として記録。
- Phase 0-2 の詳細レポートは `reports/subsector_refinement/` 配下に保存。
- 全てのスクリプトは `src/research/scripts/experiments/` 配下に残す（リファレンス用）。

## 教訓
- サブセクター細分化は「理論上の粒度向上」に対して、実データではバスケット vs ETF 相関が低く、次元増大によるノイズが BLPX 信号を劣化させる。
- **1620 銘柄へのマッピング拡張は、品質ゲートをほぼ改善せず、BLPX IC をさらに悪化させた**。より多くの銘柄を含めることはノイズ削減にはならず、薄商い銘柄や欠損データの増大で予測対象の質を下げる。
- 今後同様の細分化を試みる場合、まずバスケット vs ETF 相関 ≥ 0.7 を達成する定義の再設計が必須。それが達成できない限り、本戦略は現在の 17 次元 TOPIX-17 ETF 予測空間を維持すべきである。
