# Phase 0 完了報告: データ・マッピング基盤構築

**Date**: 2026-08-23
**Scope**: 第1版 (non-expanded, 79 サブセクター / 498 銘柄taxonomy, 時価総額加重)

## 1. 実施内容

1. **ティッカー・マッピングの整備**
   - `configs/research/subsector_mapping_generated.yaml`（494 銘柄）を使用。
   - JPX 業種コードマスタ `var/research/subsector/jpx_master/jpx_master.csv` を参照。

2. **PIT 時価総額パネル構築**
   - スクリプト: `src/research/scripts/experiments/build_subsector_market_cap.py`
   - 出力: `var/research/subsector/cap_panel.parquet`（494 銘柄、4089 日）
   - yfinance `sharesOutstanding` を最新発行済株式数として使用し、`close / adj_close` から株式分割係数を近似して過去株式数を遡及。

3. **集約行列 A の計算（時価総額加重）**
   - スクリプト: `src/research/scripts/experiments/build_subsector_aggregation.py`
   - 出力: `configs/research/subsector_aggregation_vw.yaml`
   - 形状: (17 セクター ETF) x (79 サブセクター)
   - 各行和 = 1.0（全セクターがカバーされ、凸性を満たす）

4. **カバレッジ計測**
   - `count_coverage`（taxonomy カバー銘柄数 / JPX 17 業種銘柄数）を JPX マスタから計算。
   - `cap_coverage` は `covered_cap / covered_cap within sector` として placeholder。今後 expanded cap パネルまたは JPX マスタ全銘柄 cap で精緻化予定。

5. **品質検証レポート生成**
   - スクリプト: `src/research/scripts/experiments/validate_subsector_quality.py`
   - 出力:
     - `reports/subsector_refinement/phase0_data/quality_report_vw.md`
     - `var/research/subsector/quality_summary_vw.json`

## 2. Phase 0 ゲート評価

| Gate | 内容 | 結果 | 判定 |
|---|---|---|---|
| (1) | canonical mapping の未マップ数 = 0 | 4 銘柄失敗（4530.T, 6201.T, 6406.T, 9719.T） | 注意: 残り 494 / 498 はマップ済み |
| (2) | 17 セクター全てに ≥1 サブセクター | True | PASS |
| (3) | バスケット vs ETF 相関 ρ ≥ 0.7 | min=0.2653, median=0.5122, 74/79 が未満 | **FAIL** |
| (4) | セクターカバレッジ計測・記録 | True | PASS |
| (5) | ベースライン (2010-2014) に ≥126 有効日のサブセクター数 | 78 | PASS (need ≥60) |

ゲート (3) は未達。計画に従い、未満の 74 サブセクターを `quality_report_vw.md` に記録。

## 3. 主な発見

- 時価総額加重 A 行列は全 17 セクターをカバーし、凸性を満たす。
- サブセクターバスケットと TOPIX-17 ETF の相関は全体的に低く（median 0.51, min 0.27）、79 サブセクター中 74 個が閾値 0.7 を下回る。
- これは H0「サブセクター細分化は単なる次元増大」に支持されるが、計画に従い Phase 1 で情報係数（IC）を直接検証する。

## 4. Phase 1 への前提

- ゲート (3) 未達を「記録」として扱い、計画通り Phase 1 へ進む。
- Phase 1 では `panel_subsector_oc_vw.parquet`（79 サブセクター open-to-close）および A 行列を用いて、subsector BLPX 信号を構築し、direct 17 次元 BLPX と IC を比較する。
