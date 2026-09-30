# ML overlay 再学習・候補評価

作成日: 2026-09-27 JST  
判定: **2026-09-27、利用者の明示指示により本番へ昇格。事前設定した数値ゲートは未達のままoverrideした。**

## データ再構築

厳格な全体再構築は、JPX公式日報でもAM quoteが確認できない2009-01-13の1619.T・1626.Tで始値が得られないため停止する。該当セルを推測で補わず、公式日報でAM約定値を照合できた2009年43件だけを復元し、AM quoteのない10件と価格尺度が一致しない3件を含む計13セル・8取引日を除外した。これらはすべて学習開始前である。raw SQLiteは変更していない。採用行は非strict構築後に`validate_exec_record()`で全行を検査し、失敗0件だった。

2025-10-24はJPX公式日報のAM始値・PM終値で16 ETFの32価格セルを復元した。各ETFの公式前日終値（公式終値−前日比）がrawの10月23日終値に一致することを照合した。1629.Tは公式系列とローカル系列が500倍異なるため変更していない。公式資料は[JPX 2025-10-24日報PDF](https://www.jpx.co.jp/markets/statistics-equities/daily/um3qrc0000028a0i-att/stq_20251024.pdf)で、ローカル保存版のSHA-256は`fa1bd91f82f10772eaa6cc2b563f53f0fe4d32fa1ccb748def7dafe25b2667ec`。

復元により2025年の欠落した3 trade dateが戻り、raw非strict結果から派生したpost-2015の896 betaセルだけが変化した。対象範囲外の変化は0。学習期間内ではbeta欠損によるラベルmask 0件、欠損ラベル0件、全17銘柄のcoverageは100%。2026-09-28 provisional行は除外し、最新確定ラベルは2026-09-25。

- raw SQLite SHA-256: `b08510b61a1c0ab64cd4e8a3cfc4ef23e6b0fe3a85b07a9c6b8f3d9ca210ef5b`
- corrected `df_exec`: 4,192行、2009-01-07〜2026-09-25
- `df_exec` fingerprint: `ad47b9326a1f0c79b21ac10f254d622c4a0c930b4eea880f0f7cd69474c3555c`
- 学習期間: 2015-01-05〜2026-09-25、2,784日、47,328行
- macro: Yahoo Finance/yfinanceで取得、最終価格日2026-09-25
- gap/rank/PIT: rank行8、PIT警告0。全日on-demandとfile cacheの使用比率は実行経路で計測していないため未取得。
- 過去データのprovider `available_at`は未証明。取得時刻の創作はしていない。

詳細: [rebuild audit](rebuild_audit.json)、[source reconciliation](reconciliation.json)、[input provenance](input_provenance.json)、[input hashes](../../var/results/20260927_ml_overlay_retrain/inputs/input_provenance.json).

## 候補artifact

- Version: `20260926T182011981905Z-f8b4ec178e88`
- Root: `var/results/20260927_ml_overlay_retrain/artifact`
- Model SHA-256: `616c8ba8564ece95398788450cdc50df0ce3f189c9eaebd53dd6142fd42fca8b`
- Metadata SHA-256: `33434a5d801fb21723803ac35d6bdda49ed71648dd8877d5e32b7e369ebefd32`
- Train end: 2026-09-25; training rows: 47,328
- Target: raw; target-side round-trip cost: 10 bps; seed 42
- LightGBM: `{"n_estimators": 100, "max_depth": 3, "num_leaves": 20, "learning_rate": 0.05, "min_child_samples": 300, "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1, "verbosity": -1}`
- 候補作成時のproduction `CURRENT`: `20260922T011723828589Z-7e1a81ab2617`（旧model SHA-256 `df5bc337be0b999a84780943aaf2918fa8b775be7e4357ccd37e0600aa85a`）。2026-09-27の明示指示による昇格後は候補versionを選択。

## 収集・walk-forward診断

5日ベンチマークは5/5日、85行で成功し、全日numerical/leakage auditがPASS。全期間収集は2,784/2,784 decision日、47,328行、全ティッカー2,784行ずつ。欠落日・gap missing・audit failureはいずれも0。全2,784 decisionでnumerical auditとleakage auditがPASS。Production cost条件はslippage 5 bps、financing 2.5%年率、borrow 1.15%年率、reverse 2 bps、long/short alpha 0.75/0.50、side leverage 1.5。

以下の2020〜2024年foldは各fold前5取引日をpurgeし、固定候補1 parameterizationで学習した**既知期間の診断**。現行artifactは2024-12-20まで学習済みなので、過去foldで比較すると将来学習リークになる。したがって基準はML overlay無効のBLPXモデルであり、現行artifactとの比較でも独立OOSでもない。前向きshadow比較の代わりにはならない。

| 年 | 日数 | net Sharpe 基準→候補 (差) | 最大DD 基準→候補 (差) | 平均turnover 基準→候補 | 20日block平均net差 [95% CI] |
|---:|---:|---:|---:|---:|---:|
| 2020 | 234 | 8.284→8.397 (0.1132) | -4.84%→-5.36% (-0.52pp) | 1.4960→1.5072 | 0.000264 [0.000153, 0.000372]|
| 2021 | 237 | 6.284→6.327 (0.0433) | -5.79%→-6.12% (-0.33pp) | 1.2773→1.2894 | 0.000178 [0.000090, 0.000269]|
| 2022 | 235 | 7.971→7.914 (-0.0568) | -4.15%→-4.21% (-0.06pp) | 1.2630→1.2738 | 0.000175 [0.000102, 0.000250]|
| 2023 | 238 | 4.290→4.320 (0.0303) | -5.11%→-5.72% (-0.61pp) | 1.2540→1.2662 | 0.000119 [0.000037, 0.000210]|
| 2024 | 236 | 6.872→6.883 (0.0107) | -7.28%→-7.46% (-0.18pp) | 1.2595→1.2705 | 0.000184 [0.000066, 0.000329]|

最大DDは5 foldすべてで基準より悪化した。paired bootstrapは20取引日block、1,000反復、seed 42。区間は候補−基準の日次net return差を示し、将来改善の確率や独立した有効性の証明ではない。ExperimentRegistryの`ml_overlay_training`記録は開始前5件で、今回は同一固定パラメータのfitを2回実行した（2回目は非挙動lint修正後のsource hash更新で、最初の候補を再発行）。最終件数は7件、候補parameterizationは1つ。新しいパラメータ探索ではないためDSRは算出していない。

### コスト

下表は5 fold全日の日次コストreturn fractionの単純和。年率コストや複利収益率ではない。turnoverはBacktestEngineのraw weight単位で、平均は全評価日を含む。

| 内訳（期間日次コストの単純和） | overlay無効基準 | 候補overlay | 差 |
|---|---:|---:|---:|
| slippage | 2.536924 | 2.548278 | +0.011354 |
| financing | 0.125021 | 0.125021 | +0.000000 |
| borrow | 0.038340 | 0.038340 | +0.000000 |
| reverse | 0.243375 | 0.243375 | +0.000000 |
| 合計 | 2.943660 | 2.955014 | +0.011354 |

モデルweightの最大grossは各fold 2.0、netは数値誤差内で0。side leverage 1.5適用後は実効gross 3.0、netは0。PIT multiplierは全日で記録され、値1.0または0.75。終端fallback・全zero weight日は0。production runnerの汎用ComplianceAuditorは別途未実施。

## dry-runと本番ゲート

候補はProductionRunnerから有効overlayとして読み込み確認した。再現可能な2020-01-06のdry-runにはその時点までだけで学習したfold artifact（train end 2019-12-20）を使用し、overlay適用、numerical/leakage auditともPASS、fallbackなし。API無効、broker adapterなし、注文送信0件。

候補作成後の新しいforward labelは0日で、計画上の必要数250日に達していない。provider `available_at`の来歴も未解決で、既知期間の診断では最大DDが全foldで悪化したため、当初の昇格ゲートは不合格だった。その状態を報告した後、利用者から「本番に切り替えて」と明示指示があり、2026-09-27 03:41 JSTにmanual overrideとして昇格した。これは数値ゲートの合格を意味しない。

本番root `models/ml_order_overlay/production_20260923` の `CURRENT` は `20260926T182011981905Z-f8b4ec178e88` を指し、旧version `20260922T011723828589Z-7e1a81ab2617` はrollback用に保持している。新versionのmodel SHA-256は `616c8ba8564ece95398788450cdc50df0ce3f189c9eaebd53dd6142fd42fca8b`。本番設定ファイルのroot pathは変えていない。切替後に継承解決済みproduction configからProductionRunnerを構築し、同version・digestで読み込まれることと旧versionの存在を確認した。切替操作中のAPI発注・注文送信はなく、broker adapterも生成していない。

再現・rollback記録: [昇格判断](../../docs/decisions/2026-09-27-ml-overlay-operator-directed-promotion.md)、[promotion gate JSON](promotion_gate.json)、[promotion record](../../models/ml_order_overlay/production_20260923/promotions/20260927_operator_directed_promotion.json)。問題時はproduction rootの `CURRENT` を旧version `20260922T011723828589Z-7e1a81ab2617` へ戻し、config-resolved loaderで再検証する。

ゲート記録: [promotion gate JSON](promotion_gate.json)、[collection](collection.json)、[walk-forward](walkforward.json)、[dry-run](dry_run.json)、[candidate metadata](candidate.json)。

## 再現コマンド・検証

実行スクリプト: `src/research/scripts/experiments/retrain_ml_overlay_20260927.py`。stages: `prepare`, `benchmark`, `collect`, `walkforward`, `train`, `dry-run`。長時間stageは外側timeout付きで実行した。

- ML overlay/artifact対象テスト: 34 passed
- 全テスト: `pytest tests/ -n 4 -q`、893 passed
- compileall: `src/leadlag tests tools scripts src/research` 成功

昇格後もforward 250日比較とprovider来歴の確証は未了であり、数値ゲートは未達のまま。これらの不足は本番昇格記録で明示した。
