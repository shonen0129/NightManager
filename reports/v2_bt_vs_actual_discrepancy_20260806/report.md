# V2 バックテスト vs 実取引 ズレ原因分析レポート

> 作成日: 2026-08-06
> モデル: ProductionV2Model (Residual-BLPX-RA v2)
> Config: `configs/production/production.yaml`
> 対象バックテスト: `results/v2_backtest_20200106_20260729_live`
> 対象実取引: `results/2026*_production_decision_v2` / `results/2026*_production_close_positions`

## エグゼクティブサマリー

V2 本番設定で最長期間バックテストを実行した結果、**net Sharpe 7.66、幾何累積リターン 3,175,356%、final wealth 31,754x** という強い数値が出た。一方、実取引の代替指標（受入保証金 `ukeire_hosyoukin`）や shadow-run 実現リターン（`primary_ruleD` で Sharpe 6.18、年率 84.33%）を見ると、バックテストは実取引より大幅に楽観的になっている。

主なズレ原因は以下の 5 つ。

1. **PIT / RuleD 履歴の運用バイアス**（最重要）
2. **バックテストランナーが ML overlay を含まない**
3. **9:10 執行価格・スリッページ・コスト仮定が楽観的**
4. **実際の約定・ロット制約・資本配分の非線形性**
5. **指標（`ukeire_hosyoukin`）が純粋な P&L でない**

## 1. バックテストと実取引の数値比較

| 指標 | V2 バックテスト（2020-01-06 〜 2026-07-29） | Shadow-Run `primary_ruleD`（2020-01-06 〜 2026-06-12） | 実取引代理（2026-07-29 時点） |
|------|--------------------------------------------|------------------------------------------------------|------------------------------|
| Net Sharpe | **7.66** | **6.18** | — |
| 年率リターン（AR） | **172.22%** | **84.33%** | — |
| 幾何累積リターン | **3,175,356%** | — | — |
| 年率ボラティリティ | **22.48%** | **13.65%** | — |
| Max DD | **-6.96%** | **-5.44%** | — |
| Avg Turnover | **1.36** | **2.90** | — |
| 受入保証金 | — | — | 324,280 JPY（純 P&L ではない） |

`ukeire_hosyoukin` は証拠金残高の代理指標であり、キャッシュ + 未実現損益 + 証拠金率変動を含むため、バックテストの `equity` と直接比較できない。したがってここでは **shadow-run** を実現リターンの参照基準とする。

## 2. PIT / RuleD 履歴の運用バイアス（最重要）

### 事象

- バックテストは `live/pipeline_data/gap_adjusted_distribution/20260731_024303` を使用。diagnostics CSV は **1544 行、2020-01-06 〜 2026-07-29**。
- 本番 live 用 `latest` シンボリックリンク (`live/pipeline_data/gap_adjusted_distribution/latest`) の diagnostics CSV は **13 行、2026-07-21 〜 2026-08-06**。
- `live/production_residual_blpx/pit_binning.json`（2026-08-06）では `history_count: 12`、`fallback_flag: true`、`assigned_bin: "Medium"`。

### 影響

- `src/leadlag/core/portfolio.py::get_rolling_pit_bin` は履歴数 < `rolling_window=252` の場合 `Medium/1.00` フォールバック。
- 本番では **RuleD 動的グロス調整がほぼ機能していない**（Low 0.75x への縮小がかからない）。
- バックテストでは 1543 日の履歴があり、2026-07-29 の dry-run では `PIT bin=High, mult=1.00` となる。
- 低 IR 日に live は 2.0x グロスを維持、バックテストは 0.75x（1.5x）に縮小 → コストと DD が過大評価/過小評価のズレを生む。

### 根拠コード

- `src/leadlag/models/production_v2.py:553` で `get_rolling_pit_bin` を呼び出し。
- `src/leadlag/core/portfolio.py:130` で `len(history_valid) < rolling_window` なら `Medium` フォールバック。
- `scripts/batch/run_gap_distribution.sh:133-143` で `merge_gap_distribution_diagnostics.py` により前回 `latest` の diagnostics をマージするが、**前回 latest 自身も 12 行**なので、13 行にしかならず 252 に到達しない。

### 補足

```bash
python3 scripts/experiments/_check_pit_history.py
# live/pipeline_data/gap_adjusted_distribution/latest: 13 rows
# live/pipeline_data/gap_adjusted_distribution/20260731_024303: 1544 rows
```

## 3. バックテストランナーが ML overlay を含まない

### 事象

- `scripts/run_v2_backtest.py` は `BacktestEngine.run_v2_backtest` を呼び出し、`overlay_model`/`overlay_model_dir` を渡していない。
- `tools/production/run_daily_production_v2.py` は `generate_v2_production_portfolio_with_overlay` を使用し、`models/ml_order_overlay/phase2_8/model.pkl` をロードする。

### 影響

- バックテストは **純 V2 シグナル**；本番は **V2 + ML order overlay** で最終ウェイトが変化。
- 2026-07-29 の dry-run（本番設定）では overlay 適用後のロング/ショートが以下の通り。
  - Longs: 1620.T, 1623.T, 1624.T, 1625.T, 1627.T
  - Shorts: 1619.T, 1622.T, 1629.T, 1631.T, 1633.T
- 一方、同日 close 時点の実取引ポジション（`results/20260729_145007_production_close_positions/positions_close_20260729.json`）は異なる。
  - Longs: 1619.T(2), 1620.T(5), 1621.T(2), 1631.T(1)
  - Shorts: 1618.T(2), 1622.T(5), 1623.T(1), 1629.T(350)
  - 1619.T が LONG（overlay モデルでは SHORT）、1623.T が SHORT（overlay では LONG）、1624/1625/1627/1633 不在、1618.T 追加。

### 根拠コード

- `scripts/run_v2_backtest.py:50-58`：`BacktestEngine.run_v2_backtest` 呼び出しに overlay 引数なし。
- `tools/production/run_daily_production_v2.py:243-278`：overlay ロード・適用。
- `src/leadlag/models/ml_order_overlay.py:671`：`generate_v2_production_portfolio_with_overlay`。

## 4. 9:10 執行価格・スリッページ・コスト仮定

### 事象

- AGENTS.md 既知の落とし穴に記載： **「9:10 価格近似：5分足 09:10 バーの (High+Low)/2 を執行価格としており楽観側」**。
- `BacktestEngine.run_v2_backtest` はスリッページ 5 bps/片道を仮定。
- 2026-08-06 の `compute_gap_adjusted_distribution` ログでは `1515/1545 NaN values in fractional diff output will be filled with 0.0` と警告。

### 影響

- 市場オーダーや VWAP より `(High+Low)/2` が約定しやすい方向に偏る。
- 5 bps/片道は TOPIX-17 セクター ETF 17 銘柄 × 日次ターンバー 1.36 × 実効レバ 1.5 では、実際の市場インパクト・貸株・逆日歩を過小評価する可能性。
- fractional diff の NaN を 0 埋めすると、US リターン系列に断絶 or 誤った値が入り、信号精度が低下。

### 根拠コード

- `src/leadlag/models/sre.py::compute_jp_target_returns`（9:10→大引けリターン計算）。
- `src/leadlag/execution/backtester.py:474-491`：cost 計算（slip/financing/borrow/reverse）。
- `src/leadlag/core/pipeline.py`（fractional diff NaN 警告）。

## 5. 実取引の非線形性

### 事象

- `close_execution_log.json`（2026-07-29）では `close_results` の `fill_price` が `null`、`fill_status` が **「未約定」**、全約定が確認できない。
- 実ポジションは整数株。例えば 1629.T 空売りは 350 株（260.9 円程度）、合計額は約 9.1 万円。1 枚（100 株）単位など、 fractional weights を忠実に再現できていない。
- 実ポジションに 1618.T（SHORT）、1623.T（SHORT）が入っているが、overlay/バックテストモデルでは選択されていない。実取引では前日ポジションの持越し、部分決済、約定漏れなどが影響。

### 影響

- バックテストの daily rebalancing（小数点以下のウェイト）が実際には整数ロット制約、部分約定、スリッページで歪む。
- `overnight_alpha_long=0.75, short=0.5` も、実際の持ち越しは `close_execution_log` から `held_overnight` として部分的に追跡されるが、完全には一致しない。
- `ukeire_hosyoukin` は純粋な daily P&L ではなく、**証拠金残高 + 未実現損益 + 入出金・配当・証拠金率変動**を含む。

## 6. データ処理の偏り

### 事象

- `src/leadlag/data/preprocessor.py:264-272` は 1 ティッカーでも NaN の行をスキップする（AGENTS.md 既知の落とし穴）。
- `yfinance` ダウンロードで特定ティッカー（IJR 等）が NaN になると、該当日の全レコードが削除され、`df_exec` が途中で切断される。

### 影響

- 上記により、バックテストが利用可能な日が減り、残った日が上昇トレンドに偏る（survivorship-like bias）。
- また `preprocess_data` で `r_oc` NaN の行も 0 埋めで残すよう修正済み（2026-07 廃止案）だが、過去データの再計算が必要。

## 7. コンパウンド効果の非現実性

### 事象

- V2 バックテストは side_leverage=1.5、日次リターンを資本に対して完全再投資し `(1+r).cumprod()` で成長。
- 6.5 年で 31,754 倍（CAGR 385%）となる。

### 影響

- 実際の口座には **AUM 上限・証拠金率・市場深さ・レバレッジ上限** があるため、同じ compounding は不可能。
- `ukeire_hosyoukin` や口座残高は `daily_net_returns` のような連続的な compounding を反映しない。
- Shadow-run `primary_ruleD` は同じ対象期間で年率 84%（Sharpe 6.18）であり、**V2 バックテストの 172% AR との大きな差**のうち約 2 倍は、2026 年の短期間急騰＋バックテスト最適化効果による。

## 8. 推奨対策

### 即座対応

1. **PIT diagnostics の永続化**
   - `scripts/batch/run_gap_distribution.sh` で `merge_gap_distribution_diagnostics.py` 使用前に、フル期間の diagnostics（`live/pipeline_data/gap_adjusted_distribution/20260731_024303/portfolio_gap_distribution_diagnostics.csv` 等）を `NEW_DIAG` の初期値として組み込む。
   - または、本番用 `latest` 以外に `live/pipeline_data/gap_adjusted_distribution/history` として diagnostics を累積し、`run_daily_production_v2.py` もそちらを参照する。

2. **バックテストに overlay 統合**
   - `scripts/run_v2_backtest.py` に `--overlay-model-dir` オプションを追加し、`BacktestEngine.run_v2_backtest(..., overlay_model_dir=...)` を呼ぶ。
   - これにより本番と同じ `generate_v2_production_portfolio_with_overlay` 経由でバックテストを行う。

3. **実取引比較の指標変更**
   - `ukeire_hosyoukin` ではなく、`daily_pnl_report.py` が出力する `total_daily_pnl`（実現損益 + 未実現損益）を正味資本で割った値を使用する。
   - `close_execution_log` の `fill_price` が未取得の場合は、引け時点の `evaluation_price` または次営業日の約定価格で補正。

### 中長期対応

4. **コスト仮定の再検証**
   - 実約定ログ（`sBaiBaiTesuryo`、`sBaiBaiDaikin`）から実際の 1 往復コストを推定し、バックテストの slippage_bps 更新。
   - 9:10 価格に `(High+Low)/2` ではなく 1 分足 or VWAP を採用し、楽観的バイアスを測定。

5. **シャドー運用強化**
   - `run_v2_backtest.py` の出力に `weights` を追加保存し、`tools/validation/monitor_residual_blpx_shadow_performance.py` を V2 でも実行可能にする。
   - これにより「同じモデル weights」に対する実現リターンを逐次監視し、バックテストとの差を定量化。

6. **fractional diff NaN 対策**
   - `build_common_inputs` の NaN 埋めを 0 ではなく前日値 or 線形補間に変更するか、NaN 発生原因となる US ティッカーをデータ取得段階で修正。

## 9. 結論

V2 バックテストと実取引のズレは、**RuleD PIT 履歴の運用漏れ、ML overlay の不在、楽観的執行価格/コスト、整数ロット/証拠金指標の非線形性** が複合した結果である。最もインパクトが大きいのは **PIT diagnostics が live では 13 行しかない** 点であり、これを 252 行以上にする運用修正を行えば、バックテストと live の gross scaling が大きく近づく。ただし、実 P&L 比較には `ukeire_hosyoukin` ではなく `daily_pnl_report` の正味損益を使う必要がある。
