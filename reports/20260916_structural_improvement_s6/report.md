# S6 損益計算・実約定台帳の責務分離 実施報告

## 判定

S6a（weight-based BT の純粋損益関数抽出）と S6b（実約定・在庫・費用の台帳接続）を、
既存の数値契約を保った状態で実装した。ローカル検証の範囲では PASS とする。

## 実装

- `src/leadlag/core/pnl.py`
  - `simulate_daily_pnl` を `BacktestEngine` から抽出。入力配列をコピーし、日中・夜間、
    持越し在庫、slippage、financing、borrow、reverse、turnover を一つの純粋計算境界にした。
  - `alpha_masks` と `calendar_days` を明示入力にできるため、研究の選択的持越しも同じ費用
    計算を利用する。標準BTの暦日スケールは維持し、研究共通処理の既存一営業日仮定は明示的
    override として保った。
  - `Fill`、`InventoryLot`、`FeeAccrual`、`RealizedPnlRecord`、`UnrealizedPnlRecord`、
    `InventoryLedger` を追加。FIFO、部分決済、反転、entry/exit feeの一回配賦、mark-to-market
    を実装した。
  - `fill_from_record` でbroker/close JSONの数量・価格・feeの別名を一度だけ正規化した。
- `src/leadlag/execution/backtester.py`
  - 旧損益ループを削除し、`core.pnl.simulate_daily_pnl`を呼ぶ互換adapterへ縮小した。
- `src/leadlag/execution/contracts.py` / `state_store.py`
  - `OrderObservation`へside、fill価格、fee、fill ID/sourceを追加し、S5 SQLite schema v2へ
    非破壊migrationした。`list_observed_fills`は累積pollを注文単位で最新化し、二重計上せず
    共通`Fill`を返す。個別fill IDがある場合はそのID単位で保持する。
  - 観測fillでfeeが欠落している場合はゼロ費用として確定せず、PnL台帳への変換を保留する。
    feeが明示されたシミュレーションfillだけはゼロを許容する。
- `src/research/backtest_common.py`
  - 選択的overnight holdingの費用ループを共通計算へ接続した。
- `src/leadlag/reporting/daily_pnl_report.py`
  - 確認済み実約定価格をそのまま `Fill` として台帳へ投入し、BT slippageを二重控除しない。
  - 実現PnLと残存建玉の未実現PnLを台帳から分離計算し、価格不足時だけ旧snapshotの明示的
    `total_unrealized_pnl`へフォールバックする。現在時刻を会計日として暗黙に使わない。
- `src/leadlag/reporting/metrics.py`
  - `MetricsSpec` を追加し、評価頻度、年率化頻度、flat日の扱い、return/cost単位を明示した。
- `src/leadlag/data/backtest_store.py`
  - `daily_pnl.overnight_return`を保存・復元し、日中/夜間帰属をSQLiteで失わないようにした。
  - 既存DBには起動時に不足列を追加する非破壊migrationを行う。
- `domain.portfolio.CostBreakdown`、`execution.cost_calculator.CostBreakdown` に単位ラベルを追加。
  前者はdecimal return、後者はbpsであり、円建てFill feeとも混同しない。

## 実装漏れの確認

- BacktestEngine内の旧損益ループ（`_simulate_daily_pnl_legacy`）は撤去済み。
- `daily_pnl_report`は観測fillを共通台帳へ通し、entry/exit feeを一度だけ計上する。
- 研究共通処理にも同じ純粋費用計算を接続し、費用ループの二重正本を残していない。
- SQLite復元時のovernight列、flat日を含む指標定義、bps/return/currencyの単位境界を固定した。
- brokerの生約定API呼出、実口座残高との突合、scheduler登録はこのローカル変更の範囲外であり、
  実測済みとは扱っていない。S5の状態台帳に残る約定照合結果を後続運用で入力として使う境界は、
  close execution log / position snapshotの既存出力を通じて確保した。

## 検証

- 対象回帰: `44 passed`（S6台帳、BT、daily report、SQLite store、metrics、bps cost calculator、
  S5 execution state/lifecycle）
- `ruff check`: PASS
- 変更対象の `mypy`: PASS（`core.pnl`、BT/report、metrics、store、domain/cost）
- `compileall`: PASS
- `lint-imports`: 4 contracts kept, 0 broken
- `git diff --check`: PASS

- 全 `tests/`: `715 passed, 17 warnings`（10 worker、376.18秒）

リポジトリ全体のRuffは既存のresearch実験スクリプトにある64件（主にimport整理と
不要mode/unused import）で失敗する。S6で変更したファイル群のRuffと、変更対象production
コードのmypyはPASSであり、既存負債をS6の修正として混ぜていない。
