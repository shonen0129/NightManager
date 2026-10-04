# 正規APIへの統一と未使用実装の撤去

- Date: 2026-10-04
- Status: accepted

## 判断

開発中のため後方互換層を維持しない。リポジトリ内の利用者とテストを同時に正規APIへ移し、
旧名の再公開、廃止APIの案内wrapper、未使用の実装を撤去する。
2026-09-15の[構造改善ADR](2026-09-15-structural-improvement-boundaries.md)にある
旧公開引数・ML再公開・分布booleanを残す移行判断は、本判断で更新する。

- 市場入力を使うV2計算は `ProductionV2Model.decide(inputs: DecisionInputs)`。
  日付、価格、履歴、cache参照、cache利用設定の正本は入力契約に置く。
  `source`は来歴ラベルであり、暗黙I/Oの許可を与えない。
- 保存済み分布だけの再生は `decide_from_cache(trade_date, gap_input_dir)`。
  市場履歴がないためon-demandやML overlayは実行しない。欠損・不採用ならflatとする。
  本番CLI・BTは型付き入力経路を使い、cache→on-demand→flatの規則を維持する。
- 分布状態は `DistributionResult.status` と `reason` で明示する。
  状態booleanやその推論を持たず、alert文字列から監査失敗を判定しない。
- ML fitted containerは `models/ml_order_overlay.py`、特徴量・artifact・推論は各module、
  学習・学習定数はresearchに置く。正本のclassを別名へ移す必要はない。
  学習入口は `tools/research/train_ml_order_overlay.py`。
- writerは `PortfolioDecision` を直接受ける。ML倍率のsummaryは
  `relative_allocation_multiplier_*`とし、`p_trade_*`の別名を出さない。
- BLP基底は `BLPModelBase`、設定型は `config.schemas`、銘柄次元は
  `N_US` / `N_JP` / `N_TOTAL`を直接使う。V2の費用解決は `v2.costs`を正本とする。
- 市場データパスは `var/market_data` に固定する。旧rootへの自動切替やデータ移動は行わない。
- 推論・CIのLightGBM依存は `ml-overlay` extra。学習は `research` extra。
  重複した `nonlinear` / `calendar` / `ci-ml` extraは持たない。

- リスク閾値は `AppConfig.risk` / `RiskConfig` を正本にし、実行経路へ直接渡す。
  `StrategyConfig`との同期コピーを廃止し、プログラムで指定した閾値も確実に反映する。
- ML設定の実行時正本は `AppConfig.v2`。YAMLのsectionは入口で一度だけ解決し、
  `AppConfig`に未使用の重複sectionを持たない。
- 市場cacheの読書きは `market_data_cache.py`のみ。二重書込と旧ストアへのfallbackを廃止する。
  `SqliteCacheStore`はJSON envelope / Parquetのみを扱い、raw pickleを読まない。
- gap読出しは `GapStore.load_horizon_bundle`で行列・metadata・manifestを同時に取得する。
  manifestは現行の完全な契約を要求し、欠けたkeyや旧版を補完しない。
- 相関計算のcacheは呼出側が明示して所有する。module globalの暗黙cacheを持たない。
  PIT全履歴の読出しは `history_frame()`に統一する。
- PITのIR履歴は現行の `pred_ir_gap_baseline_cost`のみを使う。異なる費用式の列へ
  自動切替せず、欠損時は設定済みの履歴不足処理へ進む。必須のpandas 3 / PyArrowに
  合わせ、未対応の旧version・保存形式の分岐を持たない。

## 撤去対象と維持する境界

呼び出しがない旧low-rankモデル、V1型、dispersion/gap-tolerance関数、SessionState、
接続されていない板執行ルール、旧gap読込中継、診断shim、無効なauto-close CLI引数を撤去する。
研究shadow生成で撤去済みmodule属性をpatchする処理と、機能していないPIT履歴制限optionも撤去する。

SQLiteのtransaction、監査、入力来歴、期限、注文の照合、実データの欠損処理は責務を持つため維持する。
NPYは固定回帰bundleと研究入出力で実際に使われる保存形式であり、別形式への一括移行は行わない。
研究成果物と過去のADR・実験記録は証拠として保持する。本判断は戦略パラメータの改善実験や
本番運用受入の完了を意味しない。

検証結果と確認範囲は [精査報告](../../reports/20261004_kiss_cleanup/report.md) に記録する。
