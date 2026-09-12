# 指標比較・シグナルトレース

指標の急変、本番とバックテストの乖離、特定日のウェイトを調べるときに読む。パスはリポジトリルート基準。

## 指標・本番との比較

- 再現には異常発生時のコード版・継承解決後の設定・入力データを使う。現在の本番設定へ差し替える場合は別比較として記録する。設定は `safe_config_copy` または dict の `copy.deepcopy` で分離し、両系統が同じ設定やキャッシュを共有していないか確認する。
- V2 の正本は `src/leadlag/execution/backtester.py::BacktestEngine.run_v2_backtest()`。対象日を短くしても事前分布・ローリング・PIT に必要な履歴を切らない。
- 主指標はフラット日を含む同一評価営業日で比較する。fallback が増えた日と理由を分解し、on-demand 利用・終端フラット・PIT multiplier を区別する。固定の5%を普遍的な障害判定にせず、有効設定・過去の同条件・事前基準と照合する。
- slippage / financing / borrow / reverse を分解し、gross − costs = net、片道/往復、週末を含む暦日課金、モデル/実効 exposure を照合する。集計は [backtest-report](../../backtest-report/SKILL.md) に従う。
- 本番との整合確認には [leadlag-fund-improvement](../../leadlag-fund-improvement/SKILL.md) の対象経路と shadow 手順を使う。ウェイト差と実約定・口座の差を分け、比較日・入力・価格・設定が一致しているか確認する。

## シグナルからウェイトまで

`src/leadlag/models/production_v2.py` → `src/leadlag/models/v2/decision_engine.py` と呼び出し先で、実際に使った入力・分布取得経路・scores・ウェイト・RuleD・overlay・監査結果を観測する。MinVar の係数や overlay の有効性・適用順序は有効設定とコードで確認する。

9:10 時点で既知の当日 US リターン・gap は入力として追跡する。学習・ローリング統計・PIT の履歴は strictly historical とし、未知の当日ターゲット・未来行が混入していないかを検証する。「観測値はすべて当日行を除く」と一律に扱わない。

インスタンスキャッシュのキー・更新・クリア位置も追い、別日・別 config・前回呼出しの値を観測していないかを確認する。比較用のコード版・インスタンス・設定を分離し、数値差に加えてランキング・閾値判定・最終ポジションの差を確認する。許容誤差は契約と数値精度から決め、過去の sign agreement / RMSE を普遍的な合格基準にしない。
