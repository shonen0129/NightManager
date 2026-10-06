# 2026-10-06: 全体監査で検出した入力・評価・保存境界の修正

Status: accepted（下記実装境界のみ。関連issue全体の完了を意味しない）

## 判断

- V2評価開始は2015-01-05以降に限定する。要求期間とsource期間が交差しない場合は拒否し、先頭行への丸めは行わない。`evaluation_period`で要求/実評価/source期間を保存する。
- entry→close targetとcarryの価格基準を接続する。carryへ渡すのは `(1+寄付gap)*(1+open→09:10)-1`、09:10欠損時はtargetと同じ寄付proxy。既存の前日側への損益帰属を保つ。数量・cash・終端inventory・artifact切替の連続会計は別途issue #33で完成させる。
- 主turnoverはside leverageを含むsimulated opening/closing notional合計の1/2。旧目標weight差は `daily_target_weight_turnover`、1/2を掛けない売買量は `daily_execution_volume`。`accounting_contract`で旧レポートと区別する。実口座数量からの実費volumeを証明したものではない。
- 欠損targetをゼロ補間しない。保有0のprice PnLは0と確定できる。保有ありの非有限targetは非有限PnLのまま、shared metricsが全評価日で拒否する。月次は12年率、DDは初期wealth1から計算する。
- MinVarは選択済みのlong/short indicesを受け取り、basketを再選択しない。各sideの銘柄数・重複・範囲を検査する。すべての本番/研究呼び出し元を正規APIへ更新し、旧q引数adapterは残さない。
- DSRはmetric status、日次schema、finite returnsとT、frequency、正の年率係数、整数trial数、finite varianceを検査する。trial Sharpe配列はN件を要求する。系列不足・定数系列を無条件の正規分布momentsへ置き換えない。computed metricsを追加metricで上書きしない。未登録の過去探索familyを復元したことにはならない。
