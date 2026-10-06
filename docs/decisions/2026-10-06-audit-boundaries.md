# 2026-10-06: 全体監査で検出した入力・評価・保存境界の修正

Status: accepted（下記実装境界のみ。関連issue全体の完了を意味しない）

## 判断

- V2評価開始は2015-01-05以降に限定する。要求期間とsource期間が交差しない場合は拒否し、先頭行への丸めは行わない。`evaluation_period`で要求/実評価/source期間を保存する。
- entry→close targetとcarryの価格基準を接続する。carryへ渡すのは `(1+寄付gap)*(1+open→09:10)-1`、09:10欠損時はtargetと同じ寄付proxy。既存の前日側への損益帰属を保つ。数量・cash・終端inventory・artifact切替の連続会計は別途issue #33で完成させる。
- 主turnoverはside leverageを含むsimulated opening/closing notional合計の1/2。旧目標weight差は `daily_target_weight_turnover`、1/2を掛けない売買量は `daily_execution_volume`。`accounting_contract`で旧レポートと区別する。実口座数量からの実費volumeを証明したものではない。
- 欠損targetをゼロ補間しない。保有0のprice PnLは0と確定できる。保有ありの非有限targetは非有限PnLのまま、shared metricsが全評価日で拒否する。月次は12年率、DDは初期wealth1から計算する。
- MinVarは選択済みのlong/short indicesを受け取り、basketを再選択しない。各sideの銘柄数・重複・範囲を検査する。すべての本番/研究呼び出し元を正規APIへ更新し、旧q引数adapterは残さない。
- DSRはmetric status、日次schema、finite returnsとT、frequency、正の年率係数、整数trial数、finite varianceを検査する。trial Sharpe配列はN件を要求する。系列不足・定数系列を無条件の正規分布momentsへ置き換えない。computed metricsを追加metricで上書きしない。未登録の過去探索familyを復元したことにはならない。
- backtestの設定保存はbroker設定を含まないallowlistと非秘密設定のSHA-256を使用する。Tachibana HTTP/transport/JSON parse例外はURL、raw response、raw server error textを含まないAPI例外へ変換し、unsafe causeをtracebackへ出さない。過去保存済み資格の失効確認・履歴scrubは別途管理する。
- US proxyは `tickers.US_INCEPTION_DATES` より前の欠損cellにのみ使用し、`us_proxy_*` provenanceを残す。後の欠損はstrict拒否、非strictでは警告して該当recordを拒否する。旧前処理cacheはcontract versionで拒否し、strict再構築へ進む。raw/live入力をこの作業で書き換えない。
- JPXの恒常的な年末年始休業日をstatic年表から分離する。未知年の国民の祝日は引き続きjpholiday/staticの更新に依存する。

## 日付の一次資料

proxy境界は保守的にfund inceptionを用い、初回上場後の不定なCCリターンをproxyで埋めない。

- XLC: 2018-06-18。[State Street XLC](https://www.ssga.com/us/en/intermediary/etfs/state-street-communication-services-select-sector-spdr-etf-xlc)
- XLRE: 2015-10-07。[State Street XLRE](https://www.ssga.com/us/en/individual/etfs/state-street-real-estate-select-sector-spdr-etf-xlre)
- MTUM/VLUE: 2013-04-16、USMV: 2011-10-18。[iShares product list](https://www.ishares.com/us/literature/brochure/ishares-product-list-en-us.pdf)、[VLUE](https://www.ishares.com/us/products/251616/VL)
- 年始3日間と12/31は休業。[JPX FAQ](https://www.jpx.co.jp/faq/others_general.html)
