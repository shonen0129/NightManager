# 感応度監査後のフォワード検証仕様

## 状態と目的

2026-09-24に仕様を凍結する。現行17ETF感応度を暫定ベースラインとして維持し、候補は研究シャドーで記録する。本番採用はしない。過去の2015〜2026データは既に参照済みである。仕様凍結後に初めて観測する日次データはprospective forward holdoutとして扱い、候補・基準を変更しない限り過去の既知期間と区別する。

この仕様は、提示されたJPX33感応度候補を年次時価総額で17ETF priorへ集約した場合の増分を、将来の日次データで一度だけ確認するためのもの。w6の金利経済解釈や33次元直接BLPXを採用する仕様ではない。

## 凍結する比較

- **Baseline**: `configs/production/production.yaml` を継承解決した現行V2と `SENSITIVITY_LABELS`。production ML overlayも含め、開始時点で固定した同じartifactをbaseline/candidate双方で使用する。overlay artifact/version、BLPXパラメータ、コスト・リスク設定のfingerprintを開始時に保存する。
- **単一primary candidate**: `configs/research/sensitivity_audit_jpx33_candidate_20260924.yaml` の33業種w3〜w6値をそのまま使用し、親TOPIX-17業種内で前年JPX年末総時価総額比に集約する。
- **ウェイトの時点ルール**: JPX公表月次統計の前年12月業種時価総額を使う。第1営業日13:00以降の公表より前には新しい年のウェイトを使わず、各年の第2日本取引日から適用する。対象市場のFirst Section/Prime切替をmetadataに記録する。これは浮動株調整済みETF保有ウェイトではなく、候補仕様そのものとして固定する。
- 17ETF直接予測と候補の差だけをprimary comparisonとする。US-only/JP-only/both w6、ランダム置換、単セル探索を将来データで追加しない。新しい候補を加えるなら別仕様・別試行として登録し、この期間を新しいforward holdoutに再利用しない。

## ターゲット、執行、評価期間

- シャドー開始は2026-09-25以降の最初の取引日。少なくとも**252営業日の完全なpaired観測**が得られるまで採否を判定しない。カレンダー営業日、信号生成数、17ETFすべて価格・気配が揃った日、paired評価日を別々に記録する。
- 予測ターゲットは各ETFの09:10時点midから大引けmidまでのリターンとし、決定時刻近傍の実際のbid/askとサイズを保存する。09:10の有効気配は09:10:00以降の最初のquoteで09:10:30を上限とする。基準となるclose quote/auctionは利用データと時刻を固定する。midが欠損・市場停止・約定不能の行は欠損理由を保存し、open-to-closeで埋めない。
- portfolio PnLは予測ターゲットと分ける。baseline/candidate双方について、同一の発注額・注文種類・09:10以降の板/気配・close執行を使って約定可能性、部分約定、非約定を記録する。mid-to-mid予測値を約定収益として扱わない。
- 片道slippage、取引手数料、financing、borrow、reverse、約定不能を記録し、観測約定値とモデル推定値を分ける。今の固定コストモデル値を実測費用と呼ばない。
- flat日を含む全paired営業日を主集計に含める。欠損日に取引をしない場合は実際にflatとなった理由を記録し、対象価格が欠損した日をゼロリターンとして作らない。

## 採否基準

すべての条件を満たす場合に限り、候補を研究上の採用候補にする。評価は片側/両側の選択を結果を見て変えず、次の事前固定条件で行う。

1. 252日以上の完全な17ETF paired quote/return観測がある。
2. 年率Net Sharpe差（candidate−baseline）が **+0.05以上**。
3. paired日次Net PnL差の20営業日non-circular moving-block bootstrap 95% CI下限が **0を上回る**（5,000回、seed 20260924）。
4. 最大ドローダウン悪化が **1.0 percentage point以内**、片道turnover増加が **10%以内**。
5. fallback率は増加しない。model grossは2.0以下、model net absolute exposureは0.05以下。side leverage後の実効exposureも別途報告する。
6. 関連する既往試行23件を含むDeflated Sharpeの試行回数を保守的に計上する。将来primary comparisonは24件目として記録する。過去期間で候補や採否基準を変更しない。

いずれかのデータ要件・基準を満たさなければ現行暫定ベースラインを維持する。合格後も研究上の採用と本番config更新を分け、別途本番昇格監査を行う。

## w6と33→17の追加作業を分離

- w6の経済的な金利感応度を主張する前に、対象市場、通貨、年限、金利水準か変化量か、予想外変化の定義、観測時刻を独立に定義する。現状の数値候補はラベルpriorであり、推定金利ベータとは呼ばない。
- 33業種を直接予測する案は、このprimary shadowから分ける。比較開始にはPIT構成銘柄・分類履歴とfloat-adjusted (W_t)、33次元の時点内 Ω、予測時点で利用可能な (W_t Ω_{33,t} W_t') の再現が必要。現在の494銘柄・2026分類snapshotは品質条件を満たさない。

## 固定した根拠と再利用制約

- 既知期間の17変種V2監査と費用・exposure結果: `reports/20260924_sensitivity_pipeline_audit/report.md`。
- 2015〜2017固定2017 capのPIT訂正影響: `reports/20260924_jpx33_pitcap_correction_2015_2017/report.md`。
- 実験registry: `var/experiments/registry.jsonl`。既知の感応度試行数を将来の補正から除外しない。
- 将来観測の仕様、予測、quote/board snapshots、約定可否、cost ledger、全期間のbaseline/candidate日次PnLは、評価開始前にfingerprintし、追記専用で保存する。
