# 実約定・資金容量・市場中立性レビュー

公式取引CSV 339件 / 4,130株を行単位で整理し、過去ログ・実建玉・余力と読み取り照合した。

- ローカルログとのgroup数量は23/23（207株）一致。ただし注文IDがCSVにないため一対一対応ではない。
- 約定価格まで一致した範囲は7グループ・8株。損益欄・費用明細を安全に使える主な範囲もここに限る。
- 実ポジションは7月29〜31日にgross 76.8万〜87.7万円、net/gross −3.06%〜−0.96%。新規信用余力は7.5万〜20.5万円。
- 9月24・25日は建玉0銘柄、信用余力0円。最近の稼働中ポートフォリオの中立性や取引容量はこのスナップショットから評価できない。
- 注文詳細の再照会はhealth checkのHTTP 404で止まり、個別注文の照会は送られていない。

詳しい制約・数値・次の照合要件は[実数量レビュー](actual_execution_cost_capacity_neutrality.md)を参照。

- [公式取引CSVから抽出した返済明細](official_repayment_transactions.csv)
- [ローカル注文ログとの照合表](local_to_official_fill_crosswalk.csv)
- [集計JSON](actual_metrics.json)
- [グループ単位の照合結果](reconciliation.json)
- 再生成: `python3 reports/20260927_actual_execution_review/build_actual_metrics.py`
