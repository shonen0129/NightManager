# LightGBM期待リターンと注文別費用の研究構成

## 状態

研究用の一次実装。現行モデル・本番設定・本番発注経路には適用しない。

## 判断

LGBMの回帰出力をリターン予測として扱い、注文費用は別の費用計算で見積もり、注文判断の段階で比較する。

現行artifactの学習ラベルは `sign(score) × 実現9:10→大引けリターン − 0.001`（往復10bps）である。研究実装では予測値へ0.001を戻して、signal方向を付けた銘柄別グロスリターン予測にする。出力は確率と呼ばず、実現値による校正も済んだとはみなさない。

各候補注文の予想増分利益を `実効レバレッジ × Δweight × LGBM期待リターン` とする。費用計算は注文notional、full spreadの半分/片道、別項目の手数料・impact、取引側数からspread / commission / impactを計算する。費用スケジュールは銘柄別入力を受け付ける。現在の過去検証では真の9:10板が揃わないため、0.20% / 0.25% / 0.30%の共通spreadシナリオを使い、手数料とimpactは未推定として0仮置きする。

費用控除後の予想利益が正の候補だけをLPへ渡し、canonical目標ウェイトまでの移動割合を選ぶ。netを0、model grossを2.0以下に制約する。financing / borrow / reverseは注文費用に重複加算せず、V2の損益計算に別計上する。

## 根拠と制約

- Stage 7で確認した現行LGBMの固定round-trip費用は10bpsで、注文リンク済みの全期間fill feeはない。
- 9:10 quote / boardの過去データが不足しており、全銘柄・全評価日の実行可能spread、板厚、価格impactを再構成できない。
- previous selected weightは実際のbroker inventoryではない。注文サイズ・lot丸め・部分約定・待ち行列も再現しない。
- LGBM targetの時間軸は9:10→大引け。ポートフォリオ損益に含む翌日gap returnは注文判断の予測値に含めない。
- 評価期間は既に確認済みのOOS区間であり、独立holdoutではない。数値が改善しても本番採用の根拠にしない。

## 再検討条件

注文IDで結合した9:10価格、約定数量・価格、手数料・逆日歩の記録が蓄積した後、銘柄・side・注文金額・板厚に応じた費用入力を検証する。別途、実際のinventoryを使うwalk-forward OOSでnet return・DD・turnover・コスト内訳を評価する。直接のグロスリターンtargetへの再学習と、費用条件を特徴量に入れる案は、その後に独立比較する。

再現スクリプトと一次結果: `src/research/scripts/experiments/experiment_lgbm_order_cost_20260923.py`、`reports/20260923_profitability_order_11/report.md`。
