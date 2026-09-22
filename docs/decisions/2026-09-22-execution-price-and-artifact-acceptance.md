# 9:10価格の選択と本番artifact受入

- 日付: 2026-09-22
- 状態: accepted（価格契約の実装。運用全体の受入完了を意味しない）

## 背景

本番は当日の価格からgapを計算するが、BT・ML学習・gap事前生成は、明示した
`open_910_returns`を判断用価格に反映していなかった。実現targetだけを9:10へ
変えても本番との入力差は残る。また、9:10欠損時の寄付代替は既存仕様である。

## 決定

`data.intraday_inputs.resolve_execution_prices`と
`PITDataLake.get_execution_snapshot`をBT・学習・gap事前生成の共通入口にする。

- 観測あり: `jp_open_trade * (1 + open_910_returns)`を価格に使う。
- 明示された欠損セル: 有限かつ正の日次寄付価格を使う。
- 不正な観測（Inf、−100%以下）、欠損した日付・銘柄、使用不能な寄付価格は拒否する。
- raw観測のNaNを0へ変更しない。価格出所は`price_sources`で入力指紋・結果へ残す。
- gapは使用価格と前日終値から再計算する。h=3/5は当日成分に同じsnapshotを使い、
  先行h−1日の履歴は保持する。targetのh=1/3/5も同じraw観測を使う。
- 本番broker quoteの出所と日次寄付代替を区別する。出所ラベルは実取得時刻の証明ではない。

来歴不正・監査失敗・入力frame不存在の拒否は維持する。寄付代替が認められることは、
古いgap bundleや未来labelを使用してよいことを意味しない。

## Artifactと運用の受入

旧pickleへverified metadataを後付けせず、2015年以降の固定入力で学習し直す。
2010–2014の事前分布は分離する。既定パラメータ1候補を、各年の学習末尾5営業日を
除いた2020–2024年のwalk-forwardと、2024年末以前の学習を固定した2025–2026年で評価する。
過去に研究した区間なので、新たな未使用holdoutとは呼ばない。

metadata整合、OOS数値、本番/BTの一致、providerの実available_at、実口座照合、
scheduler、Hosted CIは別の証拠で判定する。証拠がない項目はPASSへ変更しない。
本番artifact参照の切替は全受入が揃ってから行う。

実行条件・結果は[受入報告](../../reports/20260922_production_acceptance/report.md)を参照。
