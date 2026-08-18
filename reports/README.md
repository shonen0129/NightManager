# Reports Directory Convention

`reports/` には実験・検証・ウォークフォワード・シャドーラン等の結果を保存する。

## ディレクトリ命名規約

- 基本形: `reports/<category>/<experiment_name>/`
  - 例: `reports/walkforward/<name>/`
  - 例: `reports/shadow/<name>/`
- 各実行結果は上記ディレクトリ直下に、run 日付を先頭にしたサブディレクトリを作成する。
  - 形式: `YYYYMMDD_<run_descriptor>/`
  - 例: `20260815_111443_full_2015_20260813/`

## 運用上の注意

- カテゴリ名は英数字・ハイフン・アンダースコアを推奨。
- `<experiment_name>` は同一実験の複数 run をグループ化する識別子。
- 日付 prefix の run ディレクトリには、レポート markdown、CSV、JSON、画像などを格納する。
