# Production Decision Mode ユーザーガイド

## 概要
production.py の `--mode decision` モードで、TOPIX-17 の当日初値を入力して、即座に売買判定・発注数量まで出力することができます。

## 入力形式

### 1. CSV ファイルで初値を指定
```csv
ticker,open_price
1617.T,2500.0
1618.T,1800.0
1619.T,2100.0
... （全17銘柄）
1633.T,3100.0
```

### 2. コマンドライン実行例

**基本的な使い方（手持ち資金 500万円の場合）：**
```bash
python src/production.py \
  --mode decision \
  --trade-date 2026-03-27 \
  --jp-opens-csv test_opens.csv \
  --capital 5000000
```

**オプション一覧：**
- `--mode decision`: 当日判定モードを指定（既定値。省略可）
- `--trade-date YYYY-MM-DD`: 取引日（省略時は当日）
- `--jp-opens-csv PATH`: 初値CSV ファイルパス（必須）
- `--capital AMOUNT`: 手持ち資金（JPY）（既定値: 1,000,000）
- `--output-root DIR`: 出力ディレクトリ（既定値: results/production）
- `--run-tag TAG`: 出力フォルダの識別子（省略時はタイムスタンプ）
- `--start-date YYYY-MM-DD`: 履歴データ開始日（既定値: 2015-01-01）

## 出力形式

### コンソール出力
```
=== Trade Decision ===
ticker  open_price  signal  weight action etf_amount quantity
1617.T      2500.0 0.942650  0.000000   HOLD       0.0        0
1618.T      1800.0 0.961815  0.067882    BUY  302400.0      168
...

scale=0.5000, sigma_s=0.078463, long_short_mean_gap=0.182741
Available capital: 5,000,000 JPY
Total allocated: 3,436,200 JPY
Remaining capital: 1,563,800 JPY
```

**出力列の説明：**
- `ticker`: TOPIX-17 銘柄コード
- `open_price`: 当日初値（CSV から読み込み）
- `signal`: 当日の信号値（米国情報から推計）
- `weight`: ポートフォリオウェイト（-1.0 ～ +1.0）
- `action`: 売買判定（BUY / SELL / HOLD）
- `etf_amount`: 配置金額（JPY）
  - BUY のみ金額が必要
  - SELL/HOLD は 0
- `quantity`: 発注数量（口数）
  - weight が大きいものから優先的に配置

### CSV ファイル出力
`results/production/<run_tag>/decision_YYYYMMDD.csv`

上記のテーブルがそのまま CSV で保存されます。

## 資金配置のアルゴリズム

**ロング・ショート両方に資金配置（Weight-Proportional Allocation）：**
1. BUY（ロング）とSELL（ショート）の weight 絶対値合計の比率で、手持ち資金をロング予算・ショート予算に分割
  （信用前提のサイドレバレッジ 1.5 を適用。ロング・ショート各サイドで最大 `capital × 1.5` を使用可能）
2. 各サイドで weight の絶対値が大きい順にソート
3. 各ポジションについて、サイド予算と weight 比率で配置額を決定
4. 配置額 / 初値 を床関数で端数切り捨てして発注数量を計算

**例：**
- 手持ち資金: 5,000,000 JPY
- BUY weight 絶対値合計: 1.0, SELL weight 絶対値合計: 1.0
- サイドレバレッジ: 1.5
- → BUY 予算: 7,500,000 JPY（= 5,000,000 × 1.5）
- → SELL 予算: 7,500,000 JPY（= 5,000,000 × 1.5）
- → 各銘柄にweight比率で配分

## よくある質問

**Q. CSV のティッカーの順序は重要ですか？**
A. いいえ。1617.T～1633.T さえあれば、順序は任意です。

**Q. SELL（ショート）の資金配分はどう使われますか？**
A. ショートポジションにも weight 比率で資金が配分されます。信用取引の保証金算出の参考値として使用されます。

**Q. 資金が不足した場合、どのポジションを優先しますか？**
A. 各サイド内で weight が大きいものから購入します。資金枯渇時点で買い付け/売り付けが止まります。

## トラブルシューティング

**"CSV must have at least 17 rows" エラー**
- CSV が全17銘柄を含まないか、フォーマットが誤っている
- 各ティッカーがスペースなく正確に書かれているか確認

**"Previous close not found for ..." エラー**
- 指定した trade-date の前営業日の終値データがない
- 日付範囲内でデータが存在する日付を指定

**quantity が 0 のみの場合**
- signal がすべて小さく、weight がほぼ 0
- または資金が非常に少ない
- --capital を増やすか、別の日付を試してみてください

