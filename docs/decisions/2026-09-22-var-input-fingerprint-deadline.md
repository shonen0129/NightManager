# VaR入力指紋を既存の全体期限に含める

- Date: 2026-09-22
- Status: accepted / implemented
- 対象: [構造レビューP2](../../reports/20260921_structural_full_review/report.md)

## 結論

**小さな修正として対応した。緊急修正ではなく、P1より後の通常の修正単位として実装した。**

現状は待機期限の契約に穴があるが、数値を誤って採用したり、空のrisk履歴を正常扱いしたりする問題ではない。既存の全体レビューのBLOCKはP1と未充足の本番受入条件によるもので、P2だけを緊急の本番停止理由とはしない。

## 影響と優先度

[var_history.py](../../src/leadlag/execution/var_history.py)の`historical_input_snapshot.fingerprint`だけが、前後の準備処理と異なり`timed_call`を経由しない。

- 本番bridgeは`app_config.strategy`を渡す。現行StrategyConfigに`var_history_timeout`フィールドはなく、通常の本番経路では関数の既定値**300秒**が使われる。過去レビューの0.1秒は障害注入の条件であり、本番の設定値ではない。
- 指紋計算はメモリ内のDataFrame・配列のハッシュであり、処理自体にネットワーク、broker、SQLite待ちはない。現在の証拠から、通常入力で長時間ハングするとは言えない。
- それでも先行処理で残時間を消費すれば、この区間で超過できる。現在は計算終了後に次の残時間チェックで空の履歴を返す。[risk.py](../../src/leadlag/core/risk.py)は0サンプルを停止理由にするため、risk判定を迂回して注文する問題ではない。
- 本番ではこの呼出しをlease内で行う。超過中は後続処理とlease解放が遅れる。batchの外側には既定1800秒のプロセスグループ期限もあるが、これはVaRの300秒期限とは別の保護である。

### 計算時間の確認

各ケース5回、指紋計算だけを計測した。

| 入力 | 直接計算の中央値 | 既存timeout wrapper経由の中央値 |
|---|---:|---:|
| 固定回帰DataFrame: 4,135行 × 122列 | 3.67ms | 3.72ms |
| 上記＋2,757日分のPIT mapping＋補助frame | 25.63ms | 25.54ms |

日次データ・PIT診断は固定fixtureを使用。PIT mappingは1,209,790要素、補助入力は17列の合成frameを4つ追加した。これは全入力を揃えた本番実行の計測ではない。5回の小さな測定差から速度改善は主張しない。少なくともこの規模では、300秒の期限に対して指紋計算が大きな負荷になっている証拠はない。

## 実装内容

既存の残時間付き`timed_call`にこの計算も含める。

```python
input_snapshot_hash = timed_call(
    lambda: historical_input_snapshot.fingerprint,
    "VaR/ES run-input fingerprint",
)
```

同じ絶対期限の残時間を使い、既存の`except TimeoutError`による空履歴返却とsnapshot解放を利用する。指紋の定義・cache key・戦略パラメータの変更は不要。

### 実装の検証

修正前後の候補比較に加え、実装後の回帰テストと全テストを実行した。

| ケース | 確認結果 |
|---|---|
| 通常のcache hit | 現行と候補でcache key・返却系列が一致 |
| 指紋に0.5秒遅延、全体期限0.1秒 | 現行約0.510秒、候補約0.107秒で返る。両者とも空履歴 |
| 候補のタイムアウト後 | 後続BT・cache書込なし。gap snapshotは返却時に解放済み |
| 指紋計算の例外 | ValueErrorを握り潰さず伝播し、snapshotも解放 |

### 残る制約

`run_with_timeout`はdaemon threadへの待機を打ち切る方式で、計算を強制停止しない。検証でも、呼出し元へ返った時点では指紋threadが動いており、その後終了した。

今回の指紋は所有済みのメモリを読み、ファイル・broker・共有cacheへ書かないため、遅い完了を採用せず破棄できる。gap snapshotのディレクトリ解放後も、そのファイルを読み直さない。一時的にCPU・メモリ使用が続く点と、GILを長く保持する処理まで厳密に停止できるわけではない点は残る。プロセス全体の強制停止は外側のguardが担う。

## 他の案

| 案 | 評価 |
|---|---|
| 当面維持 | 実測上の緊急性は低く、短期保留は可能。ただし数行で揃えられる期限契約の不一致が残る |
| 指紋の前後だけで期限を確認 | 期限切れ結果の採用は防げるが、計算中の待機超過を解消しない |
| snapshot生成と指紋を同じworkerへまとめる | 実現可能だが、今回は既存の個別fingerprint用`timed_call`に揃える方が変更が小さい |
| 指紋をmemoizeする | immutable入力の最適化として別途検討できるが、初回の期限問題は別に残る。今回の実測では必須ではない |
| hash専用subprocessを新設 | 強制停止は可能になるが、大きな入力の転送と子プロセス管理を追加する。今回の影響に対して変更が大きい |

## 実装後の受入結果

1. 通常経路のcache key・返却系列を維持した。
2. 指紋計算が遅延しても期限内に空履歴を返し、後続BTやcache書込を始めないことを回帰で確認した。
3. 遅れて完了した指紋を採用せず、snapshot解放と例外伝播を維持した。
4. 対象テスト11件、全`tests/` **853 passed / 17 warnings**、compileall、Ruff、mypy、import-linterを通過した。

## 証拠

- [検証スクリプト](../../reports/20260921_structural_full_review/assess_p2.py)
- [計測・候補比較の結果](../../reports/20260921_structural_full_review/p2_assessment_results.json)
- [実行ログ](../../reports/20260921_structural_full_review/p2_assessment_verified.log) / [120秒の外側watchdog記録](../../reports/20260921_structural_full_review/p2_assessment_verified.json)

本番設定と実データは変更していない。変更対象は`var_history.py`の期限適用と回帰テストである。
