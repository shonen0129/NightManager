# 本番artifact・実運用受入

実行日: 2026-09-22。対象はP2修正後の作業ツリー。実行結果を確定した。

## 確認済みの変更

- 9:10観測がある場合のBT・学習・gap事前生成の価格選択を共通化した。欠損時の寄付代替を維持し、raw NaNと価格出所を保存する。
- `.env`の秘密鍵パスを現workspaceへ修正した。通常設定でwallet/positions読み取り、pending照合が成功した。
- scheduler 5件の登録とworkspace一致を確認した。9/22のholiday実行で各jobの終了コードを確認し、P&Lは既定dry-runへ切り替えた後にexit 0となった。実稼働日の発注一巡は未証明である。
- P2期限回帰11件、価格修正後の全テスト874件（先頭warm-up回帰を含む）、gap事前生成追加後の関連30件が成功した。

## Artifactの固定評価

[事前計画](evaluation_plan.md)に従う。固定したcanonical日次入力は2009-01-07〜2026-08-17。
2015–2025年に9:10観測はなく、仕様の寄付代替を使う。2026年の欠損セルにも同じ規則を使う。
実測のない9:10価格やprovider取得時刻は生成しない。

過去の[運用実行報告](../20260922_structural_acceptance_execution/report.md)で「全日付の実9:10観測が必須」とした受入条件は修正する。
欠損時の寄付代替は仕様であり、それだけでartifact再生成を阻止しない。同報告の7月末までという入力範囲は旧pickleについての結果で、今回のcanonical入力は8月17日まで存在する。

### 固定候補の結果

2020-01-06〜2026-08-14の1,555営業日（フラット日を除外なし）を、既定の費用・side leverage 1.5で評価した。2026-08-17は`is_provisional=true`で寄付17銘柄が無効な終端行のため、正規BTと同じく除外した。

| 指標 | ML無効対照 | 固定候補artifact |
|---|---:|---:|
| net Sharpe | 2.2047 | 2.2201 |
| gross Sharpe | 2.4678 | 2.4822 |
| 最大DD | -27.75% | -29.41% |
| 平均turnover（raw weight） | 1.2971 | 1.3083 |
| 期間費用合計（return fraction） | 3.8574 | 3.8721 |

候補はnet Sharpeがわずかに上がったが、最大DDが悪化したため、事前基準の数値ゲートは `promotion=false` だった。2020〜2024年の年次foldでも、Sharpeが改善しない年（2022年）があり、DDはいずれも候補が悪化した。20営業日block bootstrap（seed 42、1000回）の平均日次差95%区間は [0.0001269, 0.0002356] だが、Sharpe改善の有意性とは解釈しない。新パラメータは追加しておらずDSRは未算出である。

この数値ゲートの結果を変更せず、2026-09-23にユーザーが候補artifactの本番昇格を明示承認した。承認は数値ゲートを上書きする operator override として [昇格記録](promotion_record.json) と [decision](../../docs/decisions/2026-09-23-candidate-artifact-promotion.md) に固定した。candidateのactive version `20260922T011723828589Z-7e1a81ab2617` は [production_20260923](../../models/ml_order_overlay/production_20260923) へ immutable に公開し、本番configはこのversioned rootを参照する。旧 `phase2_8` のlegacy rootは参照しない。年次fold、費用4内訳、監査、parityは [artifact_evaluation.json](artifact_evaluation.json) にある。候補は3日（観測あり、銘柄単位欠損、全銘柄寄付代替）でProductionRunner・canonical BT・collectorのweight差が0、数値/リーク監査が全てPASSだった。

gap h=1/3/5のcache対on-demandは9比較で全てREADY、μ/Ωの最大差0。結果は [gap_path_parity.json](gap_path_parity.json) にある。

VaR replayは同じcandidate artifactと固定入力で、cold生成375日・warm cache 375日を比較した。両方とも終端2026-08-14、return最大差0、artifact version一致。coldは約370秒、warmは約1.5秒だった。入力fingerprintを含むdeadline・cache・canonical BTは通ったが、これは固定履歴replayであり、当日のprovider freshnessを証明しない（[var_path_parity.json](var_path_parity.json)）。

## 未受入条件の追加確認

2026-09-22 16:26 JST に、未受入条件を安全な読み取り・dry-run範囲で再処理した。全証拠は
[remaining_condition_evidence.json](remaining_condition_evidence.json) に固定した。

| 条件 | 判定 | 実施結果 |
|---|---|---|
| 過去providerの実 `available_at` | **未証明** | canonical入力は `observed_at` とsession cutoffを持つが、provider発行時刻は保存されていない。現在のmacro再取得だけはrequested/receivedを記録し、過去時刻は合成していない。 |
| broker約定・部分約定復旧と費用突合 | **部分PASS / 費用突合未受入** | 提供されたTachibana取引記録CSV（SHA-256 `48ae6774d763338950d7452e9a46fd75063c3ce7acb8cc1d0ab4f08c2f6aebff`）から、既知27注文を23グループに集約し、約定日・銘柄・売買・数量を全グループ一致（207株=207株）、単価と損益/受渡金額を取得した（[解析結果](tachibana_csv_analysis.json)）。CSVに注文IDと手数料・資金調達・貸株・逆日歩の内訳はなく、API再照会も27件すべて`CLMOrderListDetail / 991012`のため、費用の最終突合は未受入。 |
| scheduler本番一巡 | **部分PASS / 未受入** | 9/22のlaunchd実行はmarket holidayのためdecision/closeが発注なしでexit 0、distributionもexit 0。update-market-dataはstrict validationが古い履歴の先頭NaNでexit 1（現行production configのprobeでは続いて1619.T/1626.Tの2009-01-13 open=0も検知）。P&Lはメール送信をscheduler既定dry-runへ変更後にlaunchd再実行しexit 0。さらに`launchctl submit`の受入wrapperでexit 0・`dry_run=true`・`orders_submitted=false`を確認した。実稼働日の発注一巡は未証明。 |
| Hosted CI | **PASS** | 最終runはbaseline単独1件、残り872件、静的検査、wheel、smokeを全てPASS。 |

現行production configは `models/ml_order_overlay/production_20260923` を参照するよう更新した。ローダーは `CURRENT`、metadata `verified`、model SHA-256を検証し、設定解決後の本番model bundleでactive versionの一致を確認した。数値ゲートは `false` のまま記録し、今回の昇格はユーザー承認による operator override である。発注・取消・決済・再送・メール送信はこの検証では実行していない。

実注文を伴うscheduler試験の許可は受領した。ただし9/22・9/23は市場休業日で、次の営業日は9/24である。昇格後のproduction configはcandidateのversioned artifactを参照する。実稼働日の発注一巡はまだ未証明であり、update-market-dataが古い履歴の不正値でfail-closedした条件も残るため、このターンでは実注文を開始していない。

受入wrapperは固定したcanonical入力と受入専用configで実行した。現行production configをそのままdry-runしたprobeは、更新不能な過去日次open（1619.T/1626.Tの2009-01-13が0）をstrict validationが検知して停止した（[probe log](production_config_dry_run.log)）。この停止は壊れた履歴で発注しないfail-closed挙動であり、0値や未証明artifactを補ってPASSにはしていない。

## Hosted CI

専用ブランチ `codex/structural-acceptance-20260922`（最終commit `48abd1fbfc7c4ee3df934061627ef5d493655212`）へ、認証情報・口座ログ・運用cacheを除外した283ファイルを送信した。初回 [run 35694973455](https://github.com/shonen0129/NightManager/actions/runs/35694973455) はSHAP経由の古いllvmlite解決で失敗した。`research`/`nonlinear` extraをproduction CIから外し、lock済みLightGBMだけの`ci-ml` extraへ分離した後、LightGBM未導入による回帰flat fallbackを解消した。macOSで生成したfixtureのinput hashは保存し、CPU/OSをまたぐ厳密なhash比較は回帰fixtureから分離して、同一プラットフォームのprovenance unit testで継続する。最終 [run 35697322788](https://github.com/shonen0129/NightManager/actions/runs/35697322788) は成功し、baseline単独1件、残り872件、静的検査、wheel、smokeを完了した。

口座の過去実約定ログは27注文、保存済みfill detailは27件だった。提供CSVにより約定日・銘柄・売買・数量・単価・損益/受渡金額は全23グループで数量突合できた。一方、broker再照会は27件すべて`991012`で、注文IDとの一対一対応と費用内訳は未証明のままである（[historical_fill_status.json](historical_fill_status.json)、[CSV解析](tachibana_csv_analysis.json)）。

## 変更後の再検証

- 昇格後のconfig・gap provenance更新後も、全`tests/`は **874 passed / 17 warnings**（先頭warm-up回帰を含む）。
- 変更対象のRuff・mypy・shell syntax・compileallはPASS、文書リンクは102件を検証した（[compile log](compile_after_warmup.log)、[docs log](docs_after_warmup.log)）。
- 昇格後の本番configを、固定canonical入力・API無効・dry-runで実行し、candidateのactive versionをロードしたうえで `dry_run=true`・`orders_submitted=false`・exit 0を確認した（[実行ログ](promoted_production_dry_run.log)）。
