# 2026-10-04 quote capture authentication follow-up

## 確認した事実

2026-10-01 と 2026-10-02 の読み取り専用09:10 captureは、各日2回の試行後に `FAILED` となっている。保存されたエラーは両日とも `ValueError: Missing encrypted URL key 'sUrlRequest' in login response.` で、live と shadow の記録がそれぞれ存在する。

現行アダプタでは、この例外はログイン応答の `p_errno` が0かつ `sResultCode` が0の後、REQUEST用仮想URLが空の場合にだけ発生する。したがって認証応答までは成功したが、時価情報要求は送られておらず、frozen quote snapshotも作成されていない。

保存されたcapture artifactにログイン応答全体は含まれず、`sKinsyouhouMidokuFlg` の値もない。この証拠から未読書面を今回の原因と断定できない。

## API仕様との照合

現行の[立花証券v4r10 APIリファレンス](https://www.e-shiten.jp/e_api/mfds_json_api_ref_text.html)では、ログイン応答の `sKinsyouhouMidokuFlg=1` は交付書面未読を表す。未読の場合、標準Webサイトで書面を確認するまでAPIは利用できず、仮想URLは発行されない。

これは保存エラーと整合する候補だが、保存された認証フラグがないため未確認である。フラグが0または欠落していた場合は、応答の他の安全なメタデータを確認して別原因を調べる必要がある。

## 対応

- Tachibanaクライアントは、仮想URL欠落時に書面未読フラグが1なら、その理由と標準Webサイトでの確認手順をエラーに示す。
- フラグが0または応答にない場合も、その状態をエラーへ示す。認証応答全体、秘密情報、URL queryは保存しない。
- 日次運用手順に、read-only capture後の切り分け方を追記した。
- 回帰テストはフラグ1と0を固定している。broker通信は再実行していない。

## 残る受入条件

次の実取引日にread-only captureが成功し、JP17+TOPIXの全18銘柄と同一snapshot IDの後続伝播を確認する必要がある。認証応答に未読フラグが含まれ、値が1なら、口座管理者が標準Webサイトで交付書面を確認した後に定期captureで再確認する。フラグが0でも仮想URLが欠ける場合は、ログイン応答の追加診断を経て原因を特定する。

この調査中に発注、決済、注文照会は行っていない。
