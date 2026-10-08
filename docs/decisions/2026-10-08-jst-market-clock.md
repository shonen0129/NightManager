# JST market clock の正本化（#50）

## 判断

市場の現在時刻と当日付は `leadlag.utils.timestamps.jst_now()` / `jst_today()` を
正本とする。`jst_now()` は `Asia/Tokyo` の timezone-aware datetime を返し、
`jst_today()` はその市場日付を返す。ホストの timezone は市場の意味論に使わない。

## 適用範囲

`daily` の09:15 cutoff、暗黙当日 decision / daily / close の休場判定、
df_exec の営業日鮮度判定、market calendar の省略日付に適用する。
明示されたaware datetimeとcache末尾日付もJSTへ変換してから日付を取り出す。
naive の明示入力は既存のJST契約を維持する。UTCで記録するcache更新時刻とTTL、
monotonic deadlineはそれぞれ従来の時計を使用する。

## 検証契約

固定した絶対時刻を `TZ=Asia/Tokyo` / `UTC` / `America/New_York` で評価し、
09:10 decision、09:15 / 15:30 close、営業日の誤skip防止、休日skip、
2025-01-23 08:00 の2営業日stale、future cache、休日跨ぎを回帰検証する。
対象市場境界にhost-localな `now()` / `today()` を戻さないAST契約検査を含む。
実発注・scheduler実市場受入はこのオフライン回帰とは別条件（#27）とする。
