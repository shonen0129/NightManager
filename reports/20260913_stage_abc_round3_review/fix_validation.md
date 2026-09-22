# Round 3 指摘の修正・検証記録

対象は `review.md` の T01〜T05 と、同レビューで併記したVaR/ES cacheの
artifact fingerprint問題である。実運用の発注・決済・外部broker APIは実行していない。

## 修正内容

| ID | 修正 | 回帰で確認したこと |
|---|---|---|
| T01 | ML overlay の必須来歴を公開時・読込時・適用時で共通検証した。`metadata_version=2`、`metadata_status=verified`、scalarかつ有効な `train_start` / `train_end`、非空のdata/config hashを要求し、`NaT`、null、train_endを超えるlabel終端を拒否する。 | `train_end=None` と `NaT` は保存も直接適用も失敗する。学習終端日以前の適用は従来どおり失敗する。 |
| T02 | `CURRENT` のないroot直下 `model.pkl` / `metadata.json` は常に拒否する。versioned artifactではdigestと、埋込みmetadata・外部metadataの構造・来歴の一致を確認する。 | 外部metadataだけを2020年へ差替えたlegacy root artifactを読込拒否する。 |
| T03 | single-horizonでもmetadataの有無にかかわらず来歴検証する。`.npy`互換出力には `gap_metadata[_hN]_YYYYMMDD.json` sidecarを正式追加し、生成ツール・batch確認・回帰fixtureが同一bundleで扱う。 | sidecarなし `.npy` は `audit_failure=true` のflatとなり、sidecar付きbundleは通常計算とリーク監査を通る。 |
| T04 | `fetch_fill_prices` は個々の結果を保持したまま全orderを照会し、1件でもdetail照会に失敗すれば `FillPriceReconciliationError` を送出する。post-decisionはjournal照合後に非0終了へ、closeは `close_incomplete=true` へ進む。 | 既確認quantityを失わないこと、正常なFILLED submit後のdetail障害が成功終了にならないこと、closeも不完了扱いになることを確認した。 |
| T05 | 初回 `api_execution_log.json` 書込み失敗を `OrderExecutionIncomplete` に格納し、post-decisionがin-memory summaryからfill・建玉・余力・journalを照合してから再送出する。 | log書込み失敗後も全照合が実行され、後続の書込み成功時にはerrorを含むsummaryが保存されることを確認した。 |
| 追加 | VaR/ES return cacheのoverlay fingerprintは `CURRENT` が選ぶversionだけを対象にした。 | inactive version / stagingの更新ではkeyが不変、active versionの内容またはCURRENT切替ではkeyが変わる。 |

## 運用上の移行

既存のroot直下overlay artifactは安全に対応付けられないため、loaderと検証ツールが
拒否する。利用する場合は `tools/production/train_ml_order_overlay.py` で再学習し、
`CURRENT` を持つversioned artifactとして公開する必要がある。metadataだけを書換える
旧migration scriptは、再学習対象を報告するread-only scriptへ変更した。

## 検証

- `python -m compileall -q src/leadlag tests tools src/research`
- 最終対象unit/integration/regression: 91 passed
- 全テスト: `python -m pytest tests/ -n auto` → **614 passed, 17 warnings, 376.39s**
- `ruff check`（変更対象）→ passed
- `mypy`（変更した `src/leadlag` 8 modules）→ passed
- `lint-imports` → 4 contracts kept, 0 broken

`tools/research/compute_gap_adjusted_distribution.py` は既存のstrict mypy違反を多数含む
ため、今回の1 import / 保存経路の変更を含めて全体strict化はこの修正範囲に含めていない。
