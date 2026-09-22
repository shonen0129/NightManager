# 段階A〜C 第6回指摘 修正報告 — 2026-09-14

## W01 — 分布来歴の検証契約を統合

`src/leadlag/utils/distribution_provenance.py`を追加し、`sig_date`/`signal_date`の一致、scalar日付、要求trade dateとの一致、有限な整数horizonを一つの検証関数で正規化します。`gap_matrix_io.py`と`models/v2/distribution_source.py`が同じ関数を使用します。invalid bundleは互換loaderでも配列を返さず、V2のfile cacheはon-demandへ進み、blendは該当horizonを混ぜず警告を保持します。

## W02 — snapshotの総期限とworker寿命

`var_history.py`でVaRの総deadlineをsnapshot開始前に決め、SQLite接続・backup progress・retryを残り時間に拘束します。snapshotはkey計算とBTへ同じものを渡します。BT timeout後もworker closureが`TemporaryDirectory`を保持し、workerの`finally`でcleanupします。worker開始前に設定・キャッシュ処理が例外終了した場合も、呼出側のcleanup経路で一時ディレクトリを回収します。

## 回帰

- 来歴：正常、matching alias、alias矛盾、null/NaT、配列trade date、Inf horizon、h=1/3/5のloader/compute/blend。
- snapshot：A→Bの更新競合、SQLite排他ロック中の期限、BT timeout中とworker終了後のsnapshot寿命。
- 対象unit tests：`tests/unit/test_gap_matrix_io.py`、`tests/unit/test_stage_abc_followup_fixes.py`。

## 検証結果

全`tests/`、Ruff、mypy、compileall、import-linter、既存F01〜F22/R〜Uプローブを修正後に実行し、結果を同ディレクトリへ保存しました。

- [pytest全体](./pytest_full_after_cleanup.log)：646 passed / 17 warnings、711.96秒、watchdog exit=0（前回実行の[ログ](./pytest_full.log)も保存）
- [W01/W02境界プローブ](./probe_boundaries.log)：矛盾alias・非scalar/Infを拒否、on-demand切替、h1/h3/h5 blend警告、SQLite lockを約1秒で停止、timeout中のsnapshotをworker終了まで保持
- [VaR版競合プローブ](./probe_var_gap_version.log)：A snapshotをkeyとBTへ一貫して渡す
- [F01〜F22/R〜Uプローブ](./probe_review.log)：注文・監査・PIT・cache・ML来歴・コスト・metricsの再発確認
- [Ruff](./ruff.log)、[mypy](./mypy.log)、[compileall](./compileall.stderr)、[import-linter](./import_linter.log)

実broker・本番DB・本番発注は使用していません。

既知の未完了は、legacy layoutの本番ML artifact再生成、本番/BT weights照合、実約定後exposure検証です。今回の修正範囲には含めていません。
