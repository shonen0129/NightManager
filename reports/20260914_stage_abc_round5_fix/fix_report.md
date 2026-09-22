# 段階A〜C 第5回指摘 修正報告 — 2026-09-14

第5回レビューのV01〜V04を、下流の利用経路まで含めて修正しました。対象は未commit作業ツリーです。過去のユーザー変更は保持しています。

## V01 — 不整合なgap bundleの利用

`src/leadlag/utils/gap_matrix_io.py`を変更し、`.npy`のmanifest欠落、μ/Ω/metadataのdigest不一致、fatalなshape・有限性違反では、非strictでも配列を返さず`(None, None, None, alerts)`を返すようにしました。低層診断用の`load_gap_bundle(require_metadata=False)`は残し、3値互換API `load_gap_matrices`はmetadata必須を既定値にしています。metadataを必要とする呼び出しは明示的に`require_metadata=True`を渡し、V2の`FileCacheDistributionSource`も来歴検証を通過したbundleだけを利用します。

`compute_distribution`、`apply_multi_horizon_blend`、V2のFileCache sourceは無効入力をon-demandへ送り、失敗時はflatへ進みます。blendは不正horizonを混ぜず、h1へ戻した警告を保持します。FallbackPolicyはfatalな整合性・provenance失敗を`audit_failure`と診断へ伝播します。

## V02 — Tachibana状態コード

`src/leadlag/broker/tachibana/client.py`で、brokerの状態コードを数量推定より先に判定します。取消完了7、部分/全部/繰越失効11・12・19は終端`CANCELLED`とし、訂正失敗5・取消失敗8は原注文の終了を意味しないため、約定量に応じて`SUBMITTED`または`PARTIALLY_FILLED`として追跡を継続します。受付エラー・無効・障害系2・14・17・20・21は、数量欄が全量でも`FAILED`を維持します。全部約定10、部分約定9の分類は維持し、約定数量は詳細照会で引き続き収集します。

状態コード7/9/10/11/12/19、要求失敗5/8、受付エラー等2/14/17/20/21について、数量0・部分・全量を含む回帰を`tests/unit/test_tachibana_broker.py`へ追加しました。

## V03 — common-input cacheのschema identity

`src/leadlag/utils/dataframe_fingerprint.py`を追加し、shape、列名・列順・列dtype、index型・dtype・名前、index付きセル値を一体でfingerprintします。`_BLPBase._prepare_common_inputs`の`df_exec`と`p_910_df`へ適用しました。数値bufferが同じで列名だけが交換された入力は別cache keyになり、fresh builderと同じ計算を再実行します。

`tests/unit/test_stage_abc_followup_fixes.py`にschema identityの回帰を追加しました。第5回レビューの固定fixture（4135行×122列、実builder）で、修正前に発生した最大差0.0683854が解消され、cacheは再利用されません。

## V04 — VaRのgap入力snapshot

`src/leadlag/execution/var_history.py`にgap入力snapshotを追加しました。SQLiteは`Connection.backup`で一貫したDB snapshotを作り、directory-backed入力はコピー前後のfingerprintとsnapshot内容を照合して安定した世代だけを採用します。snapshotのfingerprintをcache keyへ入れ、同じsnapshot pathをV2 backtestへ渡します。snapshot取得不能時は空系列を返し、risk checkが停止側へ進みます。SQLite本体だけを直接コピーしてWALを落とす経路は使いません。

`tests/unit/test_stage_abc_followup_fixes.py`で、writerがAからBへ更新した後も、開始時に固定したsnapshotがAのμ・metadataを返すことを確認しました。第5回の実VaRプローブでも、cache keyはA、backtest読込はA、途中更新後の再利用値もAでした。

## 検証

- `.venv/bin/python -m pytest tests/ -n auto --tb=short -q`: **639 passed, 17 warnings**, 371.21秒、watchdog exit=0
- `.venv/bin/ruff check src/leadlag tests tools/research/compute_gap_adjusted_distribution.py src/research/scripts/experiments/fix_overlay_metadata.py src/research/scripts/experiments/verify_overlay_models.py`: 成功
- `.venv/bin/python -m mypy src/leadlag`: **121 source files、問題なし**
- `.venv/bin/python -m compileall -q src/leadlag tests tools scripts src/research`: 成功
- `.venv/bin/lint-imports`: **4 contracts kept, 0 broken**
- `git diff --check`、`bash -n scripts/batch/run_gap_distribution.sh`: 成功
- 第5回の再現プローブ（bundle境界、schema cache、VaR gap版、既存F01〜F22/R〜U）: 成功

pytestの17 warningsは既存研究テストのゼロ除算・定数列相関です。今回の故障注入で意図的に出るOSErrorやrisk-stopログは失敗条件の証跡です。

## 未完了の運用作業

今回の4件はコードと回帰で閉じましたが、次の運用作業は別途必要です。

- `models/ml_order_overlay/phase2_8`は現在legacy layoutのため、本番loaderが安全側に拒否します。検証済みデータからversioned artifactを再生成し、`CURRENT`を公開する必要があります。
- 同じ入力・同じML artifactを使った本番とBTのweights照合、実約定・口数丸め後のexposure検証は未実施です。
- brokerの実応答・実市場データの全状態を本番接続で検証していません。今回のbroker確認はfixtureのみで、発注・取消・決済は行っていません。
