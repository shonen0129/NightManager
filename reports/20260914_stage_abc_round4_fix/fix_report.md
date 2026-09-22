# 段階A〜C 第4回指摘の修正記録 — 2026-09-14

前回レビューのU01〜U04を修正しました。実口座・外部broker API・本番データは使用していません。

## 修正内容

### U01: `.npy` bundleの世代混在

対象は `src/leadlag/utils/gap_matrix_io.py` です。

- μ・Ω・metadataを個別の一時ファイルへ書き、fsync後にatomic replaceするようにした。
- μ・Ω・metadataそれぞれのSHA-256と対象取引日を持つ隠しcommit manifest（`.mu_gap[_hN]_YYYYMMDD.bundle.json`）を最後に公開するようにした。
- 読込時は、実際に読み込んだbytesのdigestがmanifestと一致する場合だけmetadataを有効にする。不一致・manifest欠落・取引日不一致は来歴不明としてon-demandまたはflatへ進む。
- metadataなしで再保存した場合は旧metadataを削除し、manifestのmetadata digestも`null`にする。旧来歴を新しい行列へ引き継がない。
- batchの当日bundle確認にもmanifestを追加した。
- 回帰fixtureのdefault/h=3/h=5にもmanifestを追加した。

この契約により、μだけが新しくなった状態、Ωだけが旧い状態、旧metadataが残った状態はいずれも通常の監査PASSへ進まない。SQLiteのtransaction経路は従来のbundle APIを維持している。

### U02: VaRのoverlay版固定

対象は `src/leadlag/execution/var_history.py` と `src/leadlag/execution/v2_bridge.py` です。

- `get_hist_returns_for_risk`に選択済み`overlay_model`を渡せる引数を追加した。
- standaloneのVaR呼出しでは設定されたCURRENTを一度だけロードし、live V2 bridgeでは`ProductionRunner`が既に保持している同じobjectを渡す。
- cache keyはCURRENTを再読込みせず、選択済みartifactの`artifact_version`・`model_sha256`などから作る。
- cache miss時の`BacktestEngine.run_v2_backtest`にも同じobjectを渡す。
- model objectに完全なartifact digestがない場合はpickle digestへフォールバックし、metadataだけの衝突を避ける。

これで、key計算後にCURRENTがAからBへ切り替わっても、そのrunのBTはAを使い、Bの系列をAのkeyへ保存しない。

### U03: artifact検証ツールのskip

`src/research/scripts/experiments/verify_overlay_models.py`で、CURRENTがあるartifactは必ず共通loaderを通すようにした。active model欠落、版ディレクトリ欠落、pointer不正、digest・provenance不一致は`[fail]`かつ全体終了コード1になる。未作成の任意ディレクトリだけは従来どおり対象外とする。

### U04: 移行確認ツールの誤検出

`src/research/scripts/experiments/fix_overlay_metadata.py`の探索をartifact root単位へ変更した。rootを発見した後は内部の`versions/`と`.staging-*`へ再帰せず、walk-forwardのfold rootは引き続き検出する。正常versioned artifactをlegacyとして再学習対象にしない。

## 回帰と検証

追加した`tests/unit/test_stage_abc_followup_fixes.py`では、選択overlayによるVaR key分離、BTへの同一object伝播、検証ツールのactive model欠落、移行ツールのinternal versions除外を確認した。

`tests/unit/test_gap_matrix_io.py`には、旧manifestとのdigest不一致とmetadataなし再保存の回帰を追加した。

実行結果：

- 対象回帰：37 passed
- 回帰fixture／分布source：13 passed
- 全体：**620 passed、17 warnings、372.05秒**（watchdog exit=0）
- Ruff：変更対象と既存の指定範囲で成功
- mypy：`src/leadlag` 120 source filesで成功
- compileall：指定rootで成功
- import-linter：381 dependencies、4 contracts kept
- `git diff --check`、`bash -n scripts/batch/run_gap_distribution.sh`：成功

最終全体ログは[pytest_full_final.log](./pytest_full_final.log)、対象回帰ログは[対象ログ](./targeted_final.log)です。

研究テストの17 warnings（ゼロ除算・定数列相関）は既存の警告で、今回の修正失敗ではありません。

本番設定の`models/ml_order_overlay/phase2_8`は、実artifactがlegacy rootのためloaderが拒否します。再学習・versioned artifact公開と、本番／BTの同一入力・同一artifactによるweights照合は、今回の修正とは別の運用作業として未完了です。
