# 構造改善・残件対応報告

実施日: 2026-09-22

## 結論

前回レビューの2件を修正し、回帰テストを追加した。R1/R2のgap生成については、h=1/3/5の実データ再生成で日付別観測境界とlabel availabilityのbundle証跡を確認した。

S0–S8全体はまだ完了扱いにしない。実口座の読み取り照合は、設定された秘密鍵が現workspaceにないため実行不能だった。Hosted CIは`gh`がなくrun URLを取得できない。R5の現artifactは`train_start=2026-01-05`で、2015年からの本番学習期間および同artifactに紐づくOOS証跡を満たさないため、production configへの昇格は行っていない。

## 修正内容

### 1. ML学習のtimezone付き`df_exec`

`src/research/experiments/ml_overlay_training.py`で、学習入口とcollectorの両方が所有コピーをJST日付へ正規化するようにした。UTC/JSTのtimezone-aware index、naive indexを同じ取引日キーで扱い、正規化後の重複日付は拒否する。

### 2. 欠損realized target

collectorは`y_target`のNaN/Infを教師行として追加せず、銘柄単位で除外する。LightGBM fit入口にも有限性検査を追加し、欠損targetを0へ補完して学習する経路を閉じた。除外はログへ出力する。

### 3. gap生成の観測境界

`build_gap_historical_inputs()`が日付別にUS 09:00、JP gap/beta/TOPIX/open-to-09:10 09:10の観測境界を保持するようにした。h=1/3/5のbundle metadataへ、`observed_at`、`observed_at_source=session_boundary_contract`、`calculation_as_of`、`label_available_at`、履歴fingerprintを保存する。これはセッション境界の契約証跡であり、外部providerの実取得時刻を証明するものではない。

live bridgeにも履歴全日付の`observed_at_by_date`を渡すようにした。

### 4. scheduler登録経路

`scripts/batch/setup_scheduler_macos.sh`を5件（market data update、distribution diagnostics、decision、close、P&L report）の単一登録入口に修正した。各plistを現在のworkspaceから再生成するため、移動前の作業ディレクトリを残さない。旧`install_launchd.sh`もこの入口へ委譲する。実launchdへは自動審査により再登録を実行できなかったが、fake `launchctl`を使ったテンプレート生成検証で5件のload/unloadとworkspaceパスを確認した。

## 検証

- 追加回帰: `tests/unit/test_s7_ml_boundaries.py`（timezone付き`df_exec`、NaN target除外、fit拒否）、`tests/unit/test_s3b_research_boundaries.py`（gap観測境界）。
- gap再生成: `reports/20260922_structural_completion/gap_regeneration/`。2026-08-14/17、h=1/3/5の6 bundleを生成し、全ての必須来歴字段を確認。
- scheduler/read-only state: 保存台帳のrecovery candidate 0件。[acceptance_local.json](acceptance_local.json)。5件が登録済みだが、現ホストでworkspaceパスが一致するのは4件で、`update-market-data`は旧workspaceかつ直近終了コード127。`pnl_report`は候補なし以外の実行失敗を含むため、5件の運用成功とは扱わない。再登録経路のfake `launchctl`検証は[scheduler_template_test_final.log](scheduler_template_test_final.log)。
- 実口座読み取り: [account_snapshot.log](account_snapshot.log) はDNS制限、[account_snapshot_network.log](account_snapshot_network.log) は秘密鍵パス不存在で終了。発注・取消・再送は実行していない。
- Hosted CI相当: workflow定義は `.github/workflows/ci.yml` と整合。`gh`未導入のためrun URLは未取得。ローカルCIログは下記のファイルに保存。
- 全テスト: **852 passed / 17 warnings / 104.68秒**（[tests_final.log](tests_final.log)）。CI対象のRuff、`mypy src/leadlag`、compileall、docs、lock、clean wheel build・wheel smokeもPASS（[ruff_ci_final.log](ruff_ci_final.log)、[ruff_training_final.log](ruff_training_final.log)、[mypy_ci_final.log](mypy_ci_final.log)、[compile_final2.log](compile_final2.log)、[docs_final2.log](docs_final2.log)、[lock_ci_retry.log](lock_ci_retry.log)、[wheel_final.log](wheel_final.log)）。shell構文とschedulerテンプレート検証もPASS（[shell_final.log](shell_final.log)、[scheduler_template_test_final.log](scheduler_template_test_final.log)）。

## 残る受入条件

1. providerの実available_atと取得履歴を全入力で保存し、gap生成・live・BT・VaRの全入口で突合する。
2. broker秘密鍵を正しい環境へ配置したうえで、注文なしのwallet/positions/fills照合とpartial-fill recovery journalを取得する。
3. 2015年以降の固定macro/ADR/PIT入力と全日付の実09:10 `open_910_returns`でartifactを再生成し、同一artifactのwalk-forward OOS、本番/BT weights照合、promotion reviewを実施する。現cacheではこの入力が欠けるため、ゼロ補完や合成値での昇格は行わない。
4. GitHub Actions Hosted CIのrun URLと全job成功結果を取得する。
5. `bash scripts/batch/setup_scheduler_macos.sh`を運用担当者が明示承認の上で実行し、5件すべての`launchctl print`でworkspaceパスと直近終了コードを確認する。

これらが揃うまで、本番artifact昇格およびS0–S8全体完了は宣言しない。
