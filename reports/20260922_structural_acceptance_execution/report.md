# 本番artifact・実運用受入条件の実行結果

実行日: 2026-09-22（Asia/Tokyo）  
範囲: 既存artifactの検証、1日固定の読み取り専用BT smoke、scheduler登録、broker読み取り専用照会、Hosted CI確認。発注・取消・再送・手動decision/closeは実行していない。

## 判定

**S0–S8全体および本番artifact昇格は未受入。** 実行できた条件と、証拠不足または入力不足で止まった条件を分けると次のとおり。

| 条件 | 実行結果 | 根拠 |
|---|---|---|
| 現行production configでartifactをロード | **FAIL / blocked** | `models/ml_order_overlay/phase2_8` は `CURRENT` のないlegacy形式で、loaderが拒否 |
| 既存versioned artifactを検証ロード | **PASS** | `var/results/20260920_structural_completion/ml_overlay_retrained/CURRENT` (`20260919T211305249223Z-db9535d77b24`) を検証ロード |
| versioned artifactを明示したV2 bundle構築 | **PASS** | `ProductionV2Model` とoverlayの構築成功。ただしproduction configへの昇格証明ではない |
| 同artifactの固定1日BT smoke | **実行完了 / flat fallback** | 2026-08-14を実行。09:10入力が完全でなく、分布取得失敗を安全にflat化。評価用OOS合格とは扱わない |
| gap h=1/3/5 bundle存在 | **PASS（契約証跡のみ）** | 2026-08-14/17の6 bundle、観測境界・label availability・calculation_as_of・履歴fingerprintあり |
| provider実取得 `available_at` | **FAIL / 未証明** | 6 bundleすべて `provider=null`, `available_at=null`; session boundaryの記録だけ |
| scheduler登録 | **PASS（登録・workspace）** | setup scriptを実行し、5件すべて登録済み・workspaceは現行パス |
| scheduler直近終了コード | **保留** | 再登録直後の5件は `last_exit_code=null`。手動実行でdecision/closeを起動することは安全条件外のため行っていない |
| broker wallet/positions read-only | **PASS（current key override）** | Tachibana照会成功、positions 0件、wallet取得成功 |
| broker照会の通常設定 | **FAIL / 設定不整合** | `.env` の `TACHIBANA_PRIVATE_KEY_PATH` が旧workspaceを参照 |
| pending recovery | **PASS** | `--pending` のrecovery candidate 0件、未解決runなし |
| Hosted CI | **FAIL / 証跡なし** | `gh`なし。GitHub Actions APIの最新runも `total_count=0` |
| 2015年以降の同artifact OOS・本番/BT weights照合 | **未実行 / 入力不足** | 現artifactの学習期間は `2026-01-05`〜`2026-08-13`、metadataにOOS証跡なし |

## artifact実行

production configの正規入口 `build_v2_model_bundle()` は、legacy artifactを拒否した。

```
ValueError: Legacy root overlay artifact is not accepted: .../models/ml_order_overlay/phase2_8
```

版管理済みartifactは `load_overlay_model()` と、明示 `overlay_model_dir` を指定したV2 bundle構築まで成功した。metadataは `metadata_status=verified`、学習期間は `2026-01-05`〜`2026-08-13`、OOS項目は含まれていない。

同artifactを2026-08-14の1日BT smokeで実行した結果は、17銘柄・1日、`fallback=true`、gross/netとも0。実行ログには「strict h=1 cache resolution requires a complete finite explicit open_910_returns」とあり、これは安全なflat fallbackの確認であり、性能や本番/BT一致の合格証拠ではない。

## 入力の実在性

ローカルcacheの`df_exec`は2009-01-07〜2026-07-28、4159行だが、`jp_open_910_*`列は0本。5分足cacheも2026-03-03〜2026-08-06で、2026-08-14 09:10行を持たない。したがって、2015年以降の全日付について実09:10入力を埋めたartifact再生成はこの環境では完了できない。ゼロ補完・合成値での昇格は行っていない。

## scheduler / broker / CI

`bash scripts/batch/setup_scheduler_macos.sh` を実行し、次の5件を現workspaceから再登録した。

- `com.leadlag.update-market-data`
- `com.leadlag.distribution-diagnostics`
- `com.leadlag.decision`
- `com.leadlag.close`
- `com.leadlag.pnl_report`

再登録後の`launchctl print`では5件とも登録済みかつworkspaceパス一致。直近終了コードはまだ存在しない。decision/closeの手動起動は注文につながり得るため行っていない。

brokerは既存の読み取り専用`reconcile --account-snapshot`を使用した。現workspaceの鍵をコマンド単位で指定した場合、Tachibanaからwalletとpositionsを取得できた。秘密鍵内容はログ・報告へ出していない。通常の`.env`にある鍵パスは旧workspaceのため、運用環境変数または`.env`のパス更新が必要。

Hosted CIはローカル`gh`がなく、未認証GitHub Actions APIの最新runも0件だったため、全job成功URLを取得できない。

## 実行証跡

- [artifact_probe.log](artifact_probe.log) / [artifact_probe.json](artifact_probe.json)
- [artifact_backtest_smoke_after.log](artifact_backtest_smoke_after.log)
- [collect_acceptance_after_scheduler.log](collect_acceptance_after_scheduler.log)
- [acceptance_local.json](acceptance_local.json)
- [scheduler_setup.log](scheduler_setup.log)
- [account_snapshot_current_key.log](account_snapshot_current_key.log)
- [reconcile_pending.log](reconcile_pending.log)
- [hosted_ci_api.log](hosted_ci_api.log)
- [data_inventory.log](data_inventory.log)
- [intraday_inventory.log](intraday_inventory.log)

補助スクリプトは同ディレクトリに保存し、compileallを通過した。artifactはproduction configへ差し替えていない。

## 次の受入阻害要因

1. 実providerの`available_at`と取得履歴を全入力へ保存し、gap/live/BT/VaRで突合する。
2. `.env`または運用環境の秘密鍵パスを現workspaceへ直し、read-only照会を通常設定でも再実行する。
3. 2015年以降の実09:10入力、固定macro/ADR/PITを揃え、同一artifactのwalk-forward OOSと本番/BT weights照合を実施する。
4. Hosted CIのrun URLと全job成功結果を取得する。
5. scheduler 5件を本番入力で安全に一巡させ、直近終了コードを記録する（decision/closeは発注承認と注文なし条件を別途満たしてから）。
