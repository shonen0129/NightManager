# S3b完了確認・S3c実施報告

- 実施日: 2026-09-16
- 対象: `reports/20260915_structural_improvement/plan.md` の S3b/S3c
- 判定: S3bは計算・診断出力サブ範囲をPASS、S3cはbundle/cache契約サブ範囲をPASS

## S3bの完了確認

`reports/20260916_structural_improvement_s3a_s3b/report.md` と対象コードを再確認した。
`leadlag.pipeline.gap_distribution.compute_gap_distribution`がraw/gapのμ・Ωと分母floorを純粋計算し、
`leadlag.pipeline.gap_reporting`が安定した診断DataFrame/CSVを出力する境界は維持されている。
Step 1の`Omega_struct`はh=1の明示入力として扱われ、on-demandの共分散を暗黙に流用しない。

一方、research固有の入力取得の組立、plot生成、portfolio評価、`report.md`生成はresearch scriptに残る。
このため、S3b全体ではなく「計算・診断出力」サブ範囲の完了と判定する。

## S3cで実施した変更

### Gap bundle

- `src/leadlag/domain/gap_bundle.py` に immutableな`GapBundleRef`を追加した。
- μ・Ω・metadataのSHA-256、trade date、horizon、storage format、schema/input/model/config version、ticker順を
  一つのmanifestで識別する。旧`format_version=1`の三 digest manifestも読み込める。
- `GapStore.save_horizon`はμ・Ω・metadata・manifestを一つのSQLite transactionで保存し、
  `load_horizon_bundle`は一つのsnapshotから読み出す。従来の`load_horizon`三値APIは互換のため維持した。
- NPY互換経路ではmanifestをcommit markerとして最後に公開し、読み込み時に実payloadのdigest、日付、horizon、formatを検証する。

### VaR/ES cache

- `src/leadlag/execution/var_cache.py` に`VaRCacheIdentity`と`DeadlineBudget`を追加した。
- cache keyはeffective config、`df_exec` hash、code hash、選択中overlay、gap入力版、開始日、slippageを含む。
  既存の`daily_returns:<16hex>`形式は維持した。
- `var_history.py`は一つのmonotonic絶対期限を準備・copy・worker計算へ渡し、期限切れ後の結果を採用しない。
  既存`run_with_timeout`のdaemon workerが裏で継続し得る限界とsnapshot keepaliveは維持する。

## 検証

対象契約・既存回帰テスト:

```text
61 passed in 12.37s
```

対象は`test_s3c_cache_contracts.py`、gap distribution/reporting、S3a source、V2 distribution、gap matrix I/O、
GapStore、Stage A-C follow-up fixes。

変更対象のruff、mypy、compileallはすべてPASSした。

その後、legacy `GapStore.save` による旧manifest無効化の回帰を追加し、影響範囲31件を0.40秒で再実行してPASSした。

全`tests/`を外側の1800秒 watchdog付きで再実行し、`688 passed, 18 warnings in 711.84s`だった。
warningsは既存のresearch診断での`RuntimeWarning`とmultiprocessing forkの`DeprecationWarning`で、失敗はない。
全対象ディレクトリ（`src/leadlag tests tools scripts src/research`）の`compileall`もPASSした。

## 残件

- S3b: research固有の入力取得・plot・portfolio評価・report生成の分離（後続実施で完了。詳細は[S3b全体完了報告](../20260916_structural_improvement_s3b/report.md)）。
- S3c: VaRのsnapshot/worker lifecycle全体の責務分割、legacy readerの撤去、実データでのcache/on-demand同値性比較。
- ML artifactの再生成と本番/BT weights照合は、元レビューからの運用残件として継続する。
