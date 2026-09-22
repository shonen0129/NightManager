# 第3回レビュー指摘の修正結果

実施日: 2026-09-21

## 修正内容

### M1: 補助入力の観測時刻

`HistoricalInputs.observed_at_for()`を、共通mappingを基礎に日付別mappingでキー単位に上書きする実装へ変更した。これにより、日付別mappingに含まれない共通入力の観測時刻が検査から消えない。共通のみ・日付別のみ・異なるキーの併用を回帰テストで確認した。

### M2: ADRのtimezone-aware学習入力

ADR adapterにJST日付正規化処理を集約し、日付指定なしの全期間loaderも同じ正規化を行うようにした。学習入口でもrun-owned/injected frameを期間抽出前に正規化し、timezone-awareな`train_start`/`train_end`とartifact metadataもJST日付へ統一した。推論と学習で同じ日付キーを使うことを回帰テストで確認した。

## 検証

- 対象回帰: 62 passed
- 全テスト: **848 passed / 17 warnings**（外側watchdog）
- mypy: 147 source files, PASS
- Ruff: CI対象, PASS
- compileall: PASS
- import-linter: 7 kept / 0 broken
- wheel隔離検証: clean build、古いbuild/egg-info排除、manifest、隔離install/smokeをPASS

実運用のavailable_at証跡、実scheduler・brokerとの照合、verified本番ML artifactのOOS評価と本番/BT比較、Hosted CIの成功結果は従来どおり別の受入残件である。
