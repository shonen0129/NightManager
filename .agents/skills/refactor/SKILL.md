---
name: refactor
description: 重複排除・関数分割・責務整理を提案・実装する。日米リードラグ戦略コードのリファクタリング時に不変条件を維持し、テストを弱めないことを前提とする。モデル変更・パラメータ調整・コード整理時に必ず参照すること。
---

# Refactor スキル

## 目的

日米リードラグ戦略コードの重複排除・関数分割・責務整理を行い、技術的負債を解消する。

## 前提

- **不変条件を維持**: リーク防止・ベースライン分離・市場中立制約・ティッカー定義を崩さない
- **テストを弱めない**: リファクタリング後に全テストが通ること
- **振る舞いを保持**: リファクタリングは振る舞い変更を伴わない。シグナル・ポートフォリオ出力が一致することを確認
- **lint を通す**: リファクタリング後に `ruff check src/leadlag/ --select E,F,W --ignore E501` でエラーが出ないこと

## 実装前チェックリスト（必須）

以下を実装前に必ず実行する。飛ばすと過剰実装・誤実装の原因になる:

1. **要求が解決する問題が既存コードに存在するか確認**: `code_search` / `grep_search` で既存実装を検索。既に解決済みの場合は実装不要
2. **外部APIのパラメータマッピングは仕様書を先に読む**: `docs/api/` 配下の仕様書を必ず確認。推測でマッピングを書かない
3. **実動作が変わるか確認**: デフォルト値のみ使用されるパラメータ等、実動作に影響しない変更は固定値で済ませるか変更しない
4. **最小変更か確認**: 「動く」コードではなく「必要な」コードを書く。過剰実装は最終的にリバートされる

## リファクタリング手順

1. **対象特定**: 重複コード・長関数・責務混在・グローバル状態・未クリアキャッシュ等を特定
2. **影響分析**: 変更が及ぶファイル・関数を列挙。不変条件（リーク防止・ベースライン分離・市場中立・ティッカー定義）への影響を確認
3. **テスト確認**: リファクタリング前にテストが通ることを確認:
   - 並列（要 `pytest-xdist`）: `bash scripts/run_tests_parallel.sh`（約8分）
   - 直列（フォールバック）: `python3 -m pytest tests/ -v --timeout=300`（約22分）
4. **実装**: 最小限の変更で実装。1関数の責務が1つのことになるよう分割
5. **振る舞い検証**: リファクタリング前後で出力（シグナル・ポートフォリオ・net Sharpe）が一致することを確認（検証コードは下記参照）
6. **lint 実行**: `python3 -m ruff check src/leadlag/ --select E,F,W --ignore E501` でエラーが出ないこと
7. **テスト実行**: 手順3のコマンドで回帰確認
8. **レポート**: `reports/<sprint名>/` に変更内容・影響範囲・検証結果を記録

## 振る舞い一致の検証方法

`predict_signals` は dict を返すため、DataFrame の比較には `signals` キーを取り出す必要がある:

```python
import numpy as np

# リファクタリング前
before = model.predict_signals(df_exec)

# リファクタリング後
after = model.predict_signals(df_exec)

# シグナルの完全一致を検証
for key in ["signals", "raw_pca_signals", "residual_pca_signals",
            "raw_blpx_signals", "residual_blpx_signals"]:
    np.testing.assert_array_equal(
        before[key].values, after[key].values,
        err_msg=f"{key} mismatch after refactor"
    )
```

## 注意事項

- **実験コードは除外**: `scripts/experiments/`, `src/experiments/`, `archive/` はリファクタ対象外（本番パスのみ）
- **段階的実施**: 1回のリファクタリングで複数の技術的負債を同時に解消しない。1つずつ検証
- **`ComplianceAuditor` の監査項目を無効化しない**: リーク監査は維持
- **ARCHITECTURE.md の更新**: リファクタリング完了後、`docs/ARCHITECTURE.md` のリファクタリング履歴に追記
- **config の不変性を維持**: `ProductionV2RunConfig` の `frozen=True` 等、既存の不変性担保を崩さない
- **技術的負債の最新状況**: コードレビューレポート（`reports/code_review_*.md`）または AGENTS.md の「既知の落とし穴」セクションを参照
- **事前調査**: リファクタリング対象の特定には `code-review` スキルの使用を推奨。レビュー結果に基づいて優先度を決定する
