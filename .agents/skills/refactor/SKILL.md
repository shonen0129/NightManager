---
name: refactor
description: 重複排除・関数分割・責務整理を提案・実装する。日米リードラグ戦略コードの既知の技術的負債（モンキーパッチ・グローバルキャッシュ・shallow copy問題等）の解消を含む。リファクタリング時は不変条件を維持し、テストを弱めないことを前提とする。
---

# Refactor スキル

## 目的

日米リードラグ戦略コードの重複排除・関数分割・責務整理を行い、既知の技術的負債を解消する。

## 前提

- **不変条件を維持**: リーク防止・ベースライン分離・市場中立制約・ティッカー定義を崩さない
- **テストを弱めない**: リファクタリング後に `python3 -m pytest tests/ -v` が通ること
- **振る舞いを保持**: リファクタリングは振る舞い変更を伴わない。シグナル・ポートフォリオ出力が一致することを確認

## 既知の技術的負債（優先度順）

### P1: `sre.py` のモンキーパッチ

- **問題**: `predict_signals` 内で `signals.build_c0_from_v0` をグローバル差し替え（P4シグナル計算）。スレッド非安全
- **方針**: `compute_signal` に `c0_override` 引数を追加し、グローバル差し替えを除去
- **影響範囲**: `src/leadlag/models/sre.py`, `src/leadlag/core/signal.py`
- **検証**: モンキーパッチ除去前後でシグナル出力が一致することを確認

### P2: グローバルキャッシュの汚染リスク

- **問題**: `_PRODUCTION_SIGNAL_CACHE` 等のキーにデータ識別子が無い。`predict_signals` 経由以外で呼ぶと汚染
- **問題**: `_COMMON_INPUTS_CACHE` は clear されないためメモリ単調増加
- **方針**: キーにデータ識別子（ハッシュ等）を含める。またはキャッシュをクラスインスタンス化してスコープ管理
- **影響範囲**: `src/leadlag/models/production_v2.py`, `src/leadlag/models/sre.py`

### P3: config dictのshallow copy問題

- **問題**: `base_cfg.copy()` がネストした dict を共有参照する。比較実験で両モデルが同一設定になる
- **方針**: `copy.deepcopy(base_cfg)` を徹底。または config の不変化（frozen dataclass等）を検討
- **影響範囲**: 実験スクリプト全般、`src/leadlag/cli.py`

### P4: 金利コスト日割り

- **問題**: `backtester.py` が `annual/365` を営業日課金 → 週末分が過小
- **方針**: 暦日ベースの課金に変更。オーバーナイト保有の実験で影響確認
- **影響範囲**: `src/leadlag/execution/backtester.py`

## リファクタリング手順

1. **対象特定**: 重複コード・長関数・責務混在を特定
2. **影響分析**: 変更が及ぶファイル・関数を列挙。上記技術的負債との関連を確認
3. **テスト確認**: リファクタリング前に `python3 -m pytest tests/ -v` が通ることを確認
3. **実装**: 最小限の変更で実装。1関数の責務が1つのことになるよう分割
4. **振る舞い検証**: リファクタリング前後でバックテスト出力（シグナル・ポートフォリオ・net Sharpe）が一致することを確認
5. **テスト実行**: `python3 -m pytest tests/ -v` で回帰確認
6. **レポート**: `reports/<sprint名>/` に変更内容・影響範囲・検証結果を記録

## 振る舞い一致の検証方法

```python
# リファクタリング前のシグナルを保存
import json
before_signals = model.predict_signals(df_exec)
with open("/tmp/before_signals.json", "w") as f:
    json.dump(before_signals.to_dict(), f)

# リファクタリング後のシグナルと比較
after_signals = model.predict_signals(df_exec)
assert before_signals.equals(after_signals), "Signal mismatch after refactor"
```

## 注意事項

- **実験コードは除外**: `scripts/experiments/`, `src/experiments/`, `archive/` はリファクタ対象外（本番パスのみ）
- **段階的実施**: 1回のリファクタリングで複数の技術的負債を同時に解消しない。1つずつ検証
- **`ComplianceAuditor` の監査項目を無効化しない**: リーク監査は維持
- **ARCHITECTURE.md の更新**: リファクタリング完了後、`docs/ARCHITECTURE.md` のリファクタリング履歴に追記
