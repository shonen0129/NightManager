---
name: test-gen
description: 関数・クラスに対してユニットテストを自動生成する。日米リードラグ戦略の不変条件（ルックアヘッド禁止・ベースライン期間分離・市場中立制約・ティッカー定義）を踏まえたテストを生成する。新関数・新シグナル・モデル変更時に必ず参照すること。
---

# Test Gen スキル

## 目的

日米リードラグ戦略コードの関数・クラスに対し、不変条件を担保するユニットテストを自動生成する。

## 前提

- テストフレームワーク: **pytest**
- テスト配置: `tests/unit/`（ユニット）、`tests/integration/`（統合）
- 既存テスト: unit 26本 + integration（`test_leakage_audit.py`, `test_production_residual_blpx.py` 等）
- **変更後必ず `python3 -m pytest tests/ -v` が通ること**

## テスト生成の観点

### 1. ルックアヘッド防止テスト

- ローリング統計が当日行を含まないことの検証
- PITビニングが未来情報を使用しないことの検証
- ベータ計算が strictly historical であることの検証
- 参考: `tests/integration/test_leakage_audit.py`

### 2. 境界値テスト

- 窓サイズ不足（行数 < min_periods）時の挙動
- 空DataFrame・単一行DataFrame
- NaN/Inf を含むデータでの挙動
- 全ゼロシグナルでのポートフォリオ構築

### 3. 不変条件テスト

- **ベースライン期間**: 事前分布・基準相関が2010-2014固定であること
- **バックテスト開始日**: 2015-01-05以降であること
- **市場中立制約**: net exposure ±0.05、gross ≤ 2.0（RuleD適用後）
- **ティッカー定義**: N_U=15, N_J=17, 計32次元

### 4. フォールバック階層テスト

- v2 → v1 → PCA-Ensemble のフォールバックが正しく発動すること
- フォールバック時のシグナル整合性（スケール・符号）
- フォールバック発動率の監視

### 5. 数値精度テスト

- 既知の入力に対する出力の再現性
- config dictのshallow copy問題（`copy.deepcopy` 使用の確認）
- グローバルキャッシュの汚染がないこと

### 6. Copula相関ブレンドのテスト

- **正定値性**: ブレンド後の相関行列が正定値であること
- **尾部依存**: νが小さいほど尾部依存が強いことの検証
- **特異行列フォールバック**: 特異行列入力でjitter・pseudo-inverseフォールバックが発動すること
- **ブレンド重み**: `copula_blend_weight=0`でPearsonと完全一致、`1.0`でcopulaと完全一致
- **動的ブレンド**: ストレス期（vol_ratio >= threshold）でcopula重みが増加すること
- **copula窓の当日行除外**: copula推定に当日行が含まれないこと（`correlation.py`）

### 7. MinVar weight最適化のテスト

- **α=0.0等価性**: `build_weights_minvar(alpha=0.0)`が`build_weights(weight_mode="signal")`と一致
- **α=1.0純最小分散**: α=1.0でシグナル情報が無視され、分散最小化weightになること
- **Long/Short合計**: ロング側合計=+1.0、ショート側合計=-1.0（正規化後）
- **特異Sigma_YY**: 特異行列入力でフォールバックが発動しNaNを出さないこと
- **バスケット内1銘柄**: 1銘柄のみのバスケットでweight=1.0になること
- **Omega_gap使用**: `production_v2.py`で`minvar_enabled=True`時に`Omega_gap`がweight構築に使用されること

### 8. Macro Confidenceのテスト

- **サプライズゼロ**: マクロサプライズが全てゼロの時、scale=1.0でシグナル無変化
- **マクロデータ欠損**: yfinance取得失敗時に`macro_confidence_enabled=False`にフォールバック
- **kappaスケーリング**: kappaが大きいほどシグナルが縮小されること
- **方向保存**: スケーリング後にシグナルの符号（ロング/ショート判定）が変わらないこと

## テストテンプレート

```python
import pytest
import numpy as np
import pandas as pd
import copy

# テスト対象のインポート
# from src.leadlag.<module> import <function>


class Test<FunctionName>:
    """<関数名>のユニットテスト"""

    def test_normal_case(self):
        """正常系: 標準的な入力での出力検証"""
        # 入力データの準備
        # 期待値の計算
        # assert で検証
        pass

    def test_boundary_min_periods(self):
        """境界値: min_periods不足時の挙動"""
        pass

    def test_nan_input(self):
        """異常系: NaNを含む入力"""
        pass

    def test_no_lookahead(self):
        """不変条件: ルックアヘッドがないこと"""
        # 当日行が統計計算に含まれないことを検証
        pass

    def test_baseline_period_isolation(self):
        """不変条件: ベースライン期間が2010-2014固定"""
        pass
```

## 実行手順

1. 対象関数・クラスのコードを読み込み、入出力・副作用を把握
2. 上記8観点に従いテストケースを設計
3. `tests/unit/` または `tests/integration/` にテストファイルを作成
4. `python3 -m pytest tests/<新規ファイル> -v` で個別実行し通過を確認
5. `python3 -m pytest tests/ -v` で全体回帰を確認

## 注意事項

- **既存テストを弱めない**: 新規テスト追加時に既存テストの assertion を緩めない
- **モックは最小限**: 実際のデータパスに近い形でテストする。モック多用はリーク検出漏れの原因
- **フィクスチャの再利用**: `tests/conftest.py`, `tests/fixtures/` の既存フィクスチャを優先使用
- **configのdeepcopy**: 比較実験のテストでは `copy.deepcopy` を使用しshallow copy問題を回避
