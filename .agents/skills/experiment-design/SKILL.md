---
name: experiment-design
description: 戦略改善実験の設計・実行・判定を行う。仮説定義→過学習ガード設計→バックテスト→ウォークフォワード検証→採用/不採用判定までの全体フローを管理する。新シグナル追加・モデル変更・パラメータ調整時に必ず参照すること。
---

# Experiment Design スキル

## 目的

日米リードラグ戦略の改善実験を、過学習リスクを管理しながら体系的に設計・実行・判定する。

## 前提

- `AGENTS.md` の不変条件（ルックアヘッド禁止・ベースライン分離・市場中立・ティッカー定義）を遵守
- `AGENTS.md` の「改善ワークフロー」および「過学習ガード」を前提とする
- 実験コードは本番パス（`src/leadlag/`）に入れず `scripts/experiments/`・`src/experiments/` に配置
- 本スキルは AGENTS.md に記載のない実行手順の詳細を補完する

## 実験ライフサイクル

```
1. 仮説定義 → 2. 過学習ガード設計 → 3. 実験スクリプト作成 → 4. バックテスト → 5. ウォークフォワード検証 → 6. 判定 → 7. 記録
```

### Step 1: 仮説定義

以下を明文化する。曖昧な仮説は実験の無駄遣いの原因になる:

- **解決する問題**: 現状の何が不足しているか（具体的数据・観測に基づく）
- **提案する変更**: どの関数・パラメータ・構造を変更するか
- **期待される効果**: どの指標がどれくらい改善するか（事前予測）
- **新パラメータ有無**: パラメータを追加する場合は過学習リスクが増大する旨を明記

### Step 2: 過学習ガード設計（必須）

`AGENTS.md` の過学習ガード要件を具体化する:

1. **試行回数カウント**: 過去の実験数（`archive/experiments/` 約30本 + `reports/` 配下の実験）を確認し、累積試行数を n_trials に設定
2. **パラメータ±摂動感度分析**: 新パラメータ追加時は ±20% 摂動（最低3水準: 0.8x / 1.0x / 1.2x）で Sharpe 変動を測定
3. **Deflated Sharpe Ratio**: Bailey & López de Prado (2014) により試行回数補正後の有意性を確認。閾値 DSR ≥ 0.95
4. **ウォークフォワード計画**: 年次ロール（2015-2026、12区間）で OOS 検証。purge=61日、embargo=5日

### Step 3: 実験スクリプト作成

- **配置**: `scripts/experiments/experiment_<name>.py`
- **実験用モジュール**: `src/experiments/<name>.py`（本番パス汚染回避）
- **config操作**: `copy.deepcopy(base_cfg)` を使用。`base_cfg.copy()` はネストdictの共有参照バグを引き起こす（AGENTS.md「config dictのshallow copy」参照）
- **タイムアウト**: 長時間実行を避けるため `python3 -c "..."` のインライン実行は禁止。スクリプトファイル経由で実行

### Step 4: バックテスト

`BacktestEngine.run_backtest()`（`src/leadlag/execution/backtester.py`）を使用:

```python
import copy, yaml
from leadlag.execution.backtester import BacktestEngine

with open("configs/production/production.yaml") as f:
    cfg_base = yaml.safe_load(f)

cfg = copy.deepcopy(cfg_base)
# 実験用の変更を適用
model = YourModel(cfg)
results = BacktestEngine.run_backtest(
    model, df_exec, start_date="2015-01-05",
    slippage_bps=5.0,
    overnight_alpha_long=0.75, overnight_alpha_short=0.5,
    buy_interest_annual=0.025, borrow_fee_annual=0.0115,
    reverse_fee_bps=2.0,
)
```

- **開始日**: 2015-01-05 以降（ベースライン期間 2010-2014 を分離）
- **コスト**: 片道5bps + 金利・貸株・逆日歩を含む net で評価
- **評価指標**: net Sharpe・最大DD・ターンオーバー・フォールバック発動率（`AGENTS.md` 評価指標約束事）

### Step 5: ウォークフォワード検証

先例: `scripts/experiments/experiment_a7_walkforward_dsr.py`

- **区間**: 2015-2026の年次ロール（12区間）
- **purge**: 61日（相関窓60 + 1日）、**embargo**: 5日
- **判定基準**:
  - 全区間で Sharpe > 0（負の区間が2個以内）
  - ベースライン対比で勝率 ≥ 8/12
  - 感度分析: ±20%摂動で Sharpe 変動 < 20%

### Step 6: 採用/不採用判定

#### 採用基準（すべて満たす必要あり）

1. Pooled net Sharpe が ベースラインを有意に上回る（改善幅 > 0.5 SE ≈ 0.01）
2. ウォークフォワード勝率 ≥ 8/12
3. DSR ≥ 0.95（試行回数補正後）
4. ±20% パラメータ摂動で Sharpe 変動 < 20%
5. フォールバック発動率が悪化しない
6. ターンオーバーが大幅に増加しない

#### 不採用の理由（いずれか該当）

- 改善幅が ノイズマージン内（+1% 未満等）
- ウォークフォワードでベースラインに負ける区間が多い
- 既存の正則化が十分に機能しており追加が冗長
- 構造的欠陥がありチューニングで埋められない

### Step 7: 記録

#### レポート

`reports/<experiment_name>/` に以下のフォーマットで保存（`/backtest-report` スキルと整合）:

```markdown
# <実験名>: Walk-Forward Validation Report

**Date**: YYYY-MM-DD
**Experiment**: <短い説明>
**References**: <論文・文献>
**Status**: **Adopted** / **Not adopted** — <理由>

---

## 1. Hypothesis
<解決する問題・提案する変更・期待される効果>

## 2. Methods
<実装内容・バリアント・検証フレームワーク>

## 3. Results
### 3.1 Pooled Performance
### 3.2 Walk-Forward Yearly Results
### 3.3 Deflated Sharpe Ratio
### 3.4 Sensitivity Analysis

## 4. Analysis
<なぜ採用/不採用か。構造的理由・数値的根拠>

## 5. Conclusion
<最終判定と理由>

## 6. Files
<実験コード・データ・プロットのパス>
```

#### 不採用実験のクリーンアップ（必須）

不採用の場合、**レポート以外のコード・データを破棄**する:

1. **レポートは残す**: `reports/<experiment_name>/` の markdown・CSV はそのまま保持（再検証防止用）
2. **実験コードを破棄**: `scripts/experiments/experiment_<name>.py`・`src/experiments/<name>.py` を削除
3. **本番パスの実験フックを削除**: `src/leadlag/` に追加した config オプション・関数・フラグがあれば元に戻す
4. **出力データを破棄**: `outputs/experiments/<name>/` 配下の中間データ・プロットは削除
5. **AGENTS.md に追記**: 「不採用実験の記録」セクションに1行サマリーを追記:

```markdown
- **<実験名>**（YYYY-MMM）: <1行サマリー>。理由: <構造的理由>。コードは破棄済み
```

レポートのみが再検証防止の証跡として残る。採用実験は `leadlag-fund-improvement` スキルの「採用実験の記録」に追記する。

## 注意事項

- **「Sharpe改善なし」の結論も価値がある**: 不採用実験も必ずレポート化し、再検証を防ぐ。ただしコードは破棄しレポートのみ残す
- **configのshallow copy禁止**: `base_cfg.copy()` はネストdictの共有参照バグを引き起こす。`copy.deepcopy` を使用
- **実験コードは本番パスに置かない**: `src/leadlag/` に実験コードを直接入れない
- **シャドー運用**: 採用判定後、`tools/validation/monitor_residual_blpx_shadow_performance.py` でライブ整合性を確認してから本番昇格
- **既存実験の確認**: 新規実験前に `archive/experiments/` と `reports/` で同一仮説の過去実験がないか確認する
