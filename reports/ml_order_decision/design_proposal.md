# ML 発注判定オーバーレイ 設計書

**文書 ID**: `ML-ORDER-DECISION-2026-07`
**作成日**: 2026-07-29
**対象**: 日米リードラグ戦略 V2 本番パイプライン
**ステータス**: 設計案（未実装）

---

## 1. 背景と目的

### 1.1 背景

ユーザーから以下の改善案が提案された：

> 「観測寄付（当日の寄付価格）、JP 17 銘柄名、各銘柄のシグナルを入力として、機械学習により発注すべきかどうかを判定する」

現行の V2 本番パイプライン (`src/leadlag/models/production_v2.py`) では、`mu_gap / sigma_gap` のスコアランキングと `RuleD` 動的グロス調整により、ロング・ショート各上位銘柄が選択され、市場中立ポートフォリオが構築される。寄付時点の gap は信号生成 (`_apply_gap_adjustment`) および `mu_gap` / `Omega_gap` の計算に既に組み込まれているが、**銘柄ごとに「発注すべきか」を個別に判定する層は存在しない**。

### 1.2 目的

本設計書は、上記改善案を現行アーキテクチャに統合可能な形に再構成する。特に、以下の観点を満たすことを目的とする：

- 市場中立性・総リスク制約（gross ≤ 2.0、net ±0.05）を保持
- ルックアヘッドリークを排除
- gap データ欠損時のフォールバックを保持
- 過学習リスクを管理
- 既存の `production_v2` パイプラインへの最小変更で統合可能

---

## 2. 要求定義

### 2.1 機能要件

- **入力**: 観測寄付（`jp_gap_*`）、JP 17 銘柄名、各銘柄のシグナル（`mu_gap` / `sigma_gap` スコア）、追加の市場・銘柄統計
- **出力**: 各銘柄について「発注に値するか」の確信度 `p_trade`、およびそれを反映した調整済みスコア
- **統合**: 既存の `mu_over_sigma` 選定・`solve_baseline_style` / `build_weights_minvar` ウェイト構築・`RuleD` グロス調整の前段に挿入
- **フォールバック**: モデル未使用時・片側全スキップ時・gap データ欠損時はフラットポジション（`w_final = 0`）

### 2.2 非機能要件

- 本番実行 (`run_decision_v2.sh` → `v2_bridge.py` → `execute_post_decision_flow`) の変更は最小限に留める
- バックテスト・ウォークフォワード検証で `BacktestEngine` を使用
- テストは `tests/unit/` に追加し、既存テストを弱めない

### 2.3 非対象

- 米国 ETF 15 銘柄（対象は JP 17 銘柄のみ）
- LOB（板情報）を用いた limit order 執行最適化（既存 `NetScoreRankingLob` / `apply_hard_rules` の領域）
- V2 シグナル生成モデル（BLPX）自体の置き換え

---

## 3. 現行実装との関係

### 3.1 関連コンポーネント

| コンポーネント | ファイル | 役割 |
|---|---|---|
| V2 本番ポートフォリオ生成 | `src/leadlag/models/production_v2.py` | `mu_gap / sigma_gap` スコア → 上位銘柄選択 → ウェイト構築 |
| Gap 調整分布計算 | `tools/production/compute_gap_adjusted_distribution.py` | `mu_gap` / `Omega_gap` を計算・保存 |
| Gap 調整信号 | `src/leadlag/models/blp_base.py` | グローバル `gap_open_coef` / `topix_beta_coef` による gap 補正 |
| Gap 耐性フィルタ | `src/leadlag/core/signal.py:383` | シグナルと gap から limit order 執行可否を判定（現状本番未使用） |
| LOB ハードルール | `src/leadlag/execution/microstructure/execution_constraints.py` | LOB スナップショットに基づく発注スキップ/スケール |
| 注文送信 | `src/leadlag/execution/helpers.py` | `execute_post_decision_flow` / `submit_orders_via_api`（market order） |
| V2 ブリッジ | `src/leadlag/execution/v2_bridge.py` | V2 結果を既存ブローカー層に接続 |

### 3.2 現行の gap 処理

`preprocess_data` (`src/leadlag/data/preprocessor.py:219`) では `jp_gap_* = 当日寄付 / 前日大引け - 1` が計算される。`compute_gap_adjusted_distribution` ではこの gap を用いて `mu_gap` / `Omega_gap` が生成され、`production_v2` はその `mu_gap / sigma_gap` をスコアとして銘柄を選択する。

つまり、**gap は既に信号生成に組み込まれている**が、銘柄別の反応はグローバル係数（`gap_open_coef=0.70`, `topix_beta_coef=0.6`）で一括処理されており、銘柄ごとの非線形・非対称反応は学習されていない。

> **情報の二重カウントに関する注意**: `mu_gap` は `mu_raw` に gap 調整を施した結果であり、`score = mu_gap / sigma_gap` にも gap 情報が織り込まれている。ML モデルに生の `gap` を特徴量として再度入力すると、同じ gap 情報が2つの形で入力される。モデルが学習すべき対象は「既存のグローバル gap 調整が各銘柄にとって強すぎる/弱すぎるか」の**残差**であり、gap そのものの予測ではない。この関係を特徴量設計時に明示的に意識する必要がある。

---

## 4. レビューで指摘された課題

コードレビューおよび edge-case-finder スキルによる洗い出しの結果、以下の課題が確認された：

### P1: 既存実装との重複・競合

- `NetScoreRankingLob` は既に銘柄 × シグナル × LOB で発注可否を判定しているが、gap は直接使っていない。
- `apply_gap_tolerant_filter` は既にシグナルと gap から limit order 執行可否を判定しているが、現状本番パイプラインから呼ばれていない。
- 本改善案は `production_v2` の選定層に差し込む形で設計する必要がある。

### P1: 市場中立性の崩壊リスク

- 個別銘柄をスキップすると、ロング・ショートの銘柄数とドルニュートラリティが崩れる。
- `NetScoreRankingLob` の `restore_dollar_neutrality_array` は片側が 0 になると全ウェイトをゼロにするだけで、途中の不均衡を修復しない。
- 現在の本番パス `execute_post_decision_flow` には「一部銘柄だけ除外された後の再調整」がない。

### P1: gap の定義・ルックアヘッド

- `gap` が `jp_gap_*`（寄付リターン、9:10 時点で既知）なら使用可能。
- `jp_oc_*`（大引けから大引け）や当日の終値を使うと目標リークになり即アウト。
- バックテストでは `apply_gap_tolerant_filter` が `jp_open_t1`（当日寄付）を使うが、本番執行仮定（9:10 market order）との乖離に注意。

### P1: フォールバック設計

- 全銘柄または片側がスキップされた場合、フラットポジションを返す必要がある。
- `production_v2` は gap データ欠損で既に flat するが、新判定による「reject 全般」の fallback は別途設計が必要。
- 前日 gap 行列のコピー fallback は禁止（`AGENTS.md` 6 項）。

### P2: 過学習・評価

- 新しいパラメータ（gap 閾値、score 閾値、銘柄別係数）を追加する場合、walk-forward OOS と Deflated Sharpe が必要。
- 既に `git tag archive-2026-08` の `archive/experiments/` に多数の実験があるため、同一ヒストリー上の繰り返し選択に要注意。

### P2: テスト・監査

- `apply_gap_tolerant_filter` にプロダクションテストが見当たらない。
- 市場中立維持、全スキップ時フラット、`ComplianceAuditor` との整合をテストで担保する必要がある。

---

## 5. 設計方針

### 5.1 推奨案：「スコア再重み付けオーバーレイ」

**新しい銘柄選択層を作らず、既存の V2 スコア（`mu_gap / sigma_gap`）に機械学習モデルが出力した確信度 `p_trade` を乗算してから既存の選定・ウェイト構築に渡す**ことを推奨する。

これにより：

- `solve_baseline_style` / `build_weights_minvar` による市場中立性・総リスク制約を保持できる
- `production_v2.py` の変更は `scores` 計算直後に差し込む形に留まる
- モデル不在時・全スキップ時のフォールバックが明確になる

### 5.2 なぜバイナリ skip フィルタにしないか

個別銘柄を単純に 0/1 でスキップする案もあるが、以下の理由から推奨しない：

- スキップ後の再正規化でノイズが増幅しやすい
- 片側に 1 銘柄しか残らなかった場合のフォールバック処理が複雑になる
- 閾値チューニングの過学習リスクが高い

### 5.3 機械学習モデルの役割

現行の `mu_gap` は既に gap 調整済みの期待リターンである。ML モデルの役割は：

- **銘柄別の gap 反応を学習**する（現行はグローバル `gap_open_coef`）
- **シグナルと gap の交互作用**を捕捉する（同じ gap でもシグナル強度で反応が異なる）
- **発注に値しないケースを学習**する（例： gap が大きすぎて期待リターンがコストに見合わない）

---

## 6. 全体アーキテクチャ

### 6.1 データフロー

```
[既存] US ETF リターン + JP 前日終値/当日寄付
    ↓
[既存] preprocess_data → df_exec (jp_gap_*, jp_oc_*, ...)
    ↓
[既存] compute_gap_adjusted_distribution → mu_gap, Omega_gap
    ↓
[既存] production_v2.generate_v2_production_portfolio
    ├─ sigma_gap = sqrt(diag(Omega_gap))
    ├─ scores = mu_gap / sigma_gap
    ↓  ← ここに ML オーバーレイを差し込む
[新規] ML 発注判定モデル
    ├─ 特徴量: ticker, score, gap, gap_idio, sigma_gap, 交互作用, 市場・銘柄統計
    ├─ 出力: p_trade_i ∈ [0,1] (各銘柄)
    └─ score_adjusted_i = score_i * g(p_trade_i)
    ↓
[既存] solve_baseline_style / build_weights_minvar
    ↓
[既存] RuleD グロス調整 → w_final
    ↓
[既存] v2_bridge.py → execute_post_decision_flow → submit_orders_via_api
```

### 6.2 変更ファイル（Phase 1 実験）

| パス | 内容 |
|---|---|
| `scripts/experiments/experiment_ml_order_decision.py` | 実験エントリーポイント |
| `src/experiments/ml_order_decision/` | 実験用モジュール（本番パス汚染回避） |
| `reports/ml_order_decision/` | 検証レポート |

### 6.3 変更ファイル（本番統合・Phase 2 以降）

| パス | 内容 |
|---|---|
| `src/leadlag/models/production_v2.py` | `scores` 計算直後に `ml_overlay` を差し込む（config フラグ付き） |
| `configs/production/production.yaml` | `ml_order_decision.enabled`, `model_path`, `threshold` 等を追加 |
| `tests/unit/test_ml_order_decision.py` | 単体テスト |

---

## 7. データ定義

### 7.1 特徴量

以下の特徴量を **9:10 時点で確定する情報のみ**から生成する。

| 特徴量 | 定義 | 取得元 |
|---|---|---|
| `ticker` | JP 17 銘柄（one-hot または categorical） | `JP_TICKERS` |
| `score` | `mu_gap / sigma_gap` | `production_v2.py` 計算結果 |
| `mu_gap` | gap 調整済み期待リターン | `mu_gap_*.npy` |
| `sigma_gap` | `sqrt(diag(Omega_gap))` | `Omega_gap_*.npy` |
| `gap` | `jp_gap_*`（前日大引け → 当日寄付） | `df_exec` |
| `gap_idio` | `gap - beta_i * topix_night` | `df_exec` + `jp_beta_*` |
| `score × gap` | 交互作用項 | 上記から導出 |
| `score × gap_idio` | 交互作用項 | 上記から導出 |
| `abs(score)` | シグナル強度 | 上記から導出 |
| `abs(gap)` | gap 大きさ | 上記から導出 |
| `topix_night` | TOPIX 前日大引け → 当日寄付 | `topix_night_return` |
| `regime` | RuleD PIT ビン（Low/Mid/High） | `production_v2.py` |
| `market_vol_20d` | 最近 20 日の |r_oc| 平均 | `df_exec`（PIT） |
| `ticker_hit_rate_63d` | 銘柄別 63 日 hit rate | 学習用に PIT で計算 |
| `ticker_mean_ret_63d` | 銘柄別 63 日平均リターン | 学習用に PIT で計算 |
| `ticker_vol_63d` | 銘柄別 63 日ボラ | 学習用に PIT で計算 |

### 7.2 ラベル

#### 課題：個別ヒット率 vs ポートフォリオ寄与

単純な個別ポジションの収益性（`side_i * realized_i - cost_i > 0`）でラベル付けすると、**分散寄与でポートフォリオ Sharpe を押し上げる銘柄をスキップしてしまう**リスクがある。本戦略のエッジはポートフォリオ全体の分散効果から来るもので、単独で赤字のポジションでもポートフォリオに貢献し得る。

#### 推奨ラベル：ポートフォリオ寄与シャープ（回帰タスク）

```text
side_i      = sign(score_i)
realized_i  = 9:10 → 大引けリターン（compute_jp_target_returns 相当）
cost_i      = 往復コスト（片道 5 bps × 2 = 10 bps を初期近似）
w_baseline  = solve_baseline_style(scores) によるベースラインウェイト

# 個別ポジションのポートフォリオ寄与（連続値）
contribution_i = w_baseline_i * (side_i * realized_i - cost_i)

# 回帰タスク：contribution_i を直接予測する
y_i = contribution_i
```

- **回帰タスク**を採用し、分類の閾値問題を回避する。
- `contribution_i` が正の銘柄は発注に値し、負の銘柄はスキップすべき、という連続的な判断が可能。
- ラベルは **学習時のみ** 使用し、推論時は使用しない。
- `realized_i` は `jp_oc_*` を使用。5 分足 09:10 バーが存在する期間は `compute_jp_target_returns` の調整を適用する。
- 本番当日の行は `r_oc` が `NaN`（provisional）のため、学習ラベルには使用できない。推論には影響しない。

#### 代替ラベル（比較用）

- **個別ヒット率（分類）**: `y_i = 1 if side_i * realized_i - cost_i > 0` — シンプルだが分散寄与を無視する
- **シャープ寄与**: `contribution_i / sigma_portfolio` — ボラティリティ正規化版

### 7.3 データソース

- `live/pipeline_data/gap_adjusted_distribution/<timestamp>/matrices/mu_gap_*.npy`
- `live/pipeline_data/gap_adjusted_distribution/<timestamp>/matrices/omega_gap_*.npy`
- `df_exec`（`preprocess_data` の出力）
- `results/` または `live/pipeline_data/diagnostics_weights/` の過去ウェイト（検証用）

---

## 8. モデル設計

### 8.1 Phase 1: 銘柄別 gap 係数の回帰推定（推奨スタート）

現行のグローバル `gap_open_coef` を銘柄別にデータから推定する、最もシンプルな回帰モデルから始める。

```text
# 目標変数: ポートフォリオ寄与 contribution_i（§7.2 参照）
# モデル: OLS / Ridge 回帰

contribution_i ≈ α_ticker
  + β1_ticker * score_i
  + β2_ticker * gap_i
  + β3_ticker * score_i * gap_i
  + β4 * gap_idio_i
  + β5 * sigma_gap_i
  + β6 * topix_night
  + β7 * market_vol_20d
```

- `ticker` との交互作用で銘柄別の gap 反応係数を学習する。
- **回帰タスク**なので分類閾値の問題がなく、パラメータ数も少ない。
- Ridge 回帰（L2 正則化）で過学習を抑える。
- モデル出力 `contribution_hat_i` を正規化して `p_trade_i` に変換：
  ```python
  p_trade_i = sigmoid(contribution_hat_i / scale)  # scale は訓練データの std
  ```

### 8.2 Phase 2: ロジスティック回帰 / LightGBM（Phase 1 で不十分な場合）

Phase 1 の回帰で残差構造が残る場合、非線形効果を捕捉するために拡張する。

#### Phase 2a: ロジスティック回帰（分類）

```text
p_trade = sigmoid(
    β0_ticker
  + β1 * score
  + β2 * gap
  + β3_ticker * score * gap
  + β4 * gap_idio
  + β5 * sigma_gap
  + β6 * topix_night
  + β7 * market_vol_20d
)
```

- `class_weight='balanced'` でクラス不均衡に対処。

#### Phase 2b: 小さな LightGBM

- `num_leaves=7`
- `n_estimators=50`
- `min_data_in_leaf=100`
- `ticker` を categorical として扱い、銘柄別の分割を自動学習。

### 8.3 スコア変換

ML モデルは `p_trade` を出力し、以下の変換で既存スコアに乗算する：

```python
# 案 1: 確信度で線形に補正
score_adjusted = score * (2 * p_trade - 1)

# 案 2: 確信度が高いものだけ残す
score_adjusted = score * (p_trade > threshold).astype(float)

# 案 3: 連続値でスケーリング
score_adjusted = score * p_trade
```

**推奨は案 3**（連続値）。バイナリ判定よりも滑らかで、既存の選定層との整合性が高い。

> **事前診断（必須）**: 案3では `p_trade` に銘柄間の**相対的なばらつき**が必要である。`solve_baseline_style` は各サイドを `baseline_gross / 2` に正規化するため、`p_trade` が全銘柄で一様に 0.5–0.7 のように縮小してもウェイトは変わらない（相対比のみが意味を持つ）。モデルが意味を出すには、銘柄間で `p_trade` に十分な分散がある必要がある。学習後に `p_trade` の銘柄間分散（CV など）を確認し、分散が小さい場合は案1（`2*p_trade - 1` で中心化）または案2（閾値でバイナリ）への切り替えを検討する。

---

## 9. 学習・検証

### 9.1 時系列 split

| 期間 | 用途 |
|---|---|
| 2010-2014 | ベースライン期間（分離・使用禁止） |
| 2015-01-05 〜 2020-12-31 | Train |
| 2021-01-01 〜 2022-12-31 | Validation |
| 2023-01-01 〜 2026-06-14 | Test / backtest |

- `purge=61` 日、`embargo=5` 日（`experiment-design` スキル準拠）。
- ウォークフォワードでは年次ロール（2015-2026、12 区間）で再学習。

> **前提条件: 5分足データカバレッジ確認**: `compute_jp_target_returns` は5分足 09:10 バーでラベルを調整するが、5分足データが2015〜2020に完全に存在するかは未確認。存在しない期間は `jp_oc_*`（Open-to-Close）にフォールバックされ、**ラベルの定義が期間によって不一致**になる。学習前に5分足データのカバレッジを確認し、不一致期間は学習から除外するか、ラベル定義を統一すること。

### 9.2 再学習頻度

- 実験段階では年次再学習で十分。
- 本番統合後は月次再学習を検討。再学習は過去データのみ使用し、当日行は含まない。

### 9.3 評価指標

#### オフライン

- AUC、キャリブレーション曲線
- 銘柄別 hit rate
- 特徴量重要度（`score`, `gap`, 交互作用の寄与）

#### バックテスト

- **net Sharpe**（コスト後）
- 最大 DD
- ターンオーバー
- フォールバック発動率
- ロング/ショート銘柄数の分布

#### 統計検定

- ベースライン V2 との paired t-test
- **Deflated Sharpe Ratio**（試行回数補正後 DSR ≥ 0.95）

### 9.4 感度分析

- 新パラメータ（閾値、score 乗数）の ±20% 摂動で Sharpe 変動 < 20% を確認。
- `p_trade` の変換方法（案 1/2/3）の感度を比較。

---

## 10. 本番統合

### 10.1 変更点

`production_v2.py` の `scores` 計算直後に以下を差し込む：

```python
if run_cfg.ml_order_decision_enabled:
    from leadlag.models.ml_order_decision import apply_ml_overlay
    scores = apply_ml_overlay(
        scores=scores,
        mu_gap=mu_gap,
        sigma_gap=sigma_gap,
        gap=gap_vec,  # jp_gap_*
        topix_night=topix_night_t,
        tickers=JP_TICKERS,
        model_path=run_cfg.ml_model_path,
    )
```

- `run_cfg` に `ml_order_decision_enabled`（default: False）と `ml_model_path` を追加。
- モデルファイルは `live/pipeline_data/ml_models/` に保存。

### 10.2 フォールバック

| 条件 | 挙動 |
|---|---|
| `ml_order_decision_enabled = False` | 既存動作（ML なし） |
| モデルファイル不在・ロード失敗 | 警告ログを出して既存動作（fail-safe） |
| `gap` データ欠損（`mu_gap` / `Omega_gap` なし） | 既存動作：フラットポジション |
| 片側全スキップ | `w_final = 0`（フラット） |

### 10.3 ログ

- 各銘柄の `score`, `p_trade`, `score_adjusted` を `live/production_residual_blpx/` に保存。
- `skip_reason` や `ml_adjusted` フラグを `decision_df` に追加（後方互換）。

---

## 11. エッジケース・異常系

| # | ケース | 対策 |
|---|---|---|
| 1 | 当日 `r_oc` が `NaN`（provisional 行） | 学習ラベルに使用しない。推論には影響しない。 |
| 2 | `jp_gap_*` が欠損 | 該当銘柄を `p_trade=0` または既存動作（gap data missing → flat）。 |
| 3 | 5 分足データなし（古い期間） | `jp_oc_*` をフォールバックラベルとして使用。 |
| 4 | `Omega_gap` が特異 | `build_weights_minvar` 前に正定値性を確認。 |
| 5 | ロング or ショートが 2 銘柄未満 | フラットポジション。 |
| 6 | `p_trade` が全銘柄 0.5 付近 | `score_adjusted ≈ score`、既存動作に近い。 |
| 7 | 年末年始・祝日 | 通常の `df_exec` アラインメントに従う。 |
| 8 | 米国祝日で `us_cc_*` が NaN | 既存 `preprocess_data` の NaN 処理に従う。 |
| 9 | モデル重みの読み込み失敗 | fail-safe でベースライン動作。 |
| 10 | `score * p_trade` の符号反転 | スコアの符号を保持する（`score_adjusted = score * p_trade` で符号は変わらない）。 |

---

## 12. テスト計画

### 12.1 単体テスト（`tests/unit/test_ml_order_decision.py`）

- 特徴量生成の正しさ（gap の符号、score の符号、NaN 処理）
- `score_adjusted` の符号が `score` と一致すること
- `p_trade = 0.5` で `score_adjusted == score`（案 3 の場合）
- モデル未ロード時に例外を投げないこと
- 市場中立性（`sum(w_long) == -sum(w_short)`）が保たれること

### 12.2 統合テスト

- `test_production_residual_blpx.py` に ML オーバーレイのテストを追加
- `test_leakage_audit.py` に ML 特徴量のルックアヘッドチェックを追加
- `ComplianceAuditor` の `check_pit_binning_lookahead` 等が無効化されないこと

### 12.3 バックテスト

- `BacktestEngine.run_backtest()` で 2015-01-05 〜 2026-06-14 を実行
- ベースライン V2 と ML オーバーレイ版を比較
- コスト内訳（slippage / financing / borrow / reverse）を分解して報告

---

## 13. リスクと対策

| リスク | 内容 | 対策 |
|---|---|---|
| **過学習** | 17 銘柄・小サンプルで銘柄別係数が過学習 | L2 正則化、walk-forward OOS、DSR |
| **市場中立性崩壊** | スキップでドルニュートラリティが崩れる | スコア修正で既存選定層に委ねる |
| **ルックアヘッド** | ラベル作成時に未来情報混入 | PIT 特徴量のみ使用、`jp_oc_*` を特徴量にしない |
| **gap 定義の曖昧さ** | 寄付 vs 9:10 で gap が異なる | `jp_gap_*` の定義に統一（寄付 09:00/09:05/09:10 の最初に取得可能な値） |
| **フォールバック欠如** | モデル不在・データ欠損で動作が止まる | fail-safe でベースライン動作、フラットポジション |
| **ターンオーバー増加** | ML オーバーレイが日々の `p_trade` 変動で銘柄選択を変えるとターンオーバーが増大 | `p_trade` の移動平均で平滑化、ターンオーバーペナルティ付きラベル、連続値スケーリング（案3）で急激な変動を緩和 |
| **既存コストモデルとの不整合** | 片道 5 bps では実コストを過小評価する可能性 | バックテストで金利・貸株・逆日歩を含めて評価 |

---

## 14. 実装スケジュール

| フェーズ | 内容 | 期間目安 |
|---|---|---|
| Phase 0 | 設計書レビュー・合意 | 本日 |
| Phase 1 | 実験スクリプト作成・銘柄別 gap 係数回帰 | 1 日 |
| Phase 2 | バックテスト・ウォークフォワード検証 | 1 日 |
| Phase 3 | 採用/不採用判定・レポート作成 | 0.5 日 |
| Phase 4 | （採用時）本番 `production_v2.py` 統合・テスト | 0.5 日 |

---

## 15. 採用/不採用判定基準

以下のすべてを満たした場合に採用とする：

1. Pooled net Sharpe がベースラインを有意に上回る（改善幅 > 0.5 SE ≈ 0.01）
2. ウォークフォワード勝率 ≥ 8/12
3. DSR ≥ 0.95
4. ±20% パラメータ摂動で Sharpe 変動 < 20%
5. フォールバック発動率が悪化しない
6. ターンオーバーが大幅に増加しない（50% 増以上は要再検討）

いずれかを満たさない場合は不採用とし、レポートのみを `reports/ml_order_decision/` に残す。コードは破棄する（`AGENTS.md` 改善ワークフロー準拠）。

---

## 16. 参考資料

- `AGENTS.md` — 戦略概要・不変条件・改善ワークフロー
- `docs/ARCHITECTURE.md` — アーキテクチャ詳細
- `docs/モデル技術仕様書.md` — 数理仕様
- `docs/日次運用手順書.md` — 運用時刻・発注手順
- `src/leadlag/models/production_v2.py` — V2 本番モデル
- `src/leadlag/models/net_score_ranking_lob.py` — LOB オーバーレイ
- `src/leadlag/core/signal.py:383` — `apply_gap_tolerant_filter`
- `src/leadlag/execution/microstructure/execution_constraints.py` — LOB ハードルール
- `src/leadlag/execution/helpers.py` — 注文送信パイプライン
- `src/leadlag/data/preprocessor.py` — `jp_gap_*` の生成
- `src/leadlag/models/sre.py:24` — `compute_jp_target_returns`
- `tools/production/compute_gap_adjusted_distribution.py` — `mu_gap` / `Omega_gap` の生成

---

## 17. 変更履歴

| 日付 | 変更内容 |
|---|---|
| 2026-07-29 | 初版作成（レビュー結果を踏まえた再構成） |
| 2026-07-29 | v2: 6件の修正（情報二重カウント明記、ラベルを回帰タスクに変更、Phase 1 を回帰に変更、スコア変換の事前診断追加、ターンオーバー方向修正、5分足カバレッジ確認追加） |
