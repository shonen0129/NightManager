# ルックアヘッド・フリー性の形式的解析（B8）

> 作成: 2026-07-13 / 上位モデルによる実行成果物。
> パイプライン各段の情報集合を形式化し、リーク・フリー性を段階ごとに証明または残余リスクとして明示する。

## 記法

- 行 t の意味（データ整合規約より）: US列 = 米国営業日 D_t の close-to-close（JST翌朝確定）、
  JP列（ターゲット）= 取引日 D_{t+1} の 9:10→大引け、jp_gap = D_{t+1} 寄付ギャップ（9:10時点で既知）
- **判定時点の情報集合** F_t := { US_cc[≤t], jp_gap[≤t], topix_night[≤t], y_jp_target[≤t−1], 5分足[≤ D_t の大引け] }
- リーク・フリーの定義: 行 t のシグナル S_t が F_t 可測であること
  （y_jp_target[t] 自体は D_{t+1} 大引けまで未確定なので、S_t が y[t] に依存してはならない）
- 注: 5分足の「09:10バー」は 09:05-09:10 の区間データを含む。09:10:00 時点では当該バーの High/Low は未確定（09:10 終了まで高値安値が変わり得る）。これは R1（執行価格近似）として扱う。

---

## 段階別証明

### 1. ターゲット構築 `compute_jp_target_returns`（sre.py:35-87）

y_jp_target[t] は D_{t+1} の 5分足（9:10バーの (H+L)/2）と jp_oc から構成される。
これは**ターゲット（被予測変数）**であり、シグナルの入力には行 t 当日分は使われない（下記2で証明）。
∴ ターゲット構築自体にリーク概念は適用されない。**ただし執行価格近似 R1（後述）あり。**
R1 は y_jp_target の計算に使う `p_910 = (high + low) / 2` が 09:10:00 判定時点では未確定の high/low を含むため、厳密には 9:10 より少し後の情報を含み得る。ターゲットラベル自体の楽観バイアスであり、シグナル入力のリークではない。

### 2. ローリング相関窓 `compute_signal`（core/signal.py:69-70）

```
window_returns = all_returns[window_start : current_index]
```

半開区間により行 t（= current_index）を**除外**。all_returns の JP 列は y_jp_target なので、
窓に含まれる最新行は y[t−1]（D_t 大引けで確定済み、F_t に含まれる）。
US側の当日入力は `all_returns[current_index, :n_u]` = US_cc[t]（JST朝確定、F_t に含まれる）。
∴ **リーク・フリー（証明済み）**。

### 3. ベースライン相関 `compute_baseline_correlation`（correlation.py:227-255）

2010-01-01〜2014-12-31 の行のみ使用。バックテスト start_date ≥ 2015-01-05 の制約下では
ベースライン期間 ∩ 評価期間 = ∅。c_full は評価期間の全行 t に対し定数であり、
F_t ⊇ {2014年以前のデータ} が成り立つ。
∴ **リーク・フリー（start_date ≥ 2015-01-05 が維持される限り）**。
⚠️ この保証は構成依存: `_prepare_residual_prior` の先頭1260行フォールバックが発動する構成では破れる（AGENTS.md 不変条件2）。

### 4. ローリングOLS残差化 `compute_rolling_ols_betas`（residualize.py:19-99）

- 1次元パス: rolling cov/var の後 `betas_df.shift(1)`（59-60行）により
  β_t は行 ≤ t−1 のデータのみから推定。**確認済み**
- 多次元パス: `X_train = x_data[t−window : t−1+1]` すなわち行 t−window 〜 t−1。
  行 t を含まない。**確認済み**

残差 `y_residuals_p3[t] = y[t] − β_t · topix_oc[t]` は y[t] と同時点の topix_oc[t] を使うが、
これは**残差化されたターゲットの定義**であり、残差系列がシグナル入力になるのは
相関窓（段階2）経由で行 ≤ t−1 のみ。∴ **リーク・フリー**。

### 5. ギャップ調整（signal.py:128-149）

入力は jp_gap[t]（D_{t+1} 寄付、9:10判定時点で既知）、betas_t = jp_beta[t]（R2 参照）、
topix_night[t]（D_{t+1} 朝までに確定）。定義上すべて F_t に含まれる。
∴ **リーク・フリー（jp_beta[t] が F_t 可測である前提、R2 参照）**。

### 6. PIT ビニング `load_pit_ir_history`（production_v2.py:170-218）

```
df_hist = df[df["trade_date"] < trade_date]
```

厳密不等号により当日を除外。分位閾値は履歴のみから計算。
∴ **リーク・フリー**。`check_pit_binning_lookahead` 監査と二重の防御。

### 7. Bayesian BLPX オンライン更新（bayesian_blpx.py:307-315）

```
y_actual_prev = y_jp_target[i - 1] if i > start_idx + 1 else None
```

`compute_blp_signal_bayesian` 内では `current_error = y_actual − self._prev_predicted_z` として
前日の実現値と前日の予測の誤差を計算する。`_prev_predicted_z` は `i−1` 時点の予測で、
`y_actual` は `y_jp_target[i−1]`（D_i の 9:10→大引け）を使う。

`y_jp_target[i−1]` は D_i 大引けで確定済みで F_i に含まれる。また、
`current_error` は事後的に計算され、行 i のシグナル計算（eta_t 経由）に影響を与えるが、
**計算時点では y[i] ではなく y[i−1] を使用** しているため、未確定のターゲットを参照しない。

∴ **リーク・フリー（修正済み）**。

注: 過去の実装では `y_jp_target[i]` を使うバグがあったため、修正済みのコードを使用する。

### 8. メタ学習 IC 計算（sector_relative_ensemble_blp_enhanced.py:1181-1196)

`ic_blpx_vals[i−1]` を y[i−1] とシグナル[i−1] から計算し、行 i の重み予測に使用。
y[i−1] は行 i−1 の取引日（D_i）の 9:10→大引けで、D_i 大引け時点で確定する。
行 i の判定時点は D_{i+1} 9:10 なので、y[i−1] はすでに F_t に含まれる。
∴ **リーク・フリー**。ただし境界が紛らわしいためテストでの明示的検証を推奨（R4）。

### 9. マクロ因子 `compute_macro_surprise`（EWMA ベース）

```python
for t in range(T):
    if t > 0:
        prev = vals[t - 1]
        ewma_mean = decay_mean * ewma_mean + (1.0 - decay_mean) * prev
        resid = prev - ewma_mean
        ewma_var = decay_vol * ewma_var + (1.0 - decay_vol) * resid ** 2

    sigma = np.sqrt(ewma_var)
    sigma_safe = np.where(sigma > 1e-8, sigma, 1.0)
    surprise_z[t] = (vals[t] - ewma_mean) / sigma_safe
```

`t` 時点の `surprise_z[t]` は、`vals[t]` を **t−1 時点までの EWMA 平均・分散** で標準化。
`ewma_mean` / `ewma_var` は `vals[t−1]` までのデータのみで更新された後、`vals[t]` を使う。
したがって `surprise_z[t]` は `vals[≤t]` のみに依存し、将来値 `vals[>t]` を使用しない。

∴ **リーク・フリー（確認済み）**。

注: `ProductionV2Model` の v2 プライマリパスでは `mu_gap`/`Omega_gap` 行列を直接使用するため、
`macro_confidence` は v1 フォールバック経路（`SectorRelativeEnsembleBLPEnhancedModel`）でのみ有効。

### 10. ポートフォリオ構築・RuleD（production_v2.py）

scores → ランキング → baseline_style 重み → PIT乗数。すべて当日 F_t 可測の入力の決定的関数。
∴ **リーク・フリー**。

追加経路（`mu_gap`/`Omega_gap` を入力とする副次的処理）:
- `apply_multi_horizon_blend`: h=1,3,5 の gap 調整済み分布の重み付けブレンド
- `apply_rank_reversal_overlay`: クロスセクショナル・ランク反転オーバーレイ
- `build_weights_minvar`: 予測共分散 `Omega_gap` を使った最小分散ポートフォリオ
- `copula correlation blending`: 動的 t-copula 相関ブレンド（BLPX 信号生成時）

これらの入力はすべて `mu_gap`/`Omega_gap` 行列と `scores` のみで、
`mu_gap`/`Omega_gap` 行列が F_t 可測である限りリーク・フリー。
`mu_gap`/`Omega_gap` の生成プロセスは `compute_gap_adjusted_distribution.py` で行われ、
別途確認が必要（R7 参照、第11節予定）。

---

## 残余リスク一覧と対応指示

| ID | 内容 | 深刻度 | 対応 |
|----|------|--------|------|
| R1 | 9:10価格 = (H+L)/2 近似は 09:10:00 判定時点では未確定の high/low を含み、ターゲットの楽観バイアス。リークではなく**執行価格近似誤差** | 中 | 実約定ログと 9:10 推定価格の差分を検証（B4） |
| R2 | `jp_beta_*` 列の構築は EWMA 経路なら historical だが、OWLS 経路では現在行を含む。いずれもリークではないが、挙動が経路で異なる | 低 | preprocessor の jp_beta 生成コードを文書化し、OWLS/EWMA 経路の違いを注記 |
| R3 | BayesianBLPX の `y_actual_prev` は `y_jp_target[i-1]` を使用するように修正済み。未確定の y[i] を参照しない | 解消 | 修正済み（`bayesian_blpx.py:308`） |
| R4 | メタ学習ICの境界（i−1使用）は正しいが暗黙的 | 低 | 境界を固定するユニットテスト追加（test-gen skill 使用） |
| R5 | `compute_macro_surprise` は `vals[t-1]` で EWMA 更新後に `vals[t]` を標準化 — lookahead-safe | 解消 | 確認済み（`core/macro.py:276-285`） |
| R6 | ベースライン分離は「構成依存の保証」— コードでは強制されない | 中 | `compute_baseline_correlation` または `BacktestEngine` で `baseline_end < start_date` を assert |
| R7 | `mu_gap`/`Omega_gap` 行列の生成（`compute_gap_adjusted_distribution.py`）が F_t 可測か未確認 | 高 | gap 調整済み分布の計算プロセスを別途監査・文書化 |

## 結論

主要パス（相関窓・残差化・PIT・ギャップ調整・ポートフォリオ構築・BayesianBLPX・マクロサプライズ）は**リーク・フリーを証明**。

- **R3（BayesianBLPX）**: `y_jp_target[i-1]` 使用に修正済み。未確定ターゲット参照なし。
- **R5（macro surprise）**: `compute_macro_surprise` は `vals[t-1]` で EWMA を更新後に `vals[t]` を標準化するため、lookahead-safe。
- **R2（jp_beta）**: EWMA 経路は historical、OWLS 経路は現在行を含む。いずれも 9:10 判定時点で既知の情報のみを使い、リークではない。

**残るリスク・未確認事項**:
- **R7（`mu_gap`/`Omega_gap` 行列生成）**: `compute_gap_adjusted_distribution.py` での gap 調整済み分布の計算プロセスが `F_t` 可測か、別途確認が必要。
- **R1（9:10価格近似）**: ターゲットラベルの楽観バイアス。シグナル入力のリークではない。
- **R6（ベースライン期間分離）**: コードでは `start_date` が `baseline_end` より前かどうかは確認されていない。fail-fast assert を追加すべき。

**R3 の本番影響確認済み（2026-07-13）**: `configs/production/production.yaml` の本番モデルは
`production_residual_blpx` であり、`BayesianBLPXModel` を直接使用しない。
→ **本番無風**。BayesianBLPX は研究スクリプト内でのみ参照される。

**次のアクション**:
1. R7 の詳細解析: `compute_gap_adjusted_distribution.py` の `mu_gap`/`Omega_gap` 生成過程を確認
2. R7 確認後、本ドキュメントに第11節（v2 追加経路）を追加
3. R6 の fail-fast assert 実装
4. R4 の境界ユニットテスト追加
