# B8 リーク・フリー性形式解析 — 批判的レビュー

**レビュー日**: 2026-07-13
**対象ドキュメント**: `docs/design/B8_leak_freedom_analysis.md`
**レビュアー**: Cascade

---

## 総評

B8はパイプライン各段の情報集合を形式化し、リスク・フリー性を段階的に論じる好ましい試みである。記法と結論の多くは正しい。しかし、**コードベースとの整合性に重大な問題**がある。特に **R3（BayesianBLPX）の記述は既に修正済みの古いコードを指している** 点、そして **R5（macro surprise）は実装から明らかにリーク・フリー** なのに「未検証」とするのは誤り。また、production v2 の重要な追加経路（multi-horizon blend、rank reversal overlay、minvar、macro confidence、copula）への言及が欠けている。本稿は正確さを欠くため、本番監査文書として使う前に更新が必要。

---

## 1. 誤り・陳腐化した記述

### 1.1 R3（BayesianBLPX）— 最も重大な陳腐化

**B8の記述（71-82行）**:
```
y_actual_prev = y_jp_target[i] if i > start_idx else None
```
として、これがリークだと主張。正しくは `y_jp_target[i−1]` との比較だと指摘。

**現在のコード（`src/leadlag/models/bayesian_blpx.py:308`）**:
```python
y_actual_prev = y_jp_target[i - 1] if i > start_idx + 1 else None
```

**レビュー**: 現在のコードはすでに `y_jp_target[i-1]` を使用しており、**R3は修正済み**。B8は陳腐なスナップショットを引用している。さらに、B8の第7節は `y_jp_target[i]` と `i-1` 時点の予測を比較する「1日遅れ」が必要と主張しているが、現在の実装は `y_actual_prev` として `y_jp_target[i-1]` を渡し、`_prev_predicted_z`（`i-1` 時点の予測）との誤差を計算している。これは正しい1日遅れの誤差計算。

ただし、以下の追加問題が残る:
- `compute_blp_signal_bayesian` 内で `current_error = y_actual - self._prev_predicted_z` だが、`y_actual_prev` に `y_jp_target[i-1]` が入る。`self._prev_predicted_z` は `i-1` 時点の予測（z_hat_j_t1 at i-1）。これは正しい。
- しかし、`_prev_predicted_z` は `i-1` の時点で `B_bayes` 更新後の予測を保存している。`B_bayes` は `i-1` のシグナル計算時に `y_jp_target[i-2]` から更新されたもの。したがって、i-1 予測 vs i-1 実績は正しい。
- 問題: `current_error` は `y_jp_target[i-1]`（D_i の 9:10→大引け）を使う。これは `i-1` 時点の予測（D_i 9:10 直前に計算された z_hat）を使う。D_i 9:10 直前に `jp_gap[i-1]` は既知だが、`y_jp_target[i-1]` は D_i 大引けまで未確定。しかしこの誤差は `i-1` 時点の予測と `i-1` 時点の結果を比較するもので、i-1 時点の予測時点では `y_jp_target[i-1]` は未確定。つまり、**予測時点では未来を見ていない**。誤差計算は事後的（`i` 時点）に行われ、予測時点では未来を使っていない。したがって、これは厳密にはリークではない。B8の記述はやや不正確。

**結論**: R3はコード上修正済みであり、B8は更新が必要。さらに、B8の「R3の本番影響確認済み」という結論は正しい（`production.yaml` は residual_blpx のみ）が、R3が修正済みであればこの結論を引き継ぐ必要がある。

---

### 1.2 R5（macro surprise）— 「未検証」の誤り

**B8の記述（90-93行）**:
> halflife パラメータによる EWMA は過去向き演算だが、**適用行の shift の有無は本解析では未検証** → 残余リスク R5

**現在のコード（`src/leadlag/core/macro.py:276-285`）**:
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

**レビュー**: `t` 時点の `surprise_z[t]` では、`ewma_mean` と `ewma_var` は `vals[t-1]` までで更新された後、`vals[t]` を使って z-score を計算。これは **明確に lookahead-safe** であり、`vals[t]` は現在時点の値、`ewma_mean/ewma_var` は過去のみを使った推定。shift は不要。

**結論**: R5は実装から確認できており、残余リスクではない。B8は誤って「未検証」としている。正しくは「リスク・フリー（確認済み）」と記述すべき。

---

### 1.3 R2（`jp_beta` 構築）— 実は OWLS/EWMA で挙動が異なる

**B8の記述**: preprocessor 側の `jp_beta` 生成が strictly historical か未検証 → R2（中）

**実コードの分析**:

`src/leadlag/data/preprocessor.py` では2つの経路がある:

1. **EWMA 経路**（`beta_ewma_halflife > 0`）:
```python
for t in range(beta_window, n):
    w_gap = gap_arr[t - beta_window : t]          # 行 t 未満
    w_topix = topix_arr[t - beta_window : t]      # 行 t 未満
    betas_arr[t] = w_cov / w_var_topix
```
→ **strictly historical**。`betas_arr[t]` は `gap`/`topix_night` の ≤ t-1 のデータのみを使う。OK。

2. **OWLS 経路**（`beta_ewma_halflife` が無効 / None）:
```python
topix_var = topix_for_beta.rolling(beta_window).var()
for tk in JP_TICKERS:
    cov = gap_for_beta[tk].rolling(beta_window).cov(topix_for_beta)
    betas[tk] = cov / topix_var
```
→ `rolling(beta_window)` は **現在行 t を含む** 窓 `[t-beta_window+1, t]` を使う。`gap_for_beta[t]` と `topix_for_beta[t]` はいずれも D_{t+1} 9:10 時点で既知（`topix_night` は寄付が確定すれば計算可能）。したがって **リークではない** が、**EWMA 経路と挙動が異なる**。

**重要な指摘**:
- `jp_beta[t]` が `gap[t]` を含んで推定されている場合、`
  gap_syst = jp_beta[t] * topix_night[t]`
  `gap_idio = gap[t] - gap_syst`
  は、同一時点の `gap[t]` と `topix_night[t]` で回帰した 「in-sample 残差」 に近い。これは将来情報を使わない（両方とも 9:10 既知）が、**統計的にややオーバーフィット気味** かもしれない。リークではなく、推定手法の問題。
- `topix_night` は `topix_open / topix_close.shift(1) - 1` で計算されるが、`topix_close.shift(1)` は D_t の大引け。`topix_open` は D_{t+1} の寄付。D_{t+1} 9:10 時点では既知。OK。

**結論**: R2は「未検証」というより、「挙動が経路によって異なる（EWMAは historical、OWLSは現在行を含む）」と正しく記述すべき。OWLS経路でもリークはないが、in-sample なベータ推定になっている点を注記すべき。

---

## 2. 不十分・欠落している点

### 2.1 Production v2 追加経路の解析欠落

B8は「主要パス（相関窓・残差化・PIT・ギャップ調整・ポートフォリオ構築）はリスク・フリー」と結論しているが、**ProductionV2Model の重要な追加処理**をほとんど扱っていない。

**欠落している経路**:
- `apply_multi_horizon_blend` (`src/leadlag/models/signal_enhancement.py`): h=3, h=5 の gap 行列を使った信号ブレンド。h>1 の gap 行列がどのように構築されるか（`compute_gap_adjusted_distribution.py` 内）の解析がない。
- `apply_rank_reversal_overlay` (`src/leadlag/models/signal_enhancement.py`): クロスセクショナル・ランク反転オーバーレイ。`z(rank_reversal)` の計算に `y_jp_target` 以降の情報が入らないか確認が必要。
- `solve_baseline_style` + `build_weights_minvar`: minvar 最適化の入力は `mu_gap` / `Omega_gap` のみ。`Omega_gap` は gap 調整済み分布の予測共分散。`compute_gap_adjusted_distribution.py` からどのように計算されるか。
- `compute_macro_surprise` / `compute_factor_kappa_scale` / `compute_macro_direction_adjustment`: マクロ信頼度のスケーリング。`R5` は個別に検証済みだが、結合される `ProductionV2Model` のどの箇所で適用されるか（v2 プライマリは使用しないと `production.yaml` には注記）。
- copula correlation blending: `copula_enabled: true` in `production.yaml`。`SectorRelativeEnsembleBLPEnhancedModel` 内の copula 経路が `all_returns` の未来行を使わないか確認が必要。

**特に重要な欠落**: `compute_gap_adjusted_distribution.py` による `mu_gap`/`Omega_gap` 行列の計算。これは v2 プライマリの全ての信号源であり、B8はこの部分を全く扱っていない。mu_gap は US close 後の「翌日の日本市場リターンの条件付き期待値」で、計算過程で `y_jp_target` 以降の情報を使わないか確認が必要。

---

### 2.2 R1（9:10価格近似）の記述は正しいが、影響評価が不十分

**B8の記述**: 9:10価格 = (H+L)/2 は「9:10までの高値安値」を含むため、厳密には 9:10 より数秒〜分後の情報を含み得る。これはリークではなく執行価格の楽観バイアス（AGENTS.md既知）

**レビュー**: 結論は正しい。ただし、
- `sre.py:59` で `p_910 = (high + low) / 2 if (pd.notna(high) and pd.notna(low)) else close` となっている。`high` と `low` は 09:10 バー（09:05-09:10）の高値・安値。これは 09:10 以降の値動きを含まない（09:10 終了時点の high/low）。ただし、**09:10 時点では high/low はまだ確定していない**（09:10 終了までわからない）。判定時点（09:10）に high/low を使えるか、という点で「9:10 判定時点で既知」という前提が正確でない。実際には 09:10:00 の時点では 09:10 バーの high/low は未知。05:10 分の open か 09:05 バーの close までしか使えないはず。
- B8は「執行価格の楽観バイアス」と結論しているが、**これはリークではなく実行問題**という整理は正しい。ただし影響量の定量化（約定ログとの差分）がない。

**改善案**: R1 は「実行時点（09:10:00）では high/low 未確定」という点を追加し、実約定価格との差分を検証する実験を提案すべき。

---

### 2.3 R6（ベースライン期間分離）の対応が不十分

**B8の提案**: BacktestEngine に `start_date < 2015-01-05` で fail-fast する assertion を追加

**レビュー**: これは正しい方向。ただし、`compute_baseline_correlation` は引数 `baseline_start` / `baseline_end` を取り、呼び出し側が 2010-2014 を指定する。`SectorRelativeEnsembleBLPEnhancedModel._prepare_common_inputs` や `ProductionV2Model` の呼び出しを確認しないと、fail-fast だけで不十分。`baseline_start` / `baseline_end` が 2010-2014 以外になる可能性もある。Pydantic モデルまたは `compute_baseline_correlation` 内で `baseline_end < start_date` を確認する方が堅牢。

---

### 2.4 PIT ビニングの `current_ir` 出所の確認不足

**B8の記述**: `load_pit_ir_history` で `df["trade_date"] < trade_date` として当日を除外 → リーク・フリー

**レビュー**: ファイル読み込み側の制約は正しい。ただし、`current_ir`（当日の IR）が `portfolio_gap_distribution_diagnostics.csv` の `pred_ir_gap_baseline_cost` または `pred_ir_gap_exante_cost` から来る。この `current_ir` が **実現リターンを含まない ex-ante 計算** であることを確認する必要がある。`compute_gap_adjusted_distribution.py` の計算ロジックを確認しないと、B8の結論は不完全。

---

## 3. 正しい点・強み

### 3.1 段階2（ローリング相関窓）

`core/signal.py:69-70` の `all_returns[window_start:current_index]` は半開区間で `current_index` を除外。`all_returns` の JP 列は `y_jp_target` なので、窓に入る最新は `y[t-1]`。これは正しい。

### 3.2 段階3（ベースライン相関）

`compute_baseline_correlation` は `date_index >= 2010-01-01` かつ `<= 2014-12-31` の行のみ使用。`start_date >= 2015-01-05` なら交差なし。ただし、`_prepare_residual_prior` の先頭 1260 行フォールバックが発動する場合に破れる点は正しく指摘。

### 3.3 段階4（ローリング OLS 残差化）

1D パスは `shift(1)` あり、多次元パスは `[t-window, t-1]` まで。どちらも行 t を含まない。正しい。

### 3.4 段階6（PIT ビニング）

`load_pit_ir_history` で `df["trade_date"] < trade_date`（厳密不等号）を使用。当日除外。`get_rolling_pit_bin` でも `history_valid[-rolling_window:]` を使うが、history は履歴 IR のみ。`current_ir` が ex-ante なら OK。

### 3.5 段階8（メタ学習 IC 計算）

`sector_relative_ensemble_blp_enhanced.py:1180-1196` で `i-1` 時点の y と signal を使って IC 計算。これは D_t 大引け確定済み。境界正しい。

---

## 4. 論理構造の問題

### 4.1 `F_t` の定義がやや曖昧

B8の `F_t` 定義（行10）:
> `F_t := { US_cc[≤t], jp_gap[≤t], topix_night[≤t], y_jp_target[≤t−1], 5分足[≤ D_t の大引け] }`

**問題点**:
- `5分足[≤ D_t の大引け]` は正しいか？5分足の `D_t` 大引けは `D_t` 15:15 のデータ。`df_exec` の行 t の US データは `D_t` の close-to-close（JST 翌朝）で確定。`D_t` の日本市場大引けは `D_t` 15:15（JST）で、US close は `D_t` 翌朝（日本時間）なので、`5分足[≤ D_t の大引け]` は `F_t` に含まれる。ただし、9:10 バーの high/low は 09:10 終了時まで確定しないため、判定時点（09:10:00）では含まれない。これは R1 に関連。
- `y_jp_target[≤t−1]` は正しい。`y_jp_target[t-1]` は D_t の 9:10→大引けで、D_t 大引けで確定。D_{t+1} 9:10 判定時点では既知。
- `jp_gap[≤t]` は正しい。`jp_gap[t]` は D_{t+1} 9:10 判定時点で既知。
- `topix_night[≤t]` は正しい。`topix_night[t]` は D_{t+1} 朝の TOPIX 寄付 / D_t 大引けで、D_{t+1} 9:10 判定時点では既知。

**改善案**: `5分足` の部分を「09:10 開始時点では 09:10 バーの high/low は未確定」という注記を追加。

---

### 4.2 「ターゲット構築にリーク概念は適用されない」は正しいが、繰り返し強調

B8の第1節はターゲット構築にリーク概念は適用されないと述べている。これは正しい（ターゲットは被予測変数）。ただし、R1 との関連で、`y_jp_target` の計算に 9:10 以降の価格情報が含まれる可能性は指摘している。整理としては正しい。

---

## 5. 改善提案

### 5.1 即座に修正すべき項目

1. **R3 の陳腐化を更新**: `bayesian_blpx.py` の `y_actual_prev` は `y_jp_target[i-1]` を使うように修正済み。B8 は「修正済み（R3 解消）」と明記。
2. **R5 を「確認済み・リスクなし」に更新**: `macro.py` の `compute_macro_surprise` は `vals[t-1]` で EWMA を更新し `vals[t]` を z-score 化するため、lookahead-safe。
3. **R2 を整理**: EWMA 経路は historical、OWLS 経路は現在行を含む。両方ともリークではないが、in-sample な推定になりうる点を注記。
4. **production.yaml の v2 経路を追加**: 特に `compute_gap_adjusted_distribution.py` での `mu_gap`/`Omega_gap` 計算、multi-horizon blend、rank reversal overlay、minvar、macro confidence、copula のリスク・フリー性を確認・追記。

### 5.2 追加で検証すべき項目

1. **`compute_gap_adjusted_distribution.py` の全経路**: `mu_gap` / `Omega_gap` が US close 時点で計算可能な情報のみから構成されるか確認。
2. **rank reversal overlay**: `z(rank_reversal)` が `y_jp_target` やそれ以降の情報を使わないか確認。
3. **multi-horizon blend の h>1 行列**: `mu_gap_h{h}_{date}.npy` が h 日先の情報を含まないか確認。h=3 や h=5 の行列は「将来の gap」ではなく「当日の h 日先の予測分布」でなければならない。
4. **copula correlation**: `copula_enabled` 時の `t-copula` 推定が `all_returns[window_start:current_index]` のみを使うか確認。
5. **PIT `current_ir` の ex-ante 性**: `compute_gap_adjusted_distribution.py` で `pred_ir_gap_baseline_cost` / `pred_ir_gap_exante_cost` が実現リターンを含まないことを確認。

### 5.3 テスト・監査の強化

1. **R4 境界テスト**: `sector_relative_ensemble_blp_enhanced.py` の IC 計算が `i-1` を使うことをユニットテストで固定。
2. **R6 fail-fast**: `BacktestEngine.run_backtest` または `compute_baseline_correlation` に `baseline_end < start_date` の assert を追加。
3. **R1 実約定検証**: 実約定ログ（もしあれば）と 9:10 (H+L)/2 近似の差分を計算。

---

## 6. 総合判定

| 項目 | 評価 | 備考 |
|------|------|------|
| 形式化の試み | ◎ | 記法・段階別証明は好ましい |
| コードとの整合性 | △ | R3・R5 は陳腐化または誤り |
| 主要パス解析 | ○ | 相関・残差化・PIT・ポートフォリオは正しい |
| v2 追加経路 | ✕ | ほとんど未対応 |
| 実用性 | △ | 修正後に監査文書として使える |

**結論**: B8 は価値のある素案だが、コードとの整合性が取れていない部分が複数ある。特に **R3（BayesianBLPX）が修正済みで陳腐化**、**R5（macro surprise）が実はリスクなし** という点は重大な誤り。Production v2 の追加経路（`mu_gap`/`Omega_gap` 生成、multi-horizon blend、rank reversal overlay、minvar、macro confidence、copula）の解析を追加したうえで、監査文書として更新する必要がある。
