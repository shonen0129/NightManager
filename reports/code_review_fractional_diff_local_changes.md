# ローカル変更客観レビュー: Fractional Differentiation 本番適用

**レビュー日**: 2026-07-21  
**レビュー対象**: 本番config更新、features/fractional_diff.py 新設、pipeline/blp_base 統合、関連ドキュメント  
**結論**: **CONDITIONAL PASS** — 主要な本番パス（production.yaml + BLPX-Enhanced）は正しく構成されている。ただし、旧PCA-Ensembleモデルと旧本番configの整合性、数値安定性に対する追加対応が必要。

---

## 1. レビュー範囲と調査方法

### 確認したファイル
- `configs/production/production.yaml`
- `configs/production/production_v2_primary_ruleD.yaml`
- `configs/production/production_residual_blpx.yaml`
- `src/leadlag/features/fractional_diff.py`
- `src/leadlag/features/__init__.py`
- `src/leadlag/core/pipeline.py`
- `src/leadlag/models/blp_base.py`
- `src/leadlag/models/sre.py`
- `src/leadlag/models/sector_relative_ensemble_blp_enhanced.py`
- `src/leadlag/models/sector_relative_ensemble_blp.py`
- `src/leadlag/models/sector_relative_ensemble_rrr.py`
- `tests/features/test_fractional_diff.py`
- `tools/production/compute_gap_adjusted_distribution.py`
- `tools/production/run_daily_production_v2.py`
- `docs/ARCHITECTURE.md`
- `AGENTS.md`

### 実行した検証
```bash
python3 _check_syntax.py          # 12/12 OK
python3 -m pytest tests/features/test_fractional_diff.py -v -q   # 23 passed
python3 -m pytest tests/features/test_fractional_diff.py tests/unit/test_allocator.py -v -q  # 31 passed
```

---

## 2. 問題一覧

### [P1-001] `SectorRelativeEnsembleModel`（旧PCA-Ensemble）が fractional diff を無視する
- **ファイルと行番号**: `src/leadlag/models/sre.py:168-179`
- **発生条件**: `SectorRelativeEnsembleModel` を使う実行経路で `features.fractional_diff.enabled: true` のconfigを読み込むと、分数差分が適用されない。
- **影響**:
  - `execution/helpers.py:183` の `build_strategy()`、`execution/fast.py:383`、その他旧PCA-Ensemble経路で本番configと異なるUSリターン列を使用する。
  - 同一configなのにモデル実装によって fractional diff 有無が変わり、再現性・整合性を損なう。
- **根拠（コード引用）**:
  ```python
  # sre.py 168-179
  inputs = build_common_inputs(
      df_exec,
      y_jp_target,
      n_u=self.n_u,
      n_j=self.n_j,
      ewma_half_life=self.ewma_half_life,
      beta_window=self.beta_window,
      include_v4_prior=self.include_v4_prior,
      us_res_enabled=self.us_res_enabled,
      us_res_gamma=self.us_res_gamma,
      us_res_beta_window=self.us_res_beta_window,
  )
  ```
  `frac_diff_*` パラメータが全く渡されていないため、default値（`frac_diff_enabled=False`）が適用される。
- **なぜテストで防げないか**: `test_fractional_diff.py` は `fractional_diff` モジュール単体のみテストしており、`sre.py` 経路の統合テストがない。
- **修正方針**: `sre.py` の `__init__` と `_prepare_common_inputs` でも `features.fractional_diff` configを読み、同名パラメータを `build_common_inputs` に渡す。旧PCA-Ensembleを deprecated にする場合は `execution/helpers.py`, `execution/fast.py` からの呼び出しを BLPX-Enhanced に切り替えるか、注意書きを残す。
- **確信度**: 高

---

### [P1-002] 旧本番config (`production_v2_primary_ruleD.yaml`, `production_residual_blpx.yaml`) に fractional diff セクションがない
- **ファイルと行番号**: `configs/production/production_v2_primary_ruleD.yaml`, `configs/production/production_residual_blpx.yaml`
- **発生条件**: 研究スクリプトや旧runnerがこれらconfigを指定して `SectorRelativeEnsembleBLPEnhancedModel` をインスタンス化すると、fractional diff が無効のまま動作する。
- **影響**:
  - 例: `src/research/scripts/blpx/experiment_bayesian_blpx.py` は `--config configs/production/production_v2_primary_ruleD.yaml` をデフォルトで使用。
  - 本番config (`production.yaml`) との比較実験で、同じ `d=0.1` を意図していても旧config側では分数差分が働かず、不公平な比較になりうる。
- **根拠（コード引用）**:
  - `blp_base.py:30-37` は `self.config.get("features", {}).get("fractional_diff", {})` と読む。`features` セクションが存在しないconfigでは空dict→`enabled=False`。
- **なぜテストで防げないか**: configファイルの網羅的整合性テストがない。
- **修正方針**: 
  - 選択肢A: 旧configにも同一 `features.fractional_diff` ブロックを追記。
  - 選択肢B: 旧configを `archive/configs/` に移動し、非推奨化。
  - `AGENTS.md` では `production_v2_primary_ruleD.yaml` が「旧版」と明記されているため、少なくとも旧config利用箇所への警告コメントまたは移動を推奨。
- **確信度**: 高

---

### [P2-001] `fractional_diff.py::adf_test` で数値オーバーフロー/無効値のRuntimeWarning
- **ファイルと行番号**: `src/leadlag/features/fractional_diff.py:150-225`（特に 195行 `resid = y_dep - X @ beta`）
- **発生条件**: `find_optimal_d` や `adf_test` が特定の系列（定常性が強い、または分散が極端な系列）に対して呼ばれると、`np.linalg.lstsq` の結果 `beta` が巨大になり、`X @ beta` で overflow/invalid value となる。
- **影響**:
  - 本番の `compute_gap_adjusted_distribution.py` では `find_optimal_d` / `adf_test` は呼ばれていないため、直接の本番影響はなさそうだが、モジュール内の関数として再利用される場合に誤った定常性判定を返す可能性がある。
  - テストでも `RuntimeWarning: overflow encountered in matmul`, `invalid value encountered in matmul` が13件出力されている。
- **根拠（コード引用）**:
  ```python
  beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]
  resid = y_dep - X @ beta   # ← overflow/invalid
  ```
  `X` のスケールや `y` のスケールが大きいと、OLS解が過大になり `matmul` で Inf/NaN が発生。
- **なぜテストで防げないか**: テストは `is_stationary` のbool値をチェックしているが、数値異常時に `t_stat` が非有限な場合は `return {"statistic": np.nan, ...}` となり、テスト上は `not is_stationary` として通過する。
- **修正方針**:
  1. `X` を標準化してから `lstsq` に渡す（例: `X[:, 1:] = (X[:, 1:] - mean) / std`）。
  2. `beta` または `X @ beta` に対し `np.isfinite` チェックを入れ、非有限なら早期リターン。
  3. `np.linalg.lstsq` を `np.linalg.lstsq(..., rcond=None)` のままでも、入力の前処理でスケールを抑制。
- **確信度**: 中

---

### [P2-002] ウォームアップ期間の NaN を `0.0` で fill する設計の根拠が不明確
- **ファイルと行番号**: `src/leadlag/core/pipeline.py:128-136`, `src/leadlag/features/fractional_diff.py:397-400`
- **発生条件**: `fractional_diff` 適用直後の `window` 日（最大100日）で、利用可能な過去データが不足するため NaN が生じる。これを `fillna(0.0)` で0埋めしている。
- **影響**:
  - 2010-01-04～2010-05-21頃のUSリターンが0となるため、基準相関行列 `c_full` や残差化のベースライン期間に影響を与える可能性がある。
  - 2015年本番期間には入らないが、ベースライン期間 (2010-2014) に入るため、事前分布・基準相関が歪む可能性がある。
  - 実際のバックテスト結果では改善しているので empiricalには問題なさそうだが、理論的根拠を文書化していない。
- **根拠（コード引用）**:
  ```python
  fd_df = fractional_diff_df(
      us_df, d=frac_diff_d, threshold=frac_diff_threshold, window=frac_diff_window
  ).fillna(0.0)
  ```
- **なぜテストで防げないか**: テスト `test_no_nans_after_fill` は「NaNがないこと」のみ確認し、値の妥当性（0埋めが相関に与える影響）を検証していない。
- **修正方針**: 
  - 0埋めの根拠を `ARCHITECTURE.md` または `fractional_diff.py` docstring に追記する。
  - 代替案として、window=100 なら営業日100日を捨てるか、あるいは前の非NaN値を forward-fill することを検証・文書化する。
  - もしくは、ベースライン期間に window 分の余裕（2010-04頃以降まで）を持たせることをconfigで明記。
- **確信度**: 中

---

### [P2-003] `fractional_diff` 関数の docstring と実装の不一致（warmup NaN）
- **ファイルと行番号**: `src/leadlag/features/fractional_diff.py:71-119`
- **発生条件**: `fractional_diff()` docstring には「initial warmup period の NaN」を返すと書かれているが、実装は `lookback = min(width, i + 1, window)` として partial window を使い、最初の行から値を計算する。
- **影響**:
  - ドキュメントと実装が食い違い、メンテナが誤解する可能性がある。
  - `tests/features/test_fractional_diff.py:95-103` の `test_warmup_nans` は「NaNの数に硬性な要求はしない」と自己言及的なアサーションになっており、仕様が不明確。
- **根拠（コード引用）**:
  ```python
  """Returns: Transformed series of same length, with NaN for initial warmup period."""
  ```
  対して実装:
  ```python
  lookback = min(width, i + 1, window)
  w_slice = weights[:lookback]
  x_slice = values[i - lookback + 1 : i + 1][::-1]
  result[i] = np.dot(w_slice, x_slice)
  ```
  `i=0` でも `lookback=1` なので `result[0]` は NaN にならない。
- **なぜテストで防げないか**: `test_warmup_nans` が `assert result.isna().sum() >= 0` という自明なアサーションになっている。
- **修正方針**: docstring を実装に合わせて修正するか、実装を docstring に合わせて最初 `window` 日を NaN にする。実装は valid なので docstring 修正が推奨。
- **確信度**: 高

---

### [P3-001] `tests/features/test_fractional_diff.py::test_warmup_nans` が弱い
- **ファイルと行番号**: `tests/features/test_fractional_diff.py:95-103`
- **発生条件**: 常に成功するアサーション。
- **影響**: 回帰テストとして機能しない。
- **修正方針**: 実装方針を決定後、具体的なアサーションに書き換える（例: `d=0.5, threshold=1e-5` の場合、window=100 とすると NaN が100個、または partial window なら0個）。
- **確信度**: 高

---

### [P3-002] `features/__init__.py` が空だが、これでよいか未確認
- **ファイルと行番号**: `src/leadlag/features/__init__.py`
- **発生条件**: 他の `leadlag/` サブパッケージとの一貫性。特に `core/`, `models/`, `compliance/` 等の `__init__.py` がどうなっているか。
- **影響**: `from leadlag.features import fractional_diff` 等の短縮インポートができない。利用箇所は `from leadlag.features.fractional_diff import ...` を使っているため問題ないが、一貫性のため確認。
- **修正方針**: 他サブパッケージと同様の記述をするか、空のままの意図を明記。
- **確信度**: 低

---

## 3. 肯定的な指摘

- **ルックアヘッド安全性**: `fractional_diff` は時点 `i` のみ過去データを使用（`x_slice = values[i-lookback+1:i+1]`）。実験スクリプトでの未来データ改変テストもPASS。
- **本番パスの正しさ**: `compute_gap_adjusted_distribution.py` は `SectorRelativeEnsembleBLPEnhancedModel` を使用し、これは `_BLPBase` を継承するため `features.fractional_diff` configを正しく反映する。
- **config 読み取りの堅牢性**: `blp_base.py` では `config.get("features", {}).get("fractional_diff", {})` と safe-get を使っており、セクション欠損で KeyError にはならない。
- **テストカバレッジ**: `compute_weights`, `fractional_diff`, `fractional_diff_df`, `adf_test`, `hurst_exponent`, `find_optimal_d`, `apply_fractional_diff_to_df_exec` それぞれに少なくとも2-3テストが存在。
- **ドキュメント更新**: `ARCHITECTURE.md` Phase 18 と `AGENTS.md` に検証結果が反映されている。

---

## 4. 最終判定

- **BLOCK 事項なし**: 本番の主要経路（`production.yaml` → `compute_gap_adjusted_distribution.py` → `SectorRelativeEnsembleBLPEnhancedModel`）で分数差分 `d=0.1` が有効化され、ルックアヘッドリークはない。
- **CONDITIONAL PASS 理由**: 
  - 旧PCA-Ensembleモデル (`SectorRelativeEnsembleModel`) と旧config (`production_v2_primary_ruleD.yaml`, `production_residual_blpx.yaml`) では fractional diff が無効のままとなる。
  - `adf_test` / `find_optimal_d` の数値安定性に警告あり（本番パス未使用だが、モジュールの信頼性に影響）。
  - ウォームアップ期間の `fillna(0.0)` の理論的・運用上の根拠が文書化されていない。

---

## 5. 推奨アクション（優先順位順）

1. **P1-001, P1-002**: 旧PCA-Ensemble/旧config に対して fractional diff 統合を明示（または旧経路を廃止化）。
2. **P2-001**: `adf_test` の数値安定性を改善（入力標準化または非finite値の早期リターン）。
3. **P2-002, P2-003**: ウォームアップ処理の根拠を文書化し、`test_warmup_nans` を強化。
4. **P3-002**: `features/__init__.py` の扱いを他サブパッケージと合わせて確認。

---

## 6. カバレッジ表

| 観点 | 評価 | 備考 |
|------|------|------|
| ルックアヘッドリーク | 問題なし | fractional_diff は過去データのみ使用。ただし旧PCA-Ensemble経路で無効化されるリスクあり。 |
| ベースライン期間分離 | 情報不足 | 0-fillのウォームアップが2010-2014に与える影響を数値的に確認すべき。 |
| 市場中立制約 | 該当なし | 本変更はUSリターン列変換のみ、portfolio/neutral制御には影響なし。 |
| 数値安定性 | 問題あり | `adf_test` で overflow/invalid warning。本番未使用だが改善推奨。 |
| フォールバック挙動 | 情報不足 | fractional diff 無効時は素通し。config誤りで無効化されても silent fail の可能性。 |
| コンプライアンス監査 | 問題なし | ComplianceAuditor 全クリティカル項目PASS済。 |
| 通常系ロジックエラー | 問題なし | 主要本番パスで `d=0.1` 適用確認済。 |
| 境界値・空値・NaN | 問題あり | warmup fill=0.0 の根拠が不明。 |
| エラー処理 | 問題なし | `blp_base.py` で safe-get 使用。 |
| 状態管理・キャッシュ | 該当なし | 変更箇所にモジュールグローバルキャッシュなし。 |
| パフォーマンス | 該当なし | 列ごとループ (fractional_diff_df) だが本番は15列のみ。 |
| ハング | 該当なし | ループに `k > 10000` の安全上限あり。 |
| 設定値・環境差異 | 問題あり | 旧configとの整合性が取れていない。 |
| テスト | 問題あり | `test_warmup_nans` が弱い。旧PCA-Ensemble経路のテストなし。 |
| 重複実装・到達不能コード | 情報不足 | `apply_fractional_diff_to_df_exec` と `fractional_diff_df` の重複は意図的だが整理可能。 |

---

## 7. 最終自己点検

- [x] 主要なエントリーポイントをすべて確認した
- [x] 正常系と異常系の両方を確認した
- [x] 処理経路を追跡した（`compute_gap_adjusted_distribution.py` → `BLPEnhancedModel._prepare_common_inputs` → `build_common_inputs`）
- [x] ドメイン固有リスク（リーク・中立・数値安定性）を確認した
- [x] 2回目の反証レビューを実施した
- [x] 指摘ごとに具体的な根拠がある
- [x] 未確認範囲を隠していない（旧config・旧PCA-Ensemble経路を明示）
