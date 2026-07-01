# モデル改善計画 — シグナル精度向上 (Rank IC / ICIR)

## 現状分析

### 本番モデル構成
- **モデル**: Residual-BLPX-RA v2 (`sector_relative_ensemble_blp_enhanced.py`)
- **使用シグナル**: `residual_blpx` のみ (weight=1.0)、他コンポーネントは全て無効
- **ランキング**: `mu_over_sigma` (gap調整済み期待リターン / gap調整済み標準偏差)
- **動的グロス**: RuleD (PIT binning, 0.75x or 1.0x)
- **性能**: Sharpe ~3.5, Rank IC ~0.2036, ICIR ~10.96

### 過去の改善試行と結果
| Sprint | アプローチ | 結果 |
|--------|-----------|------|
| 3A | ヒンジ特徴量オーバーレイ | FAIL: 全銘柄共通特徴量で順位不変 |
| 3B | ヒンジ×アセット暴露交差 | MARGINAL: Rank IC +0.0001, Net Return +0.36% |
| Health Score | ポジションサイズ動的調整 | REJECT: Sharpe 3.54→2.96 悪化 |

### 改善のボトルネック
1. **アンサンブル未活用**: 4コンポーネントの枠組みがあるが residual_blpx のみ使用
2. **パラメータ静的**: BLPX全パラメータが固定値、市場レジーム不問
3. **相関推定単一**: EWMA half-life=45 のみ、複数ホライズン未検証
4. **セクター事前分布固定**: `sector_eta=0.0` で完全固定マッピング
5. **非線形改善の限界**: Sprint 3B で+0.36%が最大、線形モデルの限界近傍

---

## Phase 1: 既存モデル調整 (Quick Wins)

### 1A: BLPX パラメータ最適化 [HIGH]

**目標**: 現行の固定パラメータを最適化し、Rank IC を改善

**対象パラメータ**:
| パラメータ | 現行値 | 検索範囲 | 役割 |
|-----------|--------|---------|------|
| `alpha_xx` | 0.50 | [0.1, 0.9] | US block shrinkage |
| `alpha_yy` | 0.50 | [0.1, 0.9] | JP block shrinkage |
| `alpha_yx` | 0.05 | [0.0, 0.3] | Cross-block shrinkage |
| `lambda_pca` | 0.10 | [0.0, 0.5] | PCA prior weight |
| `lambda_sector` | 0.40 | [0.0, 0.8] | Sector prior weight |
| `beta_conf` | 0.25 | [0.0, 0.5] | Confidence weighting power |
| `rho` | 0.01 | [0.001, 0.1] | Ridge regularization |
| `winsor_sigma` | 3.0 | [2.0, 5.0, None] | Outlier clipping |
| `blp_window` | 252 | [126, 189, 252, 378, 504] | Training window |
| `ewma_halflife` | 45 | [21, 30, 45, 63, 90] | EWMA half-life |

**手法**:
1. グリッドサーチ (coarse) → ベイズ最適化 (fine) の2段階
2. Walk-forward OOS Rank IC を目的関数
3. 過学習防止: train 252日 / validation 63日 / test 21日のローリング

**実装**:
- `scripts/experiment_blpx_parameter_optimization.py` を新規作成
- 既存 `scripts/experiment_parameter_sensitivity.py` を参考
- `BacktestEngine` を利用して OOS Rank IC を計測

**期待効果**: Rank IC +0.005〜0.01 (パラメータ調整のみで)

### 1B: 適応的アンサンブル重み [HIGH]

**目標**: 現在無効化されている raw_pca / residual_pca / raw_blpx を再賦活し、ローリング IC ベースで重みを動的調整

**現状**: `residual_blpx_weight=1.0`, 他全て0.0 → 単一モデル運用

**アプローチ**:
1. 各コンポーネントのローリング Rank IC (63日窓) を計算
2. IC 重み付け: `w_k = max(0, IC_k)^gamma / sum(max(0, IC_j)^gamma)`
3. フロア: 各コンポーネント最低重み 0.05 (多様性確保)
4. ゼロ/負のICコンポーネントは除外

**実装**:
- `blp_base.py` の `combine_signals` を拡張して適応重みに対応
- `configs/production.yaml` に `ensemble.adaptive: true` 設定を追加
- バックテストで固定重み vs 適応重みを比較

**期待効果**: Rank IC +0.003〜0.008 ( diversification benefit)

### 1C: 相関推定改良 [MEDIUM]

**目標**: EWMA相関行列の推定精度を向上

**アプローチ**:
1. **複数ホライズン EWMA ブレンド**:
   - short (half-life=21) + medium (45) + long (90) の3水準
   - 重み付けブレンド: `C_blended = w_s * C_short + w_m * C_medium + w_l * C_long`
   - ブレンド重みもパラメータ最適化対象

2. **LW shrinkage 強度の動的調整**:
   - 現行: `lambda_lw=0.5` 固定
   - 市場ボラティリティに応じて動的調整 (高ボラ時は縮約強化)

**実装**:
- `core/correlation.py::compute_correlation` に multi-horizon オプション追加
- `configs/production.yaml` に `correlation.multi_halflife: [21, 45, 90]` 設定

**期待効果**: Rank IC +0.002〜0.005

### 1D: セクター事前分布最適化 [MEDIUM]

**目標**: 固定セクターマッピング `M_sector` のデータ駆動型ブレンドを最適化

**現状**: `sector_eta=0.0` (完全固定), `sector_gamma=2.0` (未使用)

**アプローチ**:
1. `sector_eta` を 0.0〜0.5 の範囲でグリッドサーチ
2. `sector_gamma` を 1.0〜4.0 の範囲で最適化
3. `experiment_models.py` の `BlendSectorModel`, `CorrOnlySectorModel`, `RidgeDynamicSectorModel` を比較
4. 最良のセクター事前分布モデルを採用

**実装**:
- `scripts/experiment_sector_prior_optimization.py` を新規作成
- 既存 `experiment_models.py` のモデルバリアントを活用

**期待効果**: Rank IC +0.002〜0.005

---

## Phase 2: 新規予測モデル追加

### 2A: マルチホライズン信号ブレンド [MEDIUM]

**目標**: 1日予測に加え 3日・5日予測をブレンドし、予測の安定性を向上

**アプローチ**:
1. BLPX モデルで 1日・3日・5日先行の JP リターンを予測
2. 各ホライズンの予測を Zスコア正規化
3. 重み付けブレンド: `signal = w_1 * sig_1d + w_3 * sig_3d + w_5 * sig_5d`
4. 重みは walk-forward で最適化

**実装**:
- `compute_blp_signal` に `forecast_horizon` パラメータ追加
- `all_returns` のターゲットを h日前先行にシフト
- `configs/production.yaml` に `multi_horizon.weights` 設定

**期待効果**: Rank IC +0.003〜0.008 (ノイズ低減)

### 2B: レジーム条件付きパラメータ適応 [MEDIUM]

**目標**: 市場レジーム（ボラティリティ・クロスセクショナル分散）に応じて BLPX パラメータを動的切替

**アプローチ**:
1. レジーム分類:
   - ボラティリティレジーム: TOPIX 20日ボラの 25/75 パーセンタイルで3分割
   - 分散レジーム: JP セクター間リターン分散の 25/75 パーセンタイルで3分割
2. 各レジームで最適化されたパラメータセットを適用
3. レジーム遷移時のスムージング (EMA重み)

**実装**:
- `configs/production.yaml` に `regime.param_sets` セクション追加
- `sector_relative_ensemble_blp_enhanced.py` にレジーム判定ロジック追加
- `scripts/experiment_regime_adaptive.py` で検証

**期待効果**: Rank IC +0.003〜0.007

### 2C: 拡張非線形オーバーレイ [LOW]

**目標**: Sprint 3B の延長線上で、より強力な ML モデルによるオーバーレイを検証

**アプローチ**:
1. Sprint 3B の `combined_interaction_elasticnet` をベース
2. GBDT (LightGBM) による非線形モデル追加
3. rolling beta window を 252→504 に延長 (Sprint 3B レポート推奨)
4. より多様な asset-specific 特徴量を追加:
   - 銘柄別 gap surprise × rolling vol
   - US sector return × 銘柄別 overnight beta
   - VIX change × 銘柄別 size factor

**実装**:
- `scripts/experiment_sprint3c_extended_overlay.py` を新規作成
- Sprint 3B の `run_sprint3b_hinge_interactions.py` をベース
- LightGBM モデルを `requirements.txt` に追加

**期待効果**: Rank IC +0.001〜0.003 (Sprint 3B の限界をわずかに超える程度)

### 2D: クロスセクショナル特徴量 [LOW]

**目標**: リードラグ信号と直交するクロスセクショナル信号を統合

**アプローチ**:
1. **セクター相対モメンタム**: 直近5日の JP セクター相対リターン
2. **平均回帰シグナル**: 直近1日の JP セクター異常リターンの逆張り
3. **分散シグナル**: クロスセクショナル分散が高い時の逆張り
4. リードラグ信号との相関が低いことを確認してブレンド

**実装**:
- `src/experiments/features/cross_sectional.py` を新規作成
- `predict_signals` でリードラグ信号にクロスセクショナル信号を加重ブレンド

**期待効果**: Rank IC +0.001〜0.003 (直交信号の diversification)

---

## Phase 3: 統合検証・本番適用

### 3A: ウォークフォワード統合検証 [HIGH]

**目標**: Phase 1・2 で有効だった改善を組み合わせ、walk-forward OOS で総合評価

**検証項目**:
1. Rank IC / ICIR の改善
2. 分位リターンの単調性
3. AUM 100万円・スプレッド 10/15/20/30bps での net return / IR / max DD
4. DD 悪化なし
5. 過学習兆候なし (train-test gap)

**採用基準**:
- Net return +3% 以上改善
- IR +0.20 以上改善
- Max DD 悪化なし
- Spearman rank corr < 0.995 (ランキング実質変更)

### 3B: 本番適用 [HIGH]

**目標**: 採用された改善を production pipeline に統合

**作業**:
1. `configs/production.yaml` の更新
2. `configs/production_v2_primary_ruleD.yaml` の更新
3. テストスイートの更新・追加
4. `docs/ARCHITECTURE.md` の更新
5. `docs/モデル技術仕様書.md` の更新
6. シャドウランナーでの1ヶ月検証

---

## 優先順位・実行順序

```
Phase 1A (パラメータ最適化) ──┐
                              ├──→ Phase 3A (統合検証) ──→ Phase 3B (本番適用)
Phase 1B (適応アンサンブル) ──┤
                              │
Phase 1C (相関推定改良)   ────┤
Phase 1D (セクター事前分布) ──┘

Phase 2A (マルチホライズン) ──┐
Phase 2B (レジーム適応)   ────┤  (Phase 1完了後に評価して優先度決定)
Phase 2C (非線形オーバーレイ) ─┤
Phase 2D (クロスセクショナル) ─┘
```

**推奨**: Phase 1A → 1B を先に実行し、効果を確認してから残りの優先度を判断。

---

## リスク・注意事項

1. **過学習リスク**: パラメータ最適化は walk-forward OOS で厳格に検証
2. **現行性能維持**: Sharpe 3.5 を下回る変更は不採用
3. **バックテスト一貫性**: `BacktestEngine` と `ComplianceAuditor` の監査を全て通過
4. **リーク防止**: `check_residualization_leakage` 等、既存監査項目を全て維持
5. **フォールバック互換**: v2 → v1 → PCA-Ensemble のフォールバック階層を維持
