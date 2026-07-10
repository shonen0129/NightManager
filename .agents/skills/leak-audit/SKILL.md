---
name: leak-audit
description: ComplianceAuditorの監査項目を実行・解釈・修正する。ルックアヘッドリーク防止・残余化チェック・アンサンブル重み検証・暴露制限検証・コスト整合性検証を体系的に扱う。モデル変更・シグナル追加・フォールバック階層修正時に必ず参照すること。
---

# Leak Audit スキル

## 目的

`ComplianceAuditor`（`src/leadlag/compliance/auditor.py`）および v2 監査関数（`src/leadlag/compliance/v2_auditor.py`）の監査項目を実行・解釈・修正し、ルックアヘッドリークを未然に防ぐ。

## 監査モジュール構成

### ComplianceAuditor（`auditor.py`）

`run_audit()` は以下5カテゴリの監査を実行:

1. **リークチェック / 時系列検証**
   - `no_lookahead_detected`: `sig_date < trade_date` の全行検証
   - `signal_date_lt_trade_date`: 同上の別名
   - `us_beta_uses_t_minus_1_window`: US残余化ベータ窓がt-1基準
   - `jp_beta_uses_t_minus_1_window`: JP残余化ベータ窓がt-1基準
   - **copula窓の当日行除外**: t-copula推定に使用するローリング窓が当日行を含んでいないか（`correlation.py::estimate_t_copula`）。copula相関とPearson相関のブレンドも同様
   - **macro surpriseのlook-ahead**: `compute_macro_surprise`のローリング平均・ボラティリティ計算が当日行を含んでいないか（`macro.py`）

2. **残余化入力チェック**
   - `p4_uses_us_residualized_input`: P4シグナルがUS残余化リターンを使用
   - `p4_uses_jp_topix_residual_target`: P4のJPターゲットがP3残余化と一致
   - `gamma_zero_matches_raw_us`: gamma=0時にraw USリターンと一致
   - `gamma_one_matches_full_residual_us`: gamma=1時に完全残余化と一致

3. **アンサンブル重みチェック**
   - `ensemble_weights_sum_to_one`: 全シグナル重みの和が1.0

4. **シグナル品質 / 暴露チェック**
   - `no_nan_inf_in_signals`: シグナルにNaN/Infがない
   - `no_nan_inf_in_weights`: ウェイトにNaN/Infがない
   - `net_exposure_within_limit`: net exposure ≤ 0.051
   - `gross_exposure_within_limit`: gross exposure ≤ 2.01

5. **コスト整合性**
   - `cost_consistency_passed`: gross - costs = net が全行で一致

### v2 監査関数（`v2_auditor.py`）

- **`run_leakage_audit(sig_date, trade_date)`**: シグナル日が取引日より厳密に過去であることを検証
- **`run_numerical_audit(w, scores, Omega)`**: ウェイト・スコアの有限性、net exposure ≒ 0、共分散行列の正定値性・対称性を検証
- **MinVar weight検証**（minvar_enabled時）: `build_weights_minvar`の出力が有限、Long/Short合計が±baseline_gross/2にスケール後一致、全ゼロweightでないことを確認

## 監査の実行方法

### バックテスト時

```python
from leadlag.compliance.auditor import ComplianceAuditor

audit_res = ComplianceAuditor.run_audit(
    model=model,
    df_exec=df_exec,
    results=results,  # BacktestEngine.run_backtest() の戻り値
    output_dir="results/backtest_output",
)
```

### v2日次実行時

```python
from leadlag.compliance.v2_auditor import run_leakage_audit, run_numerical_audit

leakage = run_leakage_audit(signal_date_str, trade_date_str)
numerical = run_numerical_audit(w_final, scores, Omega_gap)
```

### テスト経由

```bash
python3 -m pytest tests/integration/test_leakage_audit.py -v
python3 -m pytest tests/integration/test_production_residual_blpx.py -v
```

## 監査失敗時の対応

### FAIL が検出された場合

1. **`no_lookahead_detected = False`**: Critical。シグナル日 ≥ 取引日の行が存在。相関窓・PITビニングの当日行混入を疑う。`signal.py` の `all_returns[window_start:current_index]` を確認
2. **`net_exposure_within_limit = False`**: Critical。ポートフォリオ制約違反。`portfolio.py::adjust_gross_exposure()` と `risk.py` のロジックを確認
3. **`no_nan_inf_in_signals = False`**: Critical。シグナル計算の数値崩壊。入力データの欠損・ゼロ除算を疑う
4. **`cost_consistency_passed = False`**: Warning。コスト計算の不整合。`backtester.py` のコスト計算ロジックを確認
5. **`ensemble_weights_sum_to_one = False`**: Warning。config の重み設定ミス

### v2監査失敗時

- `run_numerical_audit` が FAILED の場合、`fallback_on_audit_failure=True` ならフラットポジションにフォールバック（`production_v2.py:590-594`）
- `fallback_on_audit_failure=False` の場合は v2 ウェイトを保持し alerts に記録

## 出力ファイル

- `audit.json`: 全監査項目の PASS/FAIL
- `audit/<check_name>.csv`: 各項目の個別CSV
- `audit_summary.csv`: 監査サマリー（check_name, status, explanation, recommended_fix）

## 注意事項

- **監査項目を無効化しない**: `ComplianceAuditor` の監査ロジックをスキップ・緩める変更は禁止
- **監査項目の追加は可能**: 新しい不変条件が生まれた場合は `run_audit()` に項目を追加する（ただし既存項目は維持）
- **`AuditContext` の正確性**: `model.get_audit_context()` が返す `AuditContext`（`src/leadlag/models/base.py`）の値が実際のモデル設定と一致していることを確認
