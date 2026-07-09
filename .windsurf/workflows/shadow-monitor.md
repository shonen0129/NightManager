---
description: シャドー運用でライブ整合性を確認し、本番昇格前の検証を行う
---

# シャドー運用・監視

本番昇格前のライブ整合性確認。

## 手順

1. シャドー運用パッケージのビルド:

```
python3 tools/validation/build_residual_blpx_shadow_run_package.py
```

2. シャドー運用の実行:

```
python3 tools/validation/run_daily_residual_blpx_shadow.py
```

3. シャドー性能の監視:

```
python3 tools/validation/monitor_residual_blpx_shadow_performance.py
```

4. 本番モデルの検証（必要時）:

```
python3 tools/validation/validate_production_residual_blpx.py
```

5. 昇格判定基準:
   - シャドーと本番のシグナル整合率が十分であること
   - net Sharpe がベースラインを下回らないこと
   - フォールバック発動率が異常でないこと

## 注意事項

- シャドー結果は `shadow_runs/` に保存される
- 昇格時は `configs/production/` の config 更新 + フォールバック階層の維持 + `docs/ARCHITECTURE.md` のリファクタリング履歴へ追記
- `tools/validation/apply_production_residual_blpx.py` で既存結果への適用テストも可能
