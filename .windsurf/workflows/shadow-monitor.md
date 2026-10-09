---
description: shadowの入力整合と本番昇格条件を検証する
---

# シャドー運用・監視

共通規約は [AGENTS.md](../../AGENTS.md)。長時間実行には [hang-prevention](../../.agents/skills/hang-prevention/SKILL.md) のプロセス全体の停止期限を設定する。

[leadlag-fund-improvement](../../.agents/skills/leadlag-fund-improvement/SKILL.md) のshadow/昇格手順と [日次運用手順書](../../docs/日次運用手順書.md) のread-only受入境界を使う。

現行の監視入口は `tools/validation/monitor_residual_blpx_shadow_performance.py`、runtime出力は `src/leadlag/config/paths.py` で解決する。同一trade date・snapshot・モデル版を照合し、ML enabledとapplied、モデルPnLと実口座PnLを分ける。未取得の実約定費用・未完了の運用一巡は受入済みと扱わない。

研究採用と本番昇格を分け、昇格が依頼範囲ならshadow証拠・本番設定・設計文書を揃える。
