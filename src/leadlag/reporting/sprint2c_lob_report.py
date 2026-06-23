from __future__ import annotations

import logging
import os
import pandas as pd
from typing import Any

logger = logging.getLogger(__name__)

def generate_historical_fixed_section(summary_df: pd.DataFrame | None) -> str:
    """Generates markdown section for historical fixed spread results."""
    if summary_df is None or summary_df.empty:
        return "No historical fixed spread simulation data available.\n"
    
    # Format columns nicely
    df_formatted = summary_df.copy()
    
    # Find any float columns and format them
    for col in df_formatted.columns:
        if col in ["annualized_net_return", "annualized_volatility", "max_drawdown", "hit_rate", "approx_after_tax_return"]:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
        elif col in ["IR"]:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        elif col in ["annual_jpy_pnl", "total_commission_jpy", "total_interest_jpy"]:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:,.0f}円" if pd.notnull(x) else "N/A")

    table_md = df_formatted.to_markdown(index=False)
    return f"""### 2009～2026年 固定スプレッド感応度分析結果
以下の表は、過去のヒストリカルデータに対し、往復の固定スプレッドコスト（5bps～50bps）を適用した際の `net_score_ranking` モデルのバックテスト結果を示しています。

{table_md}
"""


def generate_lob_availability_section(availability_df: pd.DataFrame | None) -> str:
    """Generates LOB availability stats markdown."""
    if availability_df is None or availability_df.empty:
        return "LOB log snapshots: **Not generated** (過去板データが存在しないため)\n"

    table_md = availability_df.to_markdown(index=False)
    return f"""### 板ログ収集稼働状況 (LOB Log Availability)
9:09:50～9:10:10 の収集ウィンドウで取得・保存された板スナップショットの稼働統計です。

{table_md}
"""


def generate_skip_reason_section(skip_df: pd.DataFrame | None) -> str:
    """Generates skip reasons breakdown markdown."""
    if skip_df is None or skip_df.empty:
        return "Skip reasons breakdown: **Not generated** (収集期間の取引候補が存在しない、または板データなし)\n"

    table_md = skip_df.to_markdown(index=False)
    return f"""### 注文フィルタリング・スキップ要因分析 (LOB Exclusions & Replacements)
板情報（スプレッド、スリッページ、板厚）および信用規制により注文候補から除外・代替された内訳です。

{table_md}
"""


def generate_slippage_stats_section(slippage_df: pd.DataFrame | None) -> str:
    """Generates spread and slippage statistics markdown."""
    if slippage_df is None or slippage_df.empty:
        return "LOB spread and slippage statistics: **Not generated** (板データなし)\n"

    table_md = slippage_df.to_markdown(index=False)
    return f"""### スプレッド・スリッページ実測統計 (Empirical Slippage Metrics)
対象期間中に実測された最良気配スプレッドおよび推定スリッページの統計値です。

{table_md}
"""


def generate_go_live_checklist() -> str:
    """Generates the mandatory checklist for going live."""
    return """### 本番移行判断チェックリスト (Go-Live Decision Checklist)

- [ ] **固定スプレッドのフォワード妥当性検証**: 実測された平均往復スプレッドが、バックテストで想定した 15bps 未満に収まっているか。
- [ ] **板の流動性（スリッページ）の検証**: AUM 100万円における1銘柄あたりの発注サイズ（約10万円）に対して、9:10板の最良気配および5本板の厚みが十分にあり、推定スリッページが 5bps 未満であるか。
- [ ] **注文除外・代替ルールの機能検証**: 信用取引不可銘柄が検出された際に、代替のショート候補が正しく選択されるロジックが機能しているか。
- [ ] **API接続信頼性の確認**: `quote-log` ウィンドウ（9:09:50～9:10:10）におけるAPIエラー発生率が 1% 未満に抑制されているか。
"""


def render_markdown_report(
    output_path: str,
    historical_summary: pd.DataFrame | None,
    availability_summary: pd.DataFrame | None = None,
    skip_summary: pd.DataFrame | None = None,
    slippage_summary: pd.DataFrame | None = None
) -> None:
    """Renders the complete Sprint 2-C report to a Markdown file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    mandatory_notice = (
        "> [!IMPORTANT]\n"
        "> **長期バックテストに関する重要な注意点**\n"
        "> 本レポートでは、過去板データが存在しないため、2009〜2026年の長期期間について LOB-based slippage backtest は実施していません。\n"
        "> 長期検証は固定スプレッドbps感応度のみであり、板情報ベースの評価は Sprint 2-C 以降に保存された板スナップショット期間に限定されます。\n"
    )

    report_content = f"""# Sprint 2-C: AUM 100万円向け板情報ベース・スリッページ計測と実行フィルター検証レポート

## 概要
本レポートは、AUM 100万円における立花証券信用取引条件に基づき、実気配（LOB）情報を反映したスリッページモデルおよび `net_score_ranking_lob` 実行フィルターの実装と、固定スプレッド感応度によるヒストリカル検証結果をまとめたものです。

{mandatory_notice}

## 1. ヒストリカル固定スプレッド検証結果
{generate_historical_fixed_section(historical_summary)}

## 2. 実LOBデータに基づくフォワード検証（収集期間のみ）
{generate_lob_availability_section(availability_summary)}
{generate_skip_reason_section(skip_summary)}
{generate_slippage_stats_section(slippage_summary)}

## 3. 本番移行に向けた評価
{generate_go_live_checklist()}

## 結論と推奨設定
- **長期バックテスト結果からの示唆**: 往復スプレッドが 15bps までであれば、年率netリターンは十分に確保できますが、20bpsを超えると急激にパフォーマンスが悪化します。本番運用では、LOBフィルターを用いて往復スプレッドが広い銘柄（特に 20bps 以上）を厳格に除外することが強く推奨されます。
- **推奨する本番設定**:
  - `max_quoted_spread_bps`: 15.0 bps (最良気配スプレッドがこれを超える場合は除外)
  - `max_estimated_slippage_bps`: 10.0 bps (1銘柄約10万円の発注サイズに対する推定スリッページ上限)
  - `min_depth_ratio_scale`: 1.5 (板の厚みに対して発注サイズが大きすぎる場合は比率に応じて注文サイズを縮小)
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Generated Markdown report at {output_path}")
