"""analyze_paradox_robustness.py – 確信度分位別リターンの逆説の頑健性検証

2020年以降にOOS期間を拡張した GP-Sizing のバックテスト結果（oos_returns.csv）を読み込み、
「低確信度（kappa が低い）ほど高リターンになる」という逆説的な現象が
一貫して成立しているか（頑健か）、あるいは特定年・特定レジームの産物かを検証します。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def compute_metrics(ret_series: pd.Series, kappa_series: pd.Series, n_quantiles: int = 3):
    """リターンと確信度のシリーズから分位別の指標を計算する"""
    df = pd.DataFrame({"ret": ret_series, "kappa": kappa_series}).dropna()
    if len(df) < n_quantiles * 5:
        return pd.DataFrame()
        
    df["kappa_quantile"] = pd.qcut(
        df["kappa"], 
        q=n_quantiles, 
        labels=[f"q{i+1}" for i in range(n_quantiles)],
        duplicates="drop"
    )
    
    rows = []
    for group, sub in df.groupby("kappa_quantile", observed=True):
        r = sub["ret"]
        ar = float(r.mean() * 245)
        risk = float(r.std(ddof=1) * np.sqrt(245))
        rr = ar / risk if risk > 1e-8 else float("nan")
        rows.append({
            "quantile": group,
            "kappa_mean": float(sub["kappa"].mean()),
            "n_days": len(r),
            "ar": ar,
            "risk": risk,
            "rr": rr
        })
    return pd.DataFrame(rows).set_index("quantile")

def main():
    parser = argparse.ArgumentParser(description="確信度分位別リターンの逆説検証")
    parser.add_argument("--input", default="results/gp_oos_2020/oos_returns.csv", help="入力CSVファイル")
    parser.add_argument("--output", default="results/gp_oos_2020/paradox_analysis.md", help="出力Markdownファイル")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} が見つかりません。先に walk_forward_gp.py を実行してください。")
        sys.exit(1)
        
    # データ読み込み
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    # 必要なカラムの確認
    if not all(col in df.columns for col in ["net_linear", "net_gp", "kappa", "gross_linear"]):
        print("Error: 必要なカラムがありません。")
        sys.exit(1)
        
    net_gp = df["net_gp"]
    net_lin = df["net_linear"]
    kappa = df["kappa"]
    gross_lin = df["gross_linear"]
    
    df["year"] = df.index.year
    df["vol_20d"] = gross_lin.rolling(20).std() * np.sqrt(245)
    
    # ボラティリティレジームの定義 (全期間ベース)
    valid_vol = df["vol_20d"].dropna()
    df.loc[valid_vol.index, "vol_regime"] = pd.qcut(
        valid_vol, 
        q=3, 
        labels=["LowVol", "MidVol", "HighVol"]
    )
    
    report_lines = []
    report_lines.append("# 確信度分位別リターンの逆説（頑健性）の検証レポート")
    report_lines.append(f"\n対象期間: {df.index[0].date()} 〜 {df.index[-1].date()} (総日数: {len(df)}日)\n")
    
    # 1. 全期間の分位別リターン (ベースライン確認)
    report_lines.append("## 1. 全期間での確信度分位別パフォーマンス")
    report_lines.append("q1: 低確信度（高分散）, q3: 高確信度（低分散）")
    res_all = compute_metrics(net_gp, kappa, n_quantiles=3)
    if not res_all.empty:
        table_str = res_all.to_markdown(floatfmt=".3f")
        report_lines.append(table_str)
    
    # 2. 年別 (Year-by-Year) の検証
    report_lines.append("\n## 2. 年別 (Year-by-Year) の確信度分位別パフォーマンス")
    report_lines.append("各年において「q1 (低確信度) の方が q3 (高確信度) より高リターン」という逆説が成立しているかを確認します。")
    
    years = sorted(df["year"].unique())
    year_summary = []
    
    for y in years:
        df_y = df[df["year"] == y]
        if len(df_y) < 60: continue  # サンプルが少ない年はスキップ
        
        res_y = compute_metrics(df_y["net_gp"], df_y["kappa"], n_quantiles=3)
        if not res_y.empty and "q1" in res_y.index and "q3" in res_y.index:
            q1_ar = res_y.loc["q1", "ar"]
            q3_ar = res_y.loc["q3", "ar"]
            paradox_holds = q1_ar > q3_ar
            
            year_summary.append({
                "Year": y,
                "q1_AR": q1_ar,
                "q3_AR": q3_ar,
                "q1_RR": res_y.loc["q1", "rr"],
                "q3_RR": res_y.loc["q3", "rr"],
                "Paradox Holds?": "Yes" if paradox_holds else "No"
            })
            
    df_ys = pd.DataFrame(year_summary)
    if not df_ys.empty:
        report_lines.append(df_ys.to_markdown(index=False, floatfmt=".3f"))
        
    # 3. レジーム別 (ボラティリティ高/中/低) の検証
    report_lines.append("\n## 3. レジーム別 (ボラティリティ水準) の確信度分位別パフォーマンス")
    report_lines.append("相場のボラティリティレジーム（20日ボラティリティの3分位）ごとに逆説が成立するかを確認します。")
    
    regime_summary = []
    for regime in ["LowVol", "MidVol", "HighVol"]:
        df_r = df[df["vol_regime"] == regime]
        if len(df_r) < 60: continue
        
        res_r = compute_metrics(df_r["net_gp"], df_r["kappa"], n_quantiles=3)
        if not res_r.empty and "q1" in res_r.index and "q3" in res_r.index:
            q1_ar = res_r.loc["q1", "ar"]
            q3_ar = res_r.loc["q3", "ar"]
            paradox_holds = q1_ar > q3_ar
            
            regime_summary.append({
                "Regime": regime,
                "q1_AR": q1_ar,
                "q3_AR": q3_ar,
                "q1_RR": res_r.loc["q1", "rr"],
                "q3_RR": res_r.loc["q3", "rr"],
                "Paradox Holds?": "Yes" if paradox_holds else "No"
            })
            
    df_rs = pd.DataFrame(regime_summary)
    if not df_rs.empty:
        report_lines.append(df_rs.to_markdown(index=False, floatfmt=".3f"))
        
    # 4. 考察と結論のテンプレート
    report_lines.append("\n## 4. 分析結果と結論")
    report_lines.append("※以下の結論は出力結果を見て記述してください。")
    report_lines.append("- 年別の一貫性: 逆説は毎年成立しているか？特定の年だけか？")
    report_lines.append("- レジーム依存性: 高ボラティリティ局面のみの現象か？")
    report_lines.append("- GPモデルの予測分散の意味合い: 本戦略における「不確実性」は「避けるべきリスク」か、「収益機会（リターン源泉）」か？")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"分析完了。レポートを {output_path} に保存しました。")
    print("\n[結果サマリー]")
    if not df_ys.empty:
        print(df_ys.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    if not df_rs.empty:
        print(df_rs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

if __name__ == "__main__":
    main()
