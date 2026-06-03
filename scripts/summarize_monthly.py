import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    target_dir = "/Users/takahashimasatoshi/Library/Mobile Documents/com~apple~CloudDocs/個別株/日米ラグ_2.1/results/20260528_195455_bt_normal"
    csv_path = os.path.join(target_dir, "daily_results.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    # Load daily results
    df = pd.read_csv(csv_path)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    
    # Calculate monthly return (compounded)
    monthly_ret = (1.0 + df['daily_return']).groupby(df.index.to_period('M')).prod() - 1.0
    
    # Convert to DataFrame
    monthly_df = monthly_ret.to_frame(name='monthly_return')
    monthly_df.index.name = 'month'
    
    # Save CSV
    output_csv_path = os.path.join(target_dir, "monthly_returns.csv")
    monthly_df.to_csv(output_csv_path)
    print(f"Saved monthly returns CSV to: {output_csv_path}")
    
    # Create a nice Pivot Table for a heatmap: Years vs Months
    monthly_pivot = monthly_df.copy()
    monthly_pivot['Year'] = monthly_pivot.index.year
    monthly_pivot['Month'] = monthly_pivot.index.month
    pivot_table = monthly_pivot.pivot(index='Year', columns='Month', values='monthly_return')
    
    # Reindex months to 1-12
    pivot_table = pivot_table.reindex(columns=range(1, 13))
    
    # Save a formatted version of the pivot table to CSV as well for easy viewing
    pivot_csv_path = os.path.join(target_dir, "monthly_returns_matrix.csv")
    # Convert to percentages for readability in the matrix CSV
    pivot_pct = pivot_table * 100
    pivot_pct.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Add annual return column
    annual_ret = (1.0 + df['daily_return']).groupby(df.index.year).prod() - 1.0
    pivot_pct['YTD'] = annual_ret * 100
    pivot_pct.to_csv(pivot_csv_path, float_format='%.2f%%')
    print(f"Saved monthly matrix CSV to: {pivot_csv_path}")

    # Plot Monthly Return Heatmap/Table
    fig, ax = plt.subplots(figsize=(12, len(pivot_pct) * 0.6 + 2))
    
    # Hide the axes
    ax.axis('off')
    
    # Generate table content with formatting
    cell_text = []
    cell_colors = []
    
    for year in pivot_pct.index:
        row_text = []
        row_colors = []
        for col in pivot_pct.columns:
            val = pivot_pct.loc[year, col]
            if pd.isna(val):
                row_text.append('-')
                row_colors.append('#f9f9f9')
            else:
                row_text.append(f"{val:+.2f}%")
                if col == 'YTD':
                    # Highlight YTD column
                    if val > 0:
                        row_colors.append('#d4edda') # green
                    else:
                        row_colors.append('#f8d7da') # red
                else:
                    # Heatmap styling for monthly cells
                    if val > 0:
                        # Green scale (max saturation at +10%)
                        alpha = min(val / 10.0, 1.0) * 0.6 + 0.1
                        row_colors.append((0.8, 1.0, 0.8, alpha))
                    else:
                        # Red scale (max saturation at -10%)
                        alpha = min(abs(val) / 10.0, 1.0) * 0.6 + 0.1
                        row_colors.append((1.0, 0.8, 0.8, alpha))
        cell_text.append(row_text)
        cell_colors.append(row_colors)
        
    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=pivot_pct.index,
        colLabels=pivot_pct.columns,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    
    # Style header cells
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#343a40')
            cell.set_text_props(color='white')
            
    plt.title("Monthly Return Analysis (%)", fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    chart_path = os.path.join(target_dir, "monthly_returns_heatmap.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved monthly returns heatmap chart to: {chart_path}")
    
    # Plot standard bar chart for historical monthly returns
    plt.figure(figsize=(15, 6))
    colors = ['#28a745' if x >= 0 else '#dc3545' for x in monthly_df['monthly_return']]
    plt.bar(monthly_df.index.astype(str), monthly_df['monthly_return'] * 100, color=colors, width=0.8)
    plt.title("Historical Monthly Returns (%)", fontsize=14, weight='bold')
    plt.xlabel("Month")
    plt.ylabel("Return (%)")
    plt.xticks(rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    bar_chart_path = os.path.join(target_dir, "monthly_returns_bar.png")
    plt.savefig(bar_chart_path, dpi=300)
    plt.close()
    print(f"Saved monthly returns bar chart to: {bar_chart_path}")

if __name__ == '__main__':
    main()
