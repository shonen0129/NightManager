# 指値エントリーバックテスト: 仮定・モデル定義メモ

生成日時: 2026-06-02 15:42:05

## 1. r_hat_cc の定義

`compute_signal()` が返す `r_hat_jp_cc` を使用:

```
r_hat_jp_cc = mu_jp + sigma_jp * z_hat_j_t1
```

- `mu_jp`: ウィンドウ内の JP 銘柄 CC リターンの EWMA 平均
- `sigma_jp`: 同 EWMA 標準偏差  
- `z_hat_j_t1`: PCA 予測標準化リターン（V_J^K × f_t）

これはギャップ補正前の**PCA 予測 Close-to-Close リターン**（§4.6 ステップ3）。

最終シグナル `s_j`（`gap_residual` モード）はギャップを差し引いた残差であり `s_j ≠ r_hat_cc`。

フェア価格の算出: `P_fair[j] = P_close[t-1, j] × (1 + r_hat_cc[j])`

## 2. 指値価格の定義

### 主系列 (theory basis)
- フェア価格: `P_fair = P_close_prev × (1 + r_hat_cc)`
- ロング指値: `P_fair × (1 - m/10000)` （m bps 安く買う）
- ショート指値: `P_fair × (1 + m/10000)` （m bps 高く売る）
- m ∈ [0, 5, 10, 20, 30, 50]

### 対照群 (prev_close basis)
- ロング指値: `P_close_prev × (1 - k/10000)`
- ショート指値: `P_close_prev × (1 + k/10000)`  
- k ∈ [-10, -5, 0, 5, 10, 20]（負=不利側で約定しやすい, 正=有利側）

## 3. 約定モデル

使用データ: 当日の日足 High (P_high) / Low (P_low)

| 方向 | 約定条件 | 約定価格 |
|------|----------|----------|
| ロング(買い) | `P_low ≤ 指値` | `min(指値, P_open)`（寄付き時点で有利なら P_open） |
| ショート(売り) | `P_high ≥ 指値` | `max(指値, P_open)`（寄付き時点で有利なら P_open） |

**限界・注意点:**
- 日足 High/Low は日中いずれかの時点での最高値/最安値。到達タイミング不明。
- 指値が日中の High/Low に到達している場合でも、実際の執行は指値価格で可能とは限らない（板薄の場合）。
- 本モデルは保守的ではなく「理論的に可能な最良ケース」に近い。過楽観バイアスに注意。

## 4. スリッページ仮定

| バリアント | エントリー側 | 決済側 | 往復コスト |
|-----------|-------------|--------|-----------|
| slip_on_entry=True（保守） | 5bps | 5bps | 10bps × gross |
| slip_on_entry=False（指値） | 0bps | 5bps | 5bps × gross |
| ベースライン(成行) | 5bps | 5bps | 10bps × gross |

## 5. ウェイト処理

- **主系列 (renormalize=False)**: 元の目標ウェイト `w_j` をそのまま使用。約定しなかった銘柄は0。グロスエクスポージャーが目減りする。
- **再正規化版 (renormalize=True)**: 約定銘柄のウェイトを正規化し、ロング合計=+1, ショート合計=-1 に戻す。グロスは2を維持。

## 6. 戦略設定（既存ロジックと同一）

| パラメータ | 値 |
|-----------|-----|
| signal_mode | gap_residual |
| gap_open_coef | 0.7 |
| topix_beta_coef | 0.6 |
| weight_mode | signal |
| dispersion_filter | False |
| q | 0.3 |
| K | 6 |
| corr_window | 60 |
| ewma_half_life | 45 |
| lambda_reg | 0.75 |
| lambda_lw | 0.5 |
| slippage_bps | 5.0 |

## 7. バックテスト期間

- 全期間: 2015-01-01 〜 データ末尾
- OOS 期間: 2020-01-01 〜 データ末尾

## 8. ルックアヘッドバイアス対策

- [x] 指値価格: `P_close[t-1]`（前日終値）と `r_hat_cc`（前日の US 情報で計算）のみ
- [x] 約定判定: 当日の `P_high`, `P_low` のみ使用
- [x] ウェイト: 前日シグナルで決定（既存ロジックと同一）
- [x] 決済: 当日終値 = `P_open × (1 + r_oc)`（r_oc = Close/Open - 1）
