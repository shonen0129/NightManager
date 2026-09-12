# フラット・PIT の診断

予期しない `w_final=0` や PIT 履歴不足を調べるときに読む。パスはリポジトリルート基準。

## 分布取得とフラット化

- 対象日の decision ログと結果から、最初に正常経路を外れた箇所を確認する。ログの場所は `src/leadlag/config/paths.py` と対象バッチの出力先から特定する。
- `src/leadlag/models/v2/fallback_policy.py` / `distribution_source.py` で、当日 cache → on-demand BLPX → 終端フラットのどこを通ったかを追う。cache 不在だけでフラット化の原因が確定したとしない。
- gap の実体は有効設定と `src/leadlag/config/paths.py` から解決する。SQLite store または行列ファイルの日付・銘柄順・生成設定を確認する。ファイル名や `latest` の存在だけで鮮度を判断しない。前日行列のコピー・改名は復旧に使わない。
- on-demand は有効フラグだけでなく、BLPX モデル・履歴・当日入力の注入、成功/無効/入力不足/例外を区別する。日次経路の依存がバックテストにも注入されるとは限らない。

## PIT と監査

- 結果の `pit_binning` と `src/leadlag/models/v2/fallback.py` で `history_count`、`fallback_flag`、`multiplier`、履歴日付を確認する。履歴不足の基準は有効設定の `pit_rolling_window` に照合する。固定の1000件や旧保存パスを前提にしない。
- PIT の `fallback_multiplier` と、分布取得失敗・監査失敗による終端フラットを別々に集計する。
- 監査が FAILED なら [leak-audit](../../leak-audit/SKILL.md) を使い、数値監査とリーク監査の下流処理を別々に追う。フラット状態は、全計算が監査 PASS したことを意味しない。

発動条件と、その入力が欠けた・不正になった原因を分けて報告する。安全側のフラット化を解除するために監査・鮮度条件を緩めない。復旧を修正範囲に含む場合は、原因を直した後の正常入力で復帰と制約維持を検証する。
