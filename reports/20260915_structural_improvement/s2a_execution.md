# S2a 実行記録：macro / ADR / 9:10 I/O境界

- 実施日: 2026-09-15
- 判定: **PASS（構造抽出・既存挙動同値の範囲）**
- 対象: 修正済み作業ツリー（未commit差分を保持）

## 実施内容

| 旧責務 | S2aの正本 | 実施内容 |
|---|---|---|
| `core.macro`のyfinance取得・timeout・process cache | `data.macro` | `load_macro_prices` / `load_macro_returns` と `clear_macro_cache`を移設。coreからnetwork/cache依存を削除 |
| `MLOrderOverlayModel`のADR pickle読込・鮮度判定 | `data.adr_features` | `load_adr_features`へ移設。modelはDataFrame/Noneだけを利用 |
| `preprocessor`の5分足cache読込と09:10抽出 | `data.intraday_inputs` | `build_5m_910_prices`、`build_open_910_returns`を移設。主要本番・BT・研究入口を新adapterへ切替 |
| h=1 target計算 | `preprocessor`の計算関数 + `data.intraday_inputs`の入力取得 | 既存の欠損・非正価格・09:00→09:05→09:10 fallbackを維持し、open→09:10 returnsを明示入力化 |

## 撤去・移行方針

`models.production_v2.download_macro_prices`の公開再exportと、coreからのmacro I/O呼出を
撤去した。既存の`preprocessor.compute_jp_target_returns`は外部利用者を壊さないための
一段compatibility entryとして残るが、内部の主要利用者は`data.intraday_inputs`へ移行済み。
これは無期限の中継を正当化するものではなく、S2bで時点付き入力を導入し、利用者・
保存形式を確認してから撤去する対象として計画に明記した。

S2aではavailable_atの証拠がない過去データに新しいPIT時刻を付与していない。
cache shimの撤去、`DecisionInputs`版付け、read-only所有権、PIT label可視範囲はS2bへ残す。

## 検証

| 検証 | 結果 |
|---|---|
| S2a境界・macro・ADR・9:10・backtester回帰 | **42 passed** |
| S1完了確認（設定/factory/Runner/BT/V2回帰） | **82 passed** |
| `tests/` 全体 | **664 passed / 18 warnings / 719.05秒** |
| import-linter | **4 kept / 0 broken** |
| `compileall`（`src/leadlag src/research tools tests`） | PASS |
| plan/ADR文書検証 | PASS |

## 受入条件の判定

- macroの計算層にnetwork/cache I/Oがない: **PASS**
- ADR読込がmodel層にない: **PASS**
- h=1のtarget数値と欠損fallbackを既存テストで固定: **PASS**
- 本番・BT・ML overlay・gap生成の取得入口をadapterへ切替: **PASS**
- available_atの新規仮定を導入しない: **PASS**

S2a完了後も、入力の版付き契約・PIT可視範囲・read-only化を実装していない。これらを
追加して初めてS2全体の完了とする。

## S2a 完了再確認（2026-09-16）

S2b着手前のS2a境界を再確認し、`tests/unit/test_s2a_input_boundaries.py`を含む対象回帰を
再実行した。S2aで定めたmacro/ADR/9:10の取得adapter、h=1の欠損・非正価格fallback、
主要経路のintraday adapter利用に追加の後退はなく、対象はPASSのまま維持される。
版付き契約、read-only所有権、PIT可視範囲の追加はS2bの変更として別記録に分離した。
