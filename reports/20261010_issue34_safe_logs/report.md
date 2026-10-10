# Issue #34: cache・保存例外と再loginの追加回帰

対象: [Issue #34](https://github.com/shonen0129/NightManager/issues/34)。
開始時main: `4c370755e2114c74a3599366e135d842eb78e1f5`。
判断の正本: [credential境界ADR](../../docs/decisions/2026-10-08-credential-boundaries.md)。

## 問題と修正

| 発生条件 | 修正前 | 修正後 |
| --- | --- | --- |
| private keyのpermissionが広い | warningに鍵pathを表示 | permission値と固定警告のみ表示 |
| session/価格cacheのpath解決・I/O失敗、不正timestamp | 生例外を表示、path解決はbest-effort境界外 | 固定の操作分類でwarning、従来のfallback/無効cache拒否を維持 |
| backtest cache/DB保存失敗 | 生例外・causeを保持。commit後のcache失敗でrollback例外に置き換わる | 固定messageのBacktestStoreError、cause抑止、実行中transactionだけrollback |
| CLIのbacktest保存失敗 | 生例外をwarningへ補間 | 固定warning、CSV等の既存best-effort成果物を維持 |
| 初回/再loginの途中復号失敗 | 部分復号URLや前回の認証状態が残る | 開始時に旧状態を解除、4 URLすべて成功してから公開 |

RSA padding順序、診断の成功方式/失敗段階、全方式失敗時のValueError contractは維持する。SQLite commit後のcache失敗でDB audit trailが残る既存契約も維持する。protected session cache内への復号URL保存は既存の意図した動作であり、今回のログ非露出とは別の境界である。

## 合成値による検証

34件の追加回帰を実装した。実資格・実broker API・発注は使用しない。

- cache 7操作 × path/I/O失敗、不正timestamp、成功時のpath/URL非露出: 16件。
- 鍵permission 2ケース、実RSAによる4 URL混在padding成功、各URLでの初回/再login全方式失敗: 11件。
- backtest保存/読込/commit後cache失敗/不正設定の例外型・固定message・traceback: 4件。
- 実CLI経路のstore初期化/DB保存/2回目cache保存失敗でconsole・log・成果物を走査: 3件。

元のmainを別worktreeに置き、追加テストだけを移して対象33件を実行した結果、31件失敗・2件成功（40件対象外）。修正後の関連7ファイルは154件成功。固定messageとcommit後DB残存のassertionを補強した最終2ファイルも52件成功。

## 実行環境とチェック

Python 3.12.14、uv.lock固定のdev・ml-overlay依存を使用した。環境準備中のNumPy読込SIGBUSは取得されたnative binaryの欠損に起因した。同じlockfileのNumPy 2.5.1 wheelを再取得し、lockfile SHA-256・sizeとZIP CRCを確認して復元した。依存version/lockfileの変更やテストskip追加で回避していない。

| チェック | 結果 |
| --- | --- |
| 関連回帰 | 154件成功、補強後52件成功 |
| 固定V2 baseline | 1件成功 |
| 完全suite（baselineを別実行、残りは4 worker） | 実行中 |
| lockfile・compileall | 成功 |
| production/maintained/research Ruff | 成功 |
| production Mypy | 147 source files、成功 |
| import architecture・operational imports | 成功 |
| maintained docsと追加ADR/reportのリンク | 確認中 |
| clean wheel・research除外・installed wheel/ML artifact smoke | 成功 |

詳細log/JUnitはローカルの `var/ci/issue34-safe-logs/` に出力する。公開PRには合成検証の集計のみ記録する。

## 残る運用確認

Issue #34はコード回帰だけでcloseしない。所有者による過去DB/WAL・log・backup・support提出物の残存確認、資格失効/再発行、旧session無効化、隔離/除去の完了証跡はこのPRの対象外であり、所有者限定の運用記録で確認する。
