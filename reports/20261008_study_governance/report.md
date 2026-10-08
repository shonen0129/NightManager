# Issue #41: DSR入力とstudy探索履歴

2026-10-08。対象基準HEAD: f08d1fd。実API・発注・本番設定の変更は行っていない。

最新mainにはF16の基本入力拒否とextra_metrics保護が既存実装としてあった。本修正は明示schema/status/frequency/年率、全return系列、Nとvarianceの一致を追加検証し、根拠のないnull variance fallbackを撤去した。事後記録の探索数はlower boundで、DSRは未確認としてNoneにする。

同JSONLに仮説family事前登録、全候補、開始時provenance、失敗/中断/棄却、選択snapshotを保持する。trial outcome/訂正を開始イベントと二重に数えない。templateは固定planを読み、評価前に開始を保存する。

## 過去索引

[history_index.json](history_index.json) はtracked Markdown report 206件をすべて索引化した。既存の感応度・ML探索について読めた候補名とreported countを根拠付きで残す。family下限は感応度35、ML12（6件のコスト比較を12へ追加しない）、未分類unknown。感応度35にはreportが述べる先行25件を含むが、個別のrun provenanceは復元できない。長期候補10件のrank-IC訂正を新しい探索試行としない。各reportは独立trialではなく、履歴完全性はunknown。

## 既存DSRの残件

`var/experiments/registry.jsonl` はGit管理外でこの環境には存在せず、個別の実recordを点検・訂正していない。履歴点検toolと合成回帰で、preview、行別理由、数値再計算/未確認化、append-only訂正、legacy ID保存、idempotenceを検証する。所有環境で実registryに対して実行する必要がある。2026-10-06監査の非null20件を全件誤りと推測せず、Issueはこの点検完了までopenで扱う。

## 検証

対象回帰103件とtemplate経由2件が成功。最終状態の全testsは **1,183件成功**（baseline 1件 + 残り1,182件、4 workers）。compileall、本番/研究Ruff、mypy（152 files）、7 import契約、operational import境界、CI対象文書＋新ADR/レポートの参照（160件）、clean wheel build/verify/installed ML smokeが成功。ロック済みPython 3.12環境を使い、各長時間処理には外側のプロセス全体の停止期限を設定した。

回帰は4returns/T1000・NaN・unknown frequency・invalid status・computed上書きに加え、schema欠落・全評価系列未保存・分散の頻度/件数/値不一致・ゼロ/負/非有限分散・事前登録不備・provenance不一致・開始/中断/棄却の集計・後続失敗を含む選択時DSR・複数段訂正のlegacy ID保持を検証した。外部の実データ取得・実注文は使わず、例外の合成secretも記録へ漏らさないことを確認した。
