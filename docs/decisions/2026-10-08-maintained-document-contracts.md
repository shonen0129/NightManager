# 現行運用文書・agent手順の契約検査

Status: accepted
Scope: Issue #47（F26 / F27）

AGENTS.mdを不変条件の正本とし、IDE workflowは作業別Skillと運用手順への入口にする。旧Windsurf改善計画は履歴reportへ保存し、成績や仮説を現行モデルの事実として再公開しない。本番コード・設定の挙動は変更しない。

`scripts/ci/validate_docs.py --current` は現行文書の具体的なrepository pathと `path.py::Class.method` を静的に検査する。対象はAGENTS、README/architecture/CI/scheduler/research環境/roadmap、運用・技術仕様、agent SkillとIDE手順。Skill/IDE文書はglobで発見する。Pythonのimportは実行せずASTの宣言を照合する。runtime出力・placeholder・globと歴史reports/ADRへ現在のAPI適合は強制しない。歴史節を現行文書へ残す場合は明示的なHTML境界を使う。相対リンク・Path table検査は既存のCLI指定文書にも継続する。

意味の対応は自由文のキーワード一致で推測しない。運用・技術仕様のConfig attribute表を継承解決後のAppConfigに照合し、運用Audit case表を実leakage/numerical処理に照合する。通常closeの持越し率と単元丸めはdry-runの合成建玉で確認する。表は期待値の説明であり、新たな設定正本や監査実装ではない。設定・契約変更時は表とテストを同時に更新する。

モデルflat、numerical再監査PASSED、口座在庫ゼロは別の状態として記録する。日付監査は計算窓全体や実時計を保証せず、`held_overnight` は実約定後の建玉を保証しない。actual-live quote/account-risk/durable gateと照合・復旧手順は日次運用手順書に対応付け、既存の実経路テストで検証する。文書CIだけで実運用受入や本番昇格を認定しない。

検証証跡は [Issue #47報告](../../reports/20261008_issue47/report.md) に記録する。
