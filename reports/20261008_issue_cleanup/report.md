# Issue #36 / #43 / #48 / #18 / #50 / #51 対応

開始HEADは `a83c0e8f9a143acee68b031f209cb39b6c14d3ff`、worktreeはclean。

## 完了済みissue

#36 / #43 / #48 は既存mainの実装・関連回帰77件と
[最新Hosted CI](https://github.com/shonen0129/NightManager/actions/runs/37728717427) の成功を
照合し、根拠コメントを投稿してcompletedとしてcloseした。
deadline/lease/子孫回収、installed wheelのdeployment root分離、上場後欠損の拒否と
cell別proxy provenanceを確認した。scheduler実市場受入は #27、旧cacheの正規再構築は
別の運用作業として保持する。

## 修正内容

#50: `utils.timestamps` にawareな `jst_now()` と市場日付 `jst_today()` を集約。
daily cutoff、CLI休場判定、cache営業日鮮度、calendar暗黙todayへ適用。
awareな明示日付もJSTへ変換。UTCのTTL・更新記録とdeadlineの時計は維持。
設計は [ADR](../../docs/decisions/2026-10-08-jst-market-clock.md)。

#51: 同一cache writerの重複import 2件と重複writeを除去。
研究ツリーのRuff safe fixを適用し、残ったF841 5件を個別に確認した。
len/set/dict/np.whereの未使用結果4件は計算を除去し、来歴JSONは読込・検証の式を保持。
F401削除対象は標準/外部ライブラリと未使用ラベル・enum・correlation importであり、
producerの呼出やlazy import境界を移動していない。研究のパラメータ・計算・入力を変更せず、
`src/research tools/research` 全体をCI Ruff対象へ追加した。

着手時inventory（Ruff 0.12.4）は112件:
I001=54 / UP015=25 / F401=21 / F841=5 / UP017=2 / F541=3 / F811=2。
safe fix後はF841=5だけ。最終はlockfileのRuff 0.16.2でも0件。
意図的な免除・未対応diagnosticはない。

## 検証

- JST/TZ・calendar/cache・writerの対象回帰: 85 passed。
- close候補の関連回帰: 77 passed。
- 研究既存テスト: 16 passed。
- 固定baseline: 1 passed。残り全tests: 1,143 passed / 2 warnings。合計1,144 passed。
  4 worker、外側deadline 1,800秒、約150秒。全directoryをpytestで収集し、同じ1,144件を確認。
- compileall: 必須5 tree成功。Ruff: production/tests/maintained tools/両研究ツリー成功。
- import-linter: 7契約成功。scheduled Python import境界: 2入口成功。
- 変更文書と本reportのリンク検査: 11参照成功。
- wheel build / 157-module manifest / research除外 / isolated smoke:
  runtime root、CLI、ML artifact推論、ADR roundtrip、shared BLPXの確認に成功。

既存 `.venv/bin/python` はpyenvへのsymlinkで、xdist/mypyがないため、一時環境へ
lockfile版の検証ツールを導入した。global環境へ依存は追加していない。
最初の `-n 4` はplugin不在でexit 4となり、成功には数えていない。
ローカルmypyは環境依存の既存4件（calendar 2 / gap_adjustment 1 / model_meta 1）が残る。
変更前HEADを別directoryへ展開した同環境の検査でも同じ4件、追加エラー0。
ローカルmypy全成功とは報告せず、lockfile環境のHosted CIを最終gateとする。

## GitHub保護設定

ユーザーが設定。2026-10-08のbranch APIで `main protected=true`、required check
`quality-and-tests`、`enforcement_level=everyone`、GitHub Actions `app_id=15368`を確認した。
接続GitHub Appにはadministration権限がなく、詳細protection APIは403。
branch APIで列挙されないPR必須・strict・bypass設定はユーザーがSettings画面で確認。
PR必須・up-to-date・bypass禁止がON、approval要求がOFF、bypass対象なしとして
保存済みとの回答を受けた。APIの観測とユーザーによる画面確認を区別して記録する。

production設定・モデルartifact・live cache・scheduler・発注は変更していない。
