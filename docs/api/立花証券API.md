# 立花証券・ｅ支店・ＡＰＩ専用ページ（v4.9-000 at 2026.05.16）
立花証券・ｅ支店・ＡＰＩは無料で利用できる日本株 API です。取引や株価・ニュースの取得を高速に処理できます。

## e_api refference manual 目次
1. 共通説明
    1. ｅ支店・ＡＰＩ専用ＵＲＬ
    2. インタフェース概要
    3. ブラウザからの利用方法
    4. 共通項目、認証機能
    5. マスタデータ利用方法
    6. EXCEL(VBA)からの利用方法
2. 認証機能（認証I/F）
    1. ログイン
    2. ログアウト
3. 業務機能（REQUEST I/F）
    1. 株式新規注文
    2. 株式訂正注文
    3. 株式取消注文
    4. 株式一括取消
    5. 現物保有銘柄一覧
    6. 信用建玉一覧
    7. 買余力
    8. 建余力＆本日維持率
    9. 売却可能数量
    10. 注文一覧
    11. 注文約定一覧（詳細）
    12. 可能額サマリー
    13. 可能額推移
    14. 現物株式買付可能額詳細
    15. 信用新規建て可能額詳細
    16. リアル保証金率
4. マスタ機能（REQUEST I/F）
    1. マスタ情報ダウンロード
    2. マスタ情報問合取得
    3. ニュースヘッダー問合取得
    4. ニュースボディー問合取得
    5. 銘柄詳細情報問合取得
    6. 証金残情報問合取得
    7. 信用残情報問合取得
    8. 逆日歩情報問合取得
5. 時価情報機能（REQUEST I/F）
    1. 時価情報問合取得
    2. 蓄積情報問合取得
6. 注文約定通知（EVENT I/F）
7. 結果コード、警告コード表
8. リンク
    1. 立花証券ｅ支店
    2. ＡＰＩサービス開始のご案内
    3. ｅ支店・ＡＰＩ専用ページ

---

## 1. 共通説明

### 1. ｅ支店・ＡＰＩ専用ＵＲＬ
| URL | 環境・バージョン |
| ------ | ------ |
| https://kabuka.e-shiten.jp/e_api_v4r9/ | （本番環境、新バージョン） |
| https://demo-kabuka.e-shiten.jp/e_api_v4r9/ | （デモ環境、新バージョン） |
| https://kabuka.e-shiten.jp/e_api_v4r8/ | （本番環境、旧バージョン） |
| https://demo-kabuka.e-shiten.jp/e_api_v4r8/ | （デモ環境、旧バージョン） |

**【注意事項】**
* 現在リリースのバージョンはｅ支店・ＡＰＩ（ｖ４ｒ８、ｖ４ｒ９）です。旧バージョンの廃止予定等案内は 立花証券・ｅ支店・ＡＰＩ専用ページ「２．リリース＆改定情報」を参照願います。
* 保守や機能追加等によりＵＲＬのＰｒｅｆｉｘ（e_api_v4rN 部分）を e_api_v4rN（リビジョンＮ）またはe_api_vN（バージョンＮ）として平行リリースします。
* 後続版の平行リリース後６０日前後（保守作業は非営業日に実施）旧リリース版の利用を停止（廃止）しますので、後続版リリース後はお早めにお客様プログラムの後続版への対応をお願いいたします。
* 後続版リリース等の連絡は本ページ・リリース＆改定情報にてお知らせするため、該当ページを定期的に参照願います。
* 本マニュアルページからのリンク資料について改定がない場合は旧バージョンのままとなります。改定時は該当バージョンとして掲載いたしますので、最新版をご利用願います。※改定版公開のタイミングで旧版は自動的に廃版とさせて頂きます。

**【デモ環境】**
お客様のテスト環境としてｅ支店・デモ環境に専用環境を用意いたしました。ご利用時間帯等につきましては こちら を参照願います。

**【アクセス方法】**
* 認証機能は 「ｅ支店・ＡＰＩ専用ＵＲＬ/auth/?{引数}」
* 業務機能は 「仮想URL（REQUEST）?{引数}」
* マスタ機能は 「仮想URL（MASTER）?{引数}」
* 時価情報機能は「仮想URL（PRICE）?{引数}」
※{引数｝に JSON 文字列形式で要求を指定、指定項目等の説明は各機能説明を参照。
※認証機能例：「https://kabuka.e-shiten.jp/e_api_vNrN/auth/?{"p_no":"1","p_sd_date":"yyyy.mm.dd-hh:mn:ss.ttt","sCLMID":"CLMAuthLoginRequest","sAuthId":"oxox"}」

### 2. インタフェース概要
最初に以下資料を参照頂き本ＡＰＩの概要や構成についてご理解の上その後、必要に応じ各マニュアルをお読み下さい。
立花証券・ｅ支店・ＡＰＩ（ｖ４ｒ９）、インタフェース概要

### 3. ブラウザからの利用方法
プログラム開発前にとりあえずｅ支店・ＡＰＩをブラウザで試して見たい方は以下方法で試せます。
立花証券・ｅ支店・ＡＰＩ（ｖ４ｒ９）、ブラウザからの利用方法
以下シート中に「基本」・・・・・・・・認証、注文入力、注文一覧、「マスタ・時価」・・・・マスタ情報問合取得、 時価情報問合取得、「時価配信」・・・・・・EVENT I/F を利用した時価情報配信機能の利用方法、「ニュース」・・・・・・ニュース問合取得について記載しています。

### 4. 共通項目、認証機能
共通項目及び認証機能の使用例については以下資料を参照下さい。
立花証券・ｅ支店・ＡＰＩ（ｖ４ｒ９）、REQUEST I/F、利用方法、データ仕様

### 5. マスタデータ利用方法
マスタ情報ダウンロード機能等で取得可能な各種マスタデータについては以下を参照。
立花証券・ｅ支店・ＡＰＩ（ｖ４ｒ５）、REQUEST I/F、マスタデータ利用方法
ダウンロード要求で通知されるデータ項目等についてはマスタ情報ダウンロード の２．以降を参照。

### 6. EXCEL(VBA)からの利用方法
EXCEL(VBA)からｅ支店・ＡＰＩをご利用頂くためのサンプルモジュール（e_api_v4r9.bas）及びそれを使用した EXCEL サンプルプログラム（ｖ４ｒ９） で時価情報を取得することができます。ご利用方法等は EXCEL サンプルプログラム（シート記載内容）を御覧下さい。

---

## 2. 認証機能（認証I/F）

### 1. ログイン

#### 1. 要求
```json
{
"sCLMID":"CLMAuthLoginRequest",
"sAuthId":"authid"
}
```
* **sCLMID** (機能ＩＤ): CLMAuthLoginRequest
* **sAuthId** (認証ＩＤ): ｅ支店・ＡＰＩ、利用設定画面で生成した値

【注意】アクセス方法は URL に「ｅ支店・ＡＰＩ専用ＵＲＬ/auth/?{引数}」で指定、{引数}に JSON 文字列形式で要求を指定。詳細は ｅ支店・ＡＰＩ専用ＵＲＬ、インタフェース概要、ブラウザからの利用方法、共通項目、認証機能 参照。
【注意】要求または応答の項目順番については処理系により並べ替え操作されるため JSON 仕様準拠とし保証（記載順番と一致）しない（しなくても問題ない）。例：以下は同じ要求、または応答として処理する。
`{"項目A":"値A","項目B":"値B"}`
`{"項目B":"値B","項目A":"値A"}`

#### 2. 応答
```json
{
"sCLMID":"CLMAuthLoginAck",
"sResultCode":"0",
"sResultText":"",
"sZyoutoekiKazeiC":"1",
"sSecondPasswordOmit":"1",
"sLastLoginDate":"20231002075554",
"sSogoKouzaKubun":"1",
"sHogoAdukariKouzaKubun":"1",
"sFurikaeKouzaKubun":"1",
"sGaikokuKouzaKubun":"1",
"sMRFKouzaKubun":"0",
"sTokuteiKouzaKubunGenbutu":"2",
"sTokuteiKouzaKubunSinyou":"2",
"sTokuteiKouzaKubunTousin":"2",
"sTokuteiHaitouKouzaKubun":"1",
"sTokuteiKanriKouzaKubun":"0",
"sSinyouKouzaKubun":"1",
"sSakopKouzaKubun":"0",
"sMMFKouzaKubun":"0",
"sTyukokufKouzaKubun":"0",
"sKawaseKouzaKubun":"0",
"sHikazeiKouzaKubun":"0",
"sKinsyouhouMidokuFlg":"0",
"sUrlRequest":"UrAza1ei3lrkqQJziA0HLmqA9wvca/+kPJ7wYaVHdrOC2IQnBf/.......",
"sUrlMaster":"rw76qR3F2L29Vg3NaOEUGtFDuSwyWnVi9xE+M5V9MuNvAjwqmT38+2.....",
"sUrlPrice":"wcLIXN1dQX8WFQ/yd/OsbFOU2Nh5l2y+iCfS0+PWT6yTg3/y+OxdIk......",
"sUrlEvent":"pZ3S1k1002xPfjkQPAGGJKKTiBnwZ4417Z6n9+qxlnn5FMBl00L7/+kP....",
"sUrlEventWebSocket":"aRESx74EIyNmkreNVxXF0ScnaPIrFi9xVod1rxkIOb2ybeS....",
"sUpdateInformWebDocument":"20241001",
"sUpdateInformAPISpecFunction":"20250531"
}
```

* **sCLMID** (機能ＩＤ): CLMAuthLoginAck
* **sResultCode** (結果コード): 業務処理．エラーコード (0:正常 上記以外は 結果コード、警告コード表 参照)
* **sResultText** (結果テキスト): 「結果コード」に対応するテキスト (正常:"")
* **sZyoutoekiKazeiC** (譲渡益課税区分): 1：特定, 3：一般, 5：NISA
* **sSecondPasswordOmit** (暗証番号省略有無Ｃ): 0：無 ※0：固定値とする。各注文入力において第二パスワードの入力が必須。
* **sLastLoginDate** (最終ログイン日時): YYYYMMDDHHMMSS または 00000000000000
* **sSogoKouzaKubun** (総合口座開設区分): 0：未開設, 1：開設
* **sHogoAdukariKouzaKubun** (保護預り口座開設区分): 0：未開設, 1：開設
* **sFurikaeKouzaKubun** (振替決済口座開設区分): 0：未開設, 1：開設
* **sGaikokuKouzaKubun** (外国口座開設区分): 0：未開設, 1：開設
* **sMRFKouzaKubun** (ＭＲＦ口座開設区分): 0：未開設, 1：開設
* **sTokuteiKouzaKubunGenbutu** (特定口座区分現物): 0：一般, 1：特定源泉徴収なし, 2：特定源泉徴収あり
* **sTokuteiKouzaKubunSinyou** (特定口座区分信用): 0：一般, 1：特定源泉徴収なし, 2：特定源泉徴収あり
* **sTokuteiKouzaKubunTousin** (特定口座区分投信): 0：一般, 1：特定源泉徴収なし, 2：特定源泉徴収あり
* **sTokuteiHaitouKouzaKubun** (配当特定口座区分): 0：未開設, 1：開設
* **sTokuteiKanriKouzaKubun** (特定管理口座開設区分): 0：未開設, 1：開設
* **sSinyouKouzaKubun** (信用取引口座開設区分): 0：未開設, 1：開設
* **sSakopKouzaKubun** (先物ＯＰ口座開設区分): 0：未開設, 1：開設
* **sMMFKouzaKubun** (ＭＭＦ口座開設区分): 0：未開設, 1：開設
* **sTyukokufKouzaKubun** (中国Ｆ口座開設区分): 0：未開設, 1：開設
* **sKawaseKouzaKubun** (為替保証金口座開設区分): 0：未開設, 1：開設
* **sHikazeiKouzaKubun** (非課税口座開設区分): 0：未開設, 1：開設 ※ＮＩＳＡ口座の開設有無を示す。
* **sKinsyouhouMidokuFlg** (金商法交付書面未読フラグ): 1：未読, 0：既読 ※未読の場合、ｅ支店・ＡＰＩは利用不可のため仮想ＵＲＬは発行されず""を設定。
* **sUrlRequest** (仮想URL REQUEST): 業務機能 （REQUEST I/F）仮想URL
* **sUrlMaster** (仮想URL MASTER): マスタ機能 （REQUEST I/F）仮想URL
* **sUrlPrice** (仮想URL PRICE): 時価情報機能（REQUEST I/F）仮想URL
* **sUrlEvent** (仮想URL EVENT): 注文約定通知（EVENT I/F）仮想URL
* **sUrlEventWebSocket** (仮想URL EVENT-WebSocket): 注文約定通知（EVENT I/F WebSocket版）仮想URL
* **sUpdateInformWebDocument** (交付書面更新予定日): 標準Ｗｅｂの交付書面更新予定日決定後、該当日付を設定。
* **sUpdateInformAPISpecFunction** (ｅ支店・ＡＰＩリリース予定日): ｅ支店・ＡＰＩリリース予定日決定後、該当日付を設定。

【注意１】仮想ＵＲＬは利用設定画面で登録した公開鍵で暗号化した値です。仮想ＵＲＬを受信後、登録した公開鍵とペアとなる秘密鍵で複合化した値を仮想ＵＲＬとしてご利用ください。
【注意２】本項目は該当事象の予定日をログイン応答を利用し事前にお知らせするための項目です。

### 2. ログアウト

#### 1. 要求
```json
{
"sCLMID":"CLMAuthLogoutRequest"
}
```
* **sCLMID** (機能ＩＤ): CLMAuthLogoutRequest

#### 2. 応答
```json
{
"sCLMID":"CLMAuthLogoutAck",
"sResultCode":"0",
"sResultText":""
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMAuthLogoutAck |
| sResultCode | 結果コード | CLMAuthLoginAck.sResultCode 参照 |
| sResultText | 結果テキスト | CLMAuthLoginAck.sResultText 参照 |

## 3. 業務機能（REQUEST I/F）

### 1. 株式新規注文

#### 1. 要求
```json
{
"sCLMID":"CLMKabuNewOrder",
"sZyoutoekiKazeiC":"1",
"sIssueCode":"8411",
"sSizyouC":"00",
"sBaibaiKubun":"3",
"sCondition":"0",
"sOrderPrice":"0",
"sOrderSuryou":"100",
"sGenkinShinyouKubun":"0",
"sOrderExpireDay":"0",
"sGyakusasiOrderType":"0",
"sGyakusasiZyouken":"0",
"sGyakusasiPrice":"*",
"sTatebiType":"*",
"sTategyokuZyoutoekiKazeiC":"*",
"sSecondPassword":"pswd",
"aCLMKabuHensaiData":
[
{
"sTategyokuNumber":"999999",
"sTatebiZyuni":"1",
"sOrderSuryou":"100"
}
]
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMKabuNewOrder |
| sZyoutoekiKazeiC | 譲渡益課税区分 | 1：特定 3：一般 5：NISA（一般NISAの売買は2023年まで可、2024年以降は売却のみ可） 6：N成長（2024年から取り扱い開始、NISA成長投資枠） |
| sIssueCode | 銘柄コード | 例:6501 |
| sSizyouC | 市場 | 00：東証 |
| sBaibaiKubun | 売買区分 | 1：売 3：買 5：現渡 7：現引 |
| sCondition | 執行条件 | 0：指定なし 2：寄付 4：引け 6：不成 |
| sOrderPrice | 注文値段 | *：指定なし 0：成行 上記以外は、注文値段 |
| sOrderSuryou | 注文株数 | 例:100 |
| sGenkinShinyouKubun | 現金信用区分 | 0：現物 2：新規(制度信用6ヶ月) 4：返済(制度信用6ヶ月) 6：新規(一般信用6ヶ月) 8：返済(一般信用6ヶ月) |
| sOrderExpireDay | 注文期日 | 0：当日 上記以外は、注文期日(YYYYMMDD)[10営業日迄] |
| sGyakusasiOrderType | 逆指値注文種別 | 0：通常 1：逆指値 2：通常＋逆指値 |
| sGyakusasiZyouken | 逆指値条件 | 0：指定なし 条件値段（トリガー価格） |
| sGyakusasiPrice | 逆指値値段 | *：指定なし 0：成行 上記以外は逆指値値段 |
| sTatebiType | 建日種類 | 信用返済時に指定する返済建玉順序種類指定 *：指定なし（現物または新規） 1：個別指定 2：建日順 3：単価益順 4：単価損順 |
| sTategyokuZyoutoekiKazeiC | 建玉譲渡益課税区分 | 信用建玉における譲渡益課税区分（現引、現渡で使用） *：現引、現渡以外の取引 1：特定 3：一般 |
| sSecondPassword | 第二パスワード | 第二暗証番号（発注パスワード） |
| aCLMKabuHensaiData | 返済リスト | 信用返済（建日種類＝個別指定）時の返済建玉リスト、他取引時不要。返済建玉リストとして以下３項目を配列指定する |
| sTategyokuNumber | 新規建玉番号 | 信用建玉番号（CLMShinyouTategyokuList.sOrderTategyokuNumber） |
| sTatebiZyuni | 建日順位 | 約定時返済する返済リスト内順位（１から昇順） |
| sOrderSuryou | 注文数量 | 返済建玉株数 |

#### 2. 応答
```json
{
"sCLMID":"CLMKabuNewOrder",
"sResultCode":"0",
"sResultText":"",
"sWarningCode":"0",
"sWarningText":"",
"sOrderNumber":"9000015",
"sEigyouDay":"20221209",
"sOrderUkewatasiKingaku":"140099",
"sOrderTesuryou":"90",
"sOrderSyouhizei":"9",
"sKinri":"-",
"sOrderDate":"20221209134803"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMKabuNewOrder |
| sResultCode | 結果コード | 業務処理．エラーコード 0：正常 上記以外は 結果コード、警告コード表 参照 |
| sResultText | 結果テキスト | 「結果コード」に対応するテキスト 正常:"" |
| sWarningCode | 警告コード | 業務処理．ワーニングコード 0：正常 上記以外は 結果コード、警告コード表 参照 |
| sWarningText | 警告テキスト | 「警告コード」に対応するテキスト 正常:"" |
| sOrderNumber | 注文番号 | 採番（注文番号＋営業日でユニーク） |
| sEigyouDay | 営業日 | YYYYMMDD |
| sOrderUkewatasiKingaku | 注文受渡金額 | 0～9999999999999999 |
| sOrderTesuryou | 注文手数料 | 0～9999999999999999 |
| sOrderSyouhizei | 注文消費税 | 0～9999999999999999 |
| sKinri | 金利 | メモリ上のシステム市場弁済別取扱条件 0～999.99999：買方金利 0～999.99999：売方金利 0～999.99999：買方金利（翌営業日） 0～999.99999：売方金利（翌営業日） -：現物取引場合 |
| sOrderDate | 注文日時 | YYYYMMDDHHMMSS |


### 2. 株式訂正注文

#### 1. 要求
```json
{
"sCLMID":"CLMKabuCorrectOrder",
"sOrderNumber":"9000015",
"sEigyouDay":"20221209",
"sCondition":"0",
"sOrderPrice":"0",
"sOrderSuryou":"*",
"sOrderExpireDay":"*",
"sGyakusasiZyouken":"*",
"sGyakusasiPrice":"*",
"sSecondPassword":"pswd"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMKabuCorrectOrder |
| sOrderNumber | 注文番号 | CLMKabuNewOrder.sOrderNumber |
| sEigyouDay | 営業日 | CLMKabuNewOrder.sEigyouDay |
| sCondition | 執行条件 | *：変更なし 0：指定なし 2：寄付 4：引け 6：不成 |
| sOrderPrice | 注文値段 | *：変更なし 0：成行に変更 訂正注文値段：指値を変更 |
| sOrderSuryou | 注文数量 | *：変更なし 訂正数量：数量を変更（増株不可） ※訂正数量には、内出来を含んだ数量を指定 |
| sOrderExpireDay | 注文期日 | *：変更なし 0：当日 変更注文期日日(YYYYMMDD)[10営業日迄] |
| sGyakusasiZyouken | 逆指値条件 | *：変更なし 0：成行に変更 逆指値条件：逆指値条件を変更 |
| sGyakusasiPrice | 逆指値注文値段 | *：変更なし 0：成行に変更 逆指値注文値段：逆指値注文値段を変更 |
| sSecondPassword | 第二パスワード | 第二暗証番号（発注パスワード） |

#### 2. 応答
```json
{
"sCLMID":"CLMKabuCorrectOrder",
"sResultCode":"0",
"sResultText":"",
"sOrderNumber":"9000015",
"sEigyouDay":"20221209",
"sOrderUkewatasiKingaku":"140099",
"sOrderTesuryou":"90",
"sOrderSyouhizei":"9",
"sOrderDate":"20221209134803"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMKabuCorrectOrder |
| sResultCode | 結果コード | CLMKabuNewOrder.sResultCode 参照 |
| sResultText | 結果テキスト | CLMKabuNewOrder.sResultText 参照 |
| sOrderNumber | 注文番号 | 要求設定値 |
| sEigyouDay | 営業日 | 要求設定値 |
| sOrderUkewatasiKingaku | 注文受渡金額 | 0～9999999999999999 |
| sOrderTesuryou | 注文手数料 | 0～9999999999999999 |
| sOrderSyouhizei | 注文消費税 | 0～9999999999999999 |
| sOrderDate | 注文日時 | YYYYMMDDHHMMSS |


### 3. 株式取消注文

#### 1. 要求
```json
{
"sCLMID":"CLMKabuCancelOrder",
"sOrderNumber":"30000007",
"sEigyouDay":"20200727",
"sSecondPassword":"pswd"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMKabuCancelOrder |
| sOrderNumber | 注文番号 | CLMKabuNewOrder.sOrderNumber |
| sEigyouDay | 営業日 | CLMKabuNewOrder.sEigyouDay |
| sSecondPassword | 第二パスワード | 第二暗証番号（発注パスワード） |

#### 2. 応答
```json
{
"sCLMID":"CLMKabuCancelOrder",
"sResultCode":"0",
"sResultText":"",
"sOrderNumber":"30000007",
"sEigyouDay":"20200727",
"sOrderUkewatasiKingaku":"140099",
"sOrderDate":"20221209134803"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMKabuCancelOrder |
| sResultCode | 結果コード | CLMKabuNewOrder.sResultCode 参照 |
| sResultText | 結果テキスト | CLMKabuNewOrder.sResultText 参照 |
| sOrderNumber | 注文番号 | 要求設定値 |
| sEigyouDay | 営業日 | 要求設定値 |
| sOrderUkewatasiKingaku | 注文受渡金額 | 0～9999999999999999 |
| sOrderDate | 注文日時 | YYYYMMDDHHMMSS |


### 4. 株式一括取消

#### 1. 要求
```json
{
"sCLMID":"CLMKabuCancelOrderAll",
"sSecondPassword":"pswd"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMKabuCancelOrderAll |
| sSecondPassword | 第二パスワード | 第二暗証番号（発注パスワード） |

#### 2. 応答
```json
{
"sCLMID":"CLMKabuCancelOrderAll",
"sResultCode":"0",
"sResultText":""
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMKabuCancelOrderAll |
| sResultCode | 結果コード | CLMKabuNewOrder.sResultCode 参照 |
| sResultText | 結果テキスト | CLMKabuNewOrder.sResultText 参照 |


### 5. 現物保有銘柄一覧

#### 1. 要求
```json
{
"sCLMID":"CLMGenbutuKabuList",
"sIssueCode":"7201"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMGenbutuKabuList |
| sIssueCode | 銘柄コード | 指定あり：指定１銘柄のリスト取得（例:"7201"） 指定なし：全保有銘柄のリスト取得（例:""） |

#### 2. 応答
```json
{
"sCLMID":"CLMGenbutuKabuList",
"sResultCode":"0",
"sResultText":"",
"sWarningCode":"0",
"sWarningText":"",
"sIssueCode":"7201",
"sIppanGaisanHyoukagakuGoukei":"0",
"sIppanGaisanHyoukaSonekiGoukei":"0",
"sNisaGaisanHyoukagakuGoukei":"0",
"sNisaGaisanHyoukaSonekiGoukei":"0",
"sNseityouGaisanHyoukagakuGoukei":"0",
"sNseityouGaisanHyoukaSonekiGoukei":"0",
"sTokuteiGaisanHyoukagakuGoukei":"8315050",
"sTokuteiGaisanHyoukaSonekiGoukei":"-810050",
"sTotalGaisanHyoukagakuGoukei":"8315050",
"sTotalGaisanHyoukaSonekiGoukei":"-810050",
"aGenbutuKabuList":
[
{
"sUriOrderWarningCode":"0",
"sUriOrderWarningText":"",
"sUriOrderIssueCode":"7201",
"sUriOrderZyoutoekiKazeiC":"1",
"sUriOrderZanKabuSuryou":"4200",
"sUriOrderUritukeKanouSuryou":"4200",
"sUriOrderGaisanBokaTanka":"727.0000",
"sUriOrderHyoukaTanka":"598.0000",
"sUriOrderGaisanHyoukagaku":"2511600",
"sUriOrderGaisanHyoukaSoneki":"-541800",
"sUriOrderGaisanHyoukaSonekiRitu":"-17.74",
"sSyuzituOwarine":"0",
"sZenzituHi":"0",
"sZenzituHiPer":"0",
"sUpDownFlag":"06",
"sNissyoukinKasikabuZan":"0"
}
]
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMGenbutuKabuList |
| sResultCode | 結果コード | CLMKabuNewOrder.sResultCode 参照 |
| sResultText | 結果テキスト | CLMKabuNewOrder.sResultText 参照 |
| sWarningCode | 警告コード | CLMKabuNewOrder.sWarningCode 参照 |
| sWarningText | 警告テキスト | CLMKabuNewOrder.sWarningText 参照 |
| sIssueCode | 銘柄コード | 要求設定値 |
| sIppanGaisanHyoukagakuGoukei | 概算評価額合計 (一般口座残高) | 0～9999999999999999 |
| sIppanGaisanHyoukaSonekiGoukei | 概算評価損益合計(一般口座残高) | -999999999999999～9999999999999999 |
| sNisaGaisanHyoukagakuGoukei | 概算評価額合計 (NISA口座残高) | 0～9999999999999999 |
| sNisaGaisanHyoukaSonekiGoukei | 概算評価損益合計(NISA口座残高) | -999999999999999～9999999999999999 |
| sNseityouGaisanHyoukagakuGoukei | 概算評価額合計 (N成長口座残高) | 0～9999999999999999 |
| sNseityouGaisanHyoukaSonekiGoukei | 概算評価損益合計(N成長口座残高) | -999999999999999～9999999999999999 |
| sTokuteiGaisanHyoukagakuGoukei | 概算評価額合計 (特定口座残高) | 0～9999999999999999 |
| sTokuteiGaisanHyoukaSonekiGoukei | 概算評価損益合計(特定口座残高) | -999999999999999～9999999999999999 |
| sTotalGaisanHyoukagakuGoukei | 概算評価額合計 (残高合計) | 0～9999999999999999 |
| sTotalGaisanHyoukaSonekiGoukei | 概算評価損益合計(残高合計) | -999999999999999～9999999999999999 |
| aGenbutuKabuList | 現物保有リスト | 以下項目を配列で応答、情報が無い場合は"" |
| sUriOrderWarningCode | 警告コード | CLMKabuNewOrder.sWarningCode 参照 |
| sUriOrderWarningText | 警告テキスト | CLMKabuNewOrder.sWarningText 参照 |
| sUriOrderIssueCode | 銘柄コード | 保有銘柄コード |
| sUriOrderZyoutoekiKazeiC | 譲渡益課税区分 | CLMKabuNewOrder.sZyoutoekiKazeiC 参照 |
| sUriOrderZanKabuSuryou | 残高株数 | 0～9999999999999 |
| sUriOrderUritukeKanouSuryou | 売付可能株数 | 0～9999999999999 |
| sUriOrderGaisanBokaTanka | 概算簿価単価 | 0.0000～999999999.9999 |
| sUriOrderHyoukaTanka | 評価単価 | 0.0000～999999999.9999 |
| sUriOrderGaisanHyoukagaku | 評価金額 | 0～9999999999999999 |
| sUriOrderGaisanHyoukaSoneki | 評価損益 | -999999999999999～9999999999999999 |
| sUriOrderGaisanHyoukaSonekiRitu | 評価損益率(%) | -999999999.99～9999999999.99 |
| sSyuzituOwarine | 前日終値 | 0.0000～999999999.9999 |
| sZenzituHi | 前日比 | -9999999.9999～99999999.9999 |
| sZenzituHiPer | 前日比(%) | -999.99～999.99 |
| sUpDownFlag | 騰落率Flag(%) | 01：+5.01 以上, 06：0 変化なし, 11：-5.01 以下 など |
| sNissyoukinKasikabuZan | 証金貸株残 | 0～9999999999999 |


### 6. 信用建玉一覧

#### 1. 要求
```json
{
"sCLMID":"CLMShinyouTategyokuList",
"sIssueCode":"7201"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMShinyouTategyokuList |
| sIssueCode | 銘柄コード | 指定あり：指定１銘柄のリスト取得（例:"7201"） 指定なし：全保有銘柄のリスト取得（例:""） |

#### 2. 応答
```json
{
"sCLMID":"CLMShinyouTategyokuList",
"sResultCode":"0",
"sResultText":"",
"sWarningCode":"0",
"sWarningText":"",
"sIssueCode":"7201",
"sUritateDaikin":"0",
"sKaitateDaikin":"70000",
"sTotalDaikin":"70000",
"sHyoukaSonekiGoukeiUridate":"0",
"sHyoukaSonekiGoukeiKaidate":"-7783",
"sTokuteiHyoukaSonekiGoukei":"-7783",
"sTotalHyoukaSonekiGoukei":"-7783",
"sIppanHyoukaSonekiGoukei":"0",
"aShinyouTategyokuList":
[
{
"sOrderWarningCode":"0",
"sOrderWarningText":"",
"sOrderTategyokuNumber":"202310160003492",
"sOrderIssueCode":"7201",
"sOrderSizyouC":"00",
"sOrderBaibaiKubun":"3",
"sOrderBensaiKubun":"26",
"sOrderZyoutoekiKazeiC":"1",
"sOrderTategyokuSuryou":"100",
"sOrderTategyokuTanka":"700.0000",
"sOrderHyoukaTanka":"622.2000",
"sOrderGaisanHyoukaSoneki":"-7783",
"sOrderGaisanHyoukaSonekiRitu":"-11.11",
"sTategyokuDaikin":"70000",
"sOrderTateTesuryou":"0",
"sOrderZyunHibu":"3",
"sOrderGyakuhibu":"0",
"sOrderKakikaeryou":"0",
"sOrderKanrihi":"0",
"sOrderKasikaburyou":"0",
"sOrderSonota":"0",
"sOrderTategyokuDay":"20231016",
"sOrderTategyokuKizituDay":"20240415",
"sTategyokuSuryou":"100",
"sOrderYakuzyouHensaiKabusu":"0",
"sOrderGenbikiGenwatasiKabusu":"0",
"sOrderOrderSuryou":"0",
"sOrderHensaiKanouSuryou":"100",
"sSyuzituOwarine":"",
"sZenzituHi":"",
"sZenzituHiPer":"",
"sUpDownFlag":""
}
]
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMShinyouTategyokuList |
| sResultCode | 結果コード | CLMKabuNewOrder.sResultCode 参照 |
| sResultText | 結果テキスト | CLMKabuNewOrder.sResultText 参照 |
| sWarningCode | 警告コード | CLMKabuNewOrder.sWarningCode 参照 |
| sWarningText | 警告テキスト | CLMKabuNewOrder.sWarningText 参照 |
| sIssueCode | 銘柄コード | 要求設定値 |
| sUritateDaikin | 売建代金合計 | 0～9999999999999999 |
| sKaitateDaikin | 買建代金合計 | 0～9999999999999999 |
| sTotalDaikin | 総代金合計 | 0～9999999999999999 |
| sHyoukaSonekiGoukeiUridate | 評価損益合計_売建 | -999999999999999～9999999999999999 |
| sHyoukaSonekiGoukeiKaidate | 評価損益合計_買建 | -999999999999999～9999999999999999 |
| sTotalHyoukaSonekiGoukei | 総評価損益合計 | -999999999999999～9999999999999999 |
| sTokuteiHyoukaSonekiGoukei | 特定口座残高評価損益合計 | -999999999999999～9999999999999999 |
| sIppanHyoukaSonekiGoukei | 一般口座残高評価損益合計 | -999999999999999～9999999999999999 |
| aShinyouTategyokuList | 信用建玉リスト | 以下項目を配列で応答、情報が無い場合は"" |
| sOrderWarningCode | 警告コード | CLMKabuNewOrder.sWarningCode 参照 |
| sOrderWarningText | 警告テキスト | CLMKabuNewOrder.sWarningText 参照 |
| sOrderTategyokuNumber | 建玉番号 | 保有建玉番号 |
| sOrderIssueCode | 銘柄コード | 保有銘柄コード |
| sOrderSizyouC | 市場 | 00：東証 |
| sOrderBaibaiKubun | 売買区分 | CLMKabuNewOrder.sBaibaiKubun 参照 |
| sOrderBensaiKubun | 弁済区分 | 00：なし 26：制度信用6ヶ月 29：制度信用無期限 36：一般信用6ヶ月 39：一般信用無期限 |
| sOrderZyoutoekiKazeiC | 譲渡益課税区分 | 1：特定 3：一般 5：NISA 9：法人 |
| sOrderTategyokuSuryou | 建株数 | 0～9999999999999 |
| sOrderTategyokuTanka | 建単価 | 0.0000～999999999.9999 |
| sOrderHyoukaTanka | 評価単価 | 0.0000～999999999.9999 |
| sOrderGaisanHyoukaSoneki | 評価損益 | -999999999999999～9999999999999999 |
| sOrderGaisanHyoukaSonekiRitu| 評価損益率(%) | -999999999.99～9999999999.99 |
| sTategyokuDaikin | 建玉代金 | 0～9999999999999999 |
| sOrderTateTesuryou | 建手数料 | 0～9999999999999999 |
| sOrderZyunHibu | 順日歩 | 0～9999999999999999 |
| sOrderGyakuhibu | 逆日歩 | 0～9999999999999999 |
| sOrderKakikaeryou | 書換料 | 0～9999999999999999 |
| sOrderKanrihi | 管理費 | 0～9999999999999999 |
| sOrderKasikaburyou | 貸株料 | 0～9999999999999999 |
| sOrderSonota | その他 | 0～9999999999999999 |
| sOrderTategyokuDay | 建日 | YYYYMMDD |
| sOrderTategyokuKizituDay | 建玉期日日 | YYYYMMDD |
| sTategyokuSuryou | 建玉数量 | 0～9999999999999 |
| sOrderYakuzyouHensaiKabusu| 約定返済株数 | 0～9999999999999 |
| sOrderGenbikiGenwatasiKabusu| 現引現渡株数 | 0～9999999999999 |
| sOrderOrderSuryou | 注文中数量 | 0～9999999999999 |
| sOrderHensaiKanouSuryou | 返済可能数量 | 0～9999999999999 |


### 7. 買余力

#### 1. 要求
```json
{
"sCLMID":"CLMZanKaiKanougaku",
"sIssueCode":"",
"sSizyouC":""
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMZanKaiKanougaku |
| sIssueCode | 銘柄コード | 未使用 |
| sSizyouC | 市場 | 未使用 |

#### 2. 応答
```json
{
"sCLMID":"CLMZanKaiKanougaku",
"sResultCode":"0",
"sResultText":"",
"sWarningCode":"0",
"sWarningText":"",
"sIssueCode":"",
"sSizyouC":"",
"sSummaryUpdate":"202312311100",
"sSummaryGenkabuKaituke":"1000000",
"sSummaryNseityouTousiKanougaku":"0",
"sHusokukinHasseiFlg":"0"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMZanKaiKanougaku |
| sResultCode | 結果コード | CLMKabuNewOrder.sResultCode 参照 |
| sResultText | 結果テキスト | CLMKabuNewOrder.sResultText 参照 |
| sSummaryUpdate | 更新日時 | YYYYMMDDHHMM |
| sSummaryGenkabuKaituke | 株式現物買付可能額 | 0～9999999999999999 |
| sSummaryNseityouTousiKanougaku | NISA成長投資可能額 | 0～9999999999999999 |
| sHusokukinHasseiFlg | 不足金発生フラグ | 0：未発生 1：発生 |


### 8. 建余力＆本日維持率

#### 1. 要求
```json
{
"sCLMID":"CLMZanShinkiKanoIjiritu",
"sIssueCode":"",
"sSizyouC":""
}
```

#### 2. 応答
```json
{
"sCLMID":"CLMZanShinkiKanoIjiritu",
"sResultCode":"0",
"sResultText":"",
"sWarningCode":"0",
"sWarningText":"",
"sIssueCode":"",
"sSizyouC":"",
"sSummaryUpdate":"202312311100",
"sSummarySinyouSinkidate":"1000000",
"sItakuhosyoukin":"0.00",
"sOisyouKakuteiFlg":"0"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMZanShinkiKanoIjiritu |
| sSummaryUpdate | 更新日時 | YYYYMMDDHHMM |
| sSummarySinyouSinkidate | 信用新規建可能額 | 0～9999999999999999 |
| sItakuhosyoukin | 委託保証金率(%) | 0.00～9999999999.99 |
| sOisyouKakuteiFlg | 追証フラグ | 0：未確定 1：確定 |


### 9. 売却可能数量

#### 1. 要求
```json
{
"sCLMID":"CLMZanUriKanousuu",
"sIssueCode":"6501"
}
```

#### 2. 応答
```json
{
"sCLMID":"CLMZanUriKanousuu",
"sResultCode":"0",
"sResultText":"",
"sWarningCode":"0",
"sWarningText":"",
"sIssueCode":"6501",
"sSummaryUpdate":"202312311100",
"sZanKabuSuryouUriKanouIppan":"1000000",
"sZanKabuSuryouUriKanouTokutei":"0",
"sZanKabuSuryouUriKanouNisa":"0",
"sZanKabuSuryouUriKanouNseityou":"0"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMZanUriKanousuu |
| sSummaryUpdate | 更新日時 | YYYYMMDDHHMM |
| sZanKabuSuryouUriKanouIppan | 売付可能株数(一般) | 0～9999999999999 |
| sZanKabuSuryouUriKanouTokutei | 売付可能株数(特定) | 0～9999999999999 |
| sZanKabuSuryouUriKanouNisa | 売付可能株数(NISA) | 0～9999999999999 |
| sZanKabuSuryouUriKanouNseityou | 売付可能株数(N成長) | 0～9999999999999 |

### 10. 注文一覧

#### 1. 要求
```json
{
"sCLMID":"CLMOrderList",
"sIssueCode":"8411",
"sSikkouDay":"",
"sOrderSyoukaiStatus":""
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMOrderList |
| sIssueCode | 銘柄コード | 指定あり：指定１銘柄のリスト取得（例:"8411"） 指定なし：全保有銘柄のリスト取得（例:""） |
| sSikkouDay | 注文執行予定日（営業日） | CLMKabuNewOrder.sEigyouDay 参照 指定あり：指定１営業日のリスト取得（例:"20231018"） 指定なし：全保有営業日のリスト取得（例:""） |
| sOrderSyoukaiStatus | 注文照会状態 | ""：指定なし 1：未約定 2：全部約定 3：一部約定 4：訂正取消(可能な注文） 5：未約定+一部約定 指定あり：指定１状態のリスト取得（例:"2"） 指定なし：全保有状態のリスト取得（例:""） |

【注意】
要求項目（sCLMID 以外）は任意指定（ＡＮＤ条件）項目で、指定項目値該当情報をリストとして応答する。
注文執行予定日（営業日）は夕方の日替処理（で翌営業日に変更）以降、その前後（繰越前、繰越後）の情報取得に使用する。過去の注文情報が取得できる訳ではないので注意されたい。

#### 2. 応答
```json
{
"sCLMID":"CLMOrderList",
"sResultCode":"0",
"sResultText":"",
"sWarningCode":"0",
"sWarningText":"",
"sIssueCode":"8411",
"sOrderSyoukaiStatus":"",
"sSikkouDay":"",
"aOrderList":
[
{
"sOrderWarningCode":"0",
"sOrderWarningText":"",
"sOrderOrderNumber":"18000002",
"sOrderIssueCode":"8411",
"sOrderSizyouC":"00",
"sOrderZyoutoekiKazeiC":"1",
"sGenkinSinyouKubun":"0",
"sOrderBensaiKubun":"00",
"sOrderBaibaiKubun":"3",
"sOrderOrderSuryou":"100",
"sOrderCurrentSuryou":"0",
"sOrderOrderPrice":"2300.0000",
"sOrderCondition":"0",
"sOrderOrderPriceKubun":"2",
"sOrderGyakusasiOrderType":"0",
"sOrderGyakusasiZyouken":"0.0000",
"sOrderGyakusasiKubun":" ",
"sOrderGyakusasiPrice":"0.0000",
"sOrderTriggerType":"0",
"sOrderTatebiType":" ",
"sOrderZougen":"",
"sOrderYakuzyouSuryo":"100",
"sOrderYakuzyouPrice":"2300.0000",
"sOrderUtidekiKbn":" ",
"sOrderSikkouDay":"20231018",
"sOrderStatusCode":"10",
"sOrderStatus":"全部約定",
"sOrderYakuzyouStatus":"2",
"sOrderOrderDateTime":"20231018091407",
"sOrderOrderExpireDay":"20231031",
"sOrderKurikosiOrderFlg":"0",
"sOrderCorrectCancelKahiFlg":"1",
"sGaisanDaikin":"230187"
}
]
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMOrderList |
| sResultCode | 結果コード | CLMKabuNewOrder.sResultCode 参照 |
| sResultText | 結果テキスト | CLMKabuNewOrder.sResultText 参照 |
| sWarningCode | 警告コード | CLMKabuNewOrder.sWarningCode 参照 |
| sWarningText | 警告テキスト | CLMKabuNewOrder.sWarningText 参照 |
| sIssueCode | 銘柄コード | 要求設定値 |
| sSikkouDay | 注文執行予定日 | 要求設定値 |
| sOrderSyoukaiStatus | 注文照会状態 | 要求設定値 |
| aOrderList | 注文リスト | 以下項目を配列で応答、情報が無い場合は"" |
| sOrderWarningCode | 警告コード | CLMKabuNewOrder.sWarningCode 参照 |
| sOrderWarningText | 警告テキスト | CLMKabuNewOrder.sWarningText 参照 |
| sOrderOrderNumber | 注文番号 | CLMKabuNewOrder.sOrderNumber 参照 |
| sOrderIssueCode | 銘柄コード | CLMKabuNewOrder.sIssueCode 参照 |
| sOrderSizyouC | 市場 | CLMKabuNewOrder.SizyouC 参照 |
| sOrderZyoutoekiKazeiC | 譲渡益課税区分 | CLMKabuNewOrder.sZyoutoekiKazeiC 参照 |
| sGenkinSinyouKubun | 現金信用区分 | CLMKabuNewOrder.sGenkinShinyouKubun 参照 |
| sOrderBensaiKubun | 弁済区分 | 00：なし 26：制度信用6ヶ月 29：制度信用無期限 36：一般信用6ヶ月 39：一般信用無期限 |
| sOrderBaibaiKubun | 売買区分 | CLMKabuNewOrder.sBaibaiKubun 参照 |
| sOrderOrderSuryou | 注文株数 | 0～9999999999999 |
| sOrderCurrentSuryou | 有効株数 | Ｎ≦CLMKabuNewOrder.sOrderSuryou |
| sOrderOrderPrice | 注文単価 | 0.0000～999999999.9999 |
| sOrderCondition | 執行条件 | CLMKabuNewOrder.sCondition 参照 |
| sOrderOrderPriceKubun | 注文値段区分 | " "：未使用 1：成行 2：指値 3：親注文より高い 4：親注文より低い |
| sOrderGyakusasiOrderType| 逆指値注文種別 | CLMKabuNewOrder.sGyakusasiOrderType 参照 |
| sOrderGyakusasiZyouken | 逆指値条件 | 0.0000～999999999.9999 |
| sOrderGyakusasiKubun | 逆指値値段区分 | " "：未使用 0：成行 1：指値 |
| sOrderGyakusasiPrice | 逆指値値段 | 0.0000～999999999.9999 |
| sOrderTriggerType | トリガータイプ | 0：未トリガー（初期値） トリガー発火後は以下に遷移。 1：自動 2：手動発注 3：手動失効 |
| sOrderTatebiType | 建日種類 | 信用返済時に指定する返済建玉順序種類指定 " "：指定なし 1：個別指定 2：建日順 3：単価益順 4：単価損順 |
| sOrderZougen | リバース増減値 | 未使用 |
| sOrderYakuzyouSuryo | 成立株数 | 0～9999999999999 |
| sOrderYakuzyouPrice | 成立単価 | 0.0000～999999999.9999 |
| sOrderUtidekiKbn | 内出来区分 | " "：約定分割以外 2：約定分割 |
| sOrderSikkouDay | 執行日 | YYYYMMDD |
| sOrderStatusCode | 状態コード | [通常注文]の状態 0：受付未済 1：未約定 2：受付エラー 3：訂正中 4：訂正完了 5：訂正失敗 6：取消中 7：取消完了 8：取消失敗 9：一部約定 10：全部約定 11：一部失効 12：全部失効 13：発注待ち 14：無効 15：切替注文 16：切替完了 17：切替注文失敗 19：繰越失効 20：一部障害処理 21：障害処理 <br> [逆指値注文]、[通常+逆指値注文]の状態 15：逆指注文(切替中) 16：逆指注文(未約定) 17：逆指注文(失敗) 50：発注中 |
| sOrderStatus | 状態名称 | 状態コードの名称 |
| sOrderYakuzyouStatus | 約定ステータス | 0：未約定 1：一部約定 2：全部約定 3：約定中 |
| sOrderOrderDateTime | 注文日付 | YYYYMMDDHHMMSS 00000000000000 |
| sOrderOrderExpireDay | 有効期限 | YYYYMMDD 00000000 |
| sOrderKurikosiOrderFlg| 繰越注文フラグ | 0：当日注文 1：繰越注文 2：無効 |
| sOrderCorrectCancelKahiFlg| 訂正取消可否フラグ | 0：可(取消、訂正) 1：否 2：一部可(取消のみ) |
| sGaisanDaikin | 概算代金 | -999999999999999～9999999999999999 |


### 11. 注文約定一覧（詳細）

#### 1. 要求
```json
{
"sCLMID":"CLMOrderListDetail",
"sOrderNumber":"18000002",
"sEigyouDay":"20231018"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMOrderListDetail |
| sOrderNumber | 注文番号 | CLMKabuNewOrder.sOrderNumber 参照 |
| sEigyouDay | 営業日 | CLMKabuNewOrder.sEigyouDay 参照 |
【注意】 要求項目は全て必須指定です。

#### 2. 応答
```json
{
"sCLMID":"CLMOrderListDetail",
"sResultCode":"0",
"sResultText":"",
"sWarningCode":"0",
"sWarningText":"",
"sOrderNumber":"18000004",
"sEigyouDay":"20231018",
"sIssueCode":"8411",
"sOrderSizyouC":"00",
"sOrderBaibaiKubun":"1",
"sGenkinSinyouKubun":"4",
"sOrderBensaiKubun":"26",
"sOrderCondition":"0",
"sOrderOrderPriceKubun":"1",
"sOrderOrderPrice":"0.0000",
"sOrderOrderSuryou":"100",
"sOrderCurrentSuryou":"0",
"sOrderStatusCode":"10",
"sOrderStatus":"全部約定",
"sOrderOrderDateTime":"20231018104821",
"sOrderOrderExpireDay":"00000000",
"sChannel":"1",
"sGenbutuZyoutoekiKazeiC":"1",
"sSinyouZyoutoekiKazeiC":"1",
"sGyakusasiOrderType":"0",
"sGyakusasiZyouken":"0.0000",
"sGyakusasiKubun":" ",
"sGyakusasiPrice":"0.0000",
"sTriggerType":"0",
"sTriggerTime":"00000000000000",
"sUkewatasiDay":"20231020",
"sYakuzyouPrice":"2300.0000",
"sYakuzyouSuryou":"100",
"sBaiBaiDaikin":"230000",
"sUtidekiKubun":" ",
"sGaisanDaikin":"-11",
"sBaiBaiTesuryo":"0",
"sShouhizei":"0",
"sTatebiType":"1",
"sSizyouErrorCode":"",
"sZougen":"",
"sOrderAcceptTime":"20231018104942",
"sOrderExpireDayLimit":"20231031",
"aYakuzyouSikkouList":
[
{
"sYakuzyouWarningCode":"0",
"sYakuzyouWarningText":"",
"sYakuzyouSuryou":"100",
"sYakuzyouPrice":"2300.0000",
"sYakuzyouDate":"20231018104942"
}
],
"aKessaiOrderTategyokuList":
[
{
"sKessaiWarningCode":"0",
"sKessaiWarningText":"",
"sKessaiTatebiZyuni":"1",
"sKessaiTategyokuDay":"20231018",
"sKessaiTategyokuPrice":"2300.0000",
"sKessaiOrderSuryo":"100",
"sKessaiYakuzyouSuryo":"100",
"sKessaiYakuzyouPrice":"2300.0000",
"sKessaiTateTesuryou":"0",
"sKessaiZyunHibu":"11",
"sKessaiGyakuhibu":"0",
"sKessaiKakikaeryou":"0",
"sKessaiKanrihi":"0",
"sKessaiKasikaburyou":"0",
"sKessaiSonota":"0",
"sKessaiSoneki":"-11"
}
]
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMOrderListDetail |
| sResultCode | 結果コード | CLMKabuNewOrder.sResultCode 参照 |
| sResultText | 結果テキスト | CLMKabuNewOrder.sResultText 参照 |
| sWarningCode | 警告コード | CLMKabuNewOrder.sWarningCode 参照 |
| sWarningText | 警告テキスト | CLMKabuNewOrder.sWarningText 参照 |
| sOrderNumber | 注文番号 | CLMKabuNewOrder.sOrderNumber 参照 |
| sEigyouDay | 営業日 | CLMKabuNewOrder.sEigyouDay 参照 |
| sIssueCode | 銘柄コード | CLMKabuNewOrder.sIssueCode 参照 |
| sOrderSizyouC | 市場 | CLMKabuNewOrder.sSizyouC 参照 |
| sOrderBaibaiKubun | 売買区分 | CLMKabuNewOrder.sBaibaiKubun 参照 |
| sGenkinSinyouKubun | 現金信用区分 | CLMKabuNewOrder.sGenkinShinyouKubun 参照 |
| sOrderBensaiKubun | 弁済区分 | CLMOrderList.sOrderBensaiKubun 参照 |
| sOrderCondition | 執行条件 | CLMKabuNewOrder.sCondition 参照 |
| sOrderOrderPriceKubun | 注文値段区分 | CLMOrderList.sOrderOrderPriceKubun 参照 |
| sOrderOrderPrice | 注文単価 | CLMOrderList.sOrderOrderPrice 参照 |
| sOrderOrderSuryou | 注文株数 | CLMOrderList.sOrderOrderSuryou 参照 |
| sOrderCurrentSuryou | 有効株数 | CLMOrderList.sOrderCurrentSuryou 参照 |
| sOrderStatusCode | 状態コード | CLMOrderList.sOrderStatusCode 参照 |
| sOrderStatus | 状態名称 | CLMOrderList.sOrderStatus 参照 |
| sOrderOrderDateTime | 注文日付 | CLMOrderList.sOrderOrderDateTime 参照 |
| sOrderOrderExpireDay | 有効期限 | CLMOrderList.sOrderOrderExpireDay 参照 |
| sChannel | チャネル | 1：標準Ｗｅｂ（PC） 2：コールセンター（CC2）など（詳細はマニュアル参照） F：ｅ支店・ＡＰＩ（API） |
| sGenbutuZyoutoekiKazeiC | 現物口座区分 | CLMOrderList.sOrderZyoutoekiKazeiC 参照 |
| sSinyouZyoutoekiKazeiC | 建玉口座区分 | 1：特定 3：一般 |
| sGyakusasiOrderType | 逆指値注文種別 | 0：通常 1：逆指値 2：通常＋逆指値 |
| sGyakusasiZyouken | 逆指値条件 | 0.0000～999999999.9999 |
| sGyakusasiKubun | 逆指値値段区分 | CLMOrderList.sOrderGyakusasiKubun 参照 |
| sGyakusasiPrice | 逆指値値段 | 0.0000～999999999.9999 |
| sTriggerType | トリガータイプ | CLMOrderList.sOrderTriggerType 参照 |
| sTriggerTime | トリガー日時 | YYYYMMDDHHMMSS または 00000000000000 |
| sUkewatasiDay | 受渡日 | YYYYMMDD または 00000000 |
| sYakuzyouPrice | 約定単価 | 0.0000～999999999.9999 |
| sYakuzyouSuryou | 約定株数 | 0～9999999999999 |
| sBaiBaiDaikin | 売買代金 | 0～9999999999999999 |
| sUtidekiKubun | 内出来区分 | CLMOrderList.sOrderUtidekiKbn 参照 |
| sGaisanDaikin | 概算代金 | 0～9999999999999999 |
| sBaiBaiTesuryo | 手数料 | 0～9999999999999999 |
| sShouhizei | 消費税 | 0～9999999999999999 |
| sTatebiType | 建日種類 | CLMOrderList.sOrderTatebiType 参照 |
| sSizyouErrorCode | 取引所エラー等理由コード | ""：正常 上記以外は マスタ情報ダウンロード 参照 |
| sZougen | リバース増減値 | 未使用 |
| sOrderAcceptTime | 取引所受付／エラー時刻 | YYYYMMDDHHMMSS または 00000000000000 |
| sOrderExpireDayLimit | 注文失効日付 | YYYYMMDD |
| aYakuzyouSikkouList | 約定失効リスト | 以下項目を配列で応答、情報が無い場合は"" |
| sYakuzyouWarningCode | 警告コード | CLMKabuNewOrder.sWarningCode 参照 |
| sYakuzyouWarningText | 警告テキスト | CLMKabuNewOrder.sWarningText 参照 |
| sYakuzyouSuryou | 約定数量 | 0～9999999999999 |
| sYakuzyouPrice | 約定価格 | 0.0000～999999999.9999 |
| sYakuzyouDate | 約定日時 | YYYYMMDDHHMMSS または 00000000000000 |
| aKessaiOrderTategyokuList | 決済注文建株指定リスト | 以下項目を配列で応答、情報が無い場合は"" |
| sKessaiWarningCode | 警告コード | CLMKabuNewOrder.sWarningCode 参照 |
| sKessaiSonota | その他 | 0～9999999999999999 （他明細略） |
| sKessaiSoneki | 決済損益/受渡代金 | -999999999999999～9999999999999999 |


### 12. 可能額サマリー

#### 1. 要求
```json
{
"sCLMID":"CLMZanKaiSummary"
}
```

#### 2. 応答
```json
{
"sCLMID":"CLMZanKaiSummary",
"sResultCode":"0",
"sResultText":"",
"sWarningCode":"0",
"sWarningText":"",
"sUpdateDate":"202310200849",
"sOisyouHasseiFlg":"0",
"sOhzsKeisanDay":"",
"sOhzsGenkinHosyoukin":"",
"sOhzsDaiyouHyoukagaku":"",
"sOhzsSasiireHosyoukin":"",
"sOhzsHyoukaSoneki":"",
"sOhzsSyokeihi":"",
"sOhzsMiukeKessaiSon":"",
"sOhzsMiukeKessaiEki":"",
"sOhzsUkeireHosyoukin":"",
"sOhzsTatekabuDaikin":"",
"sOhzsItakuHosyoukinRitu":"",
"sTatekaekinHasseiFlg":"0",
"sThzNyukinKigenDay":"",
"sThzSeisangaku":"",
"sThzHibakariKousokukin":"",
"sThzHurikaegaku":"",
"sThzHituyouNyukingaku":"",
"sThzKakuteiFlg":"",
"sGenbutuKabuKaituke":"1144578",
"sSinyouSinkidate":"26291254",
"sSinyouGenbiki ":"1144578",
"sHosyouKinritu":"12427.44",
"sNseityouTousiKanougaku":"",
"sTousinKaituke":"1144578",
"sRuitouKaituke":"0",
"sIPOKounyu":"1144578",
"sSyukkin":"1144578",
"sFusokugaku":"0",
"sLargeKaidateYoryoku":"0",
"sMiniKaidateYoryoku ":"0",
"sLargeUridateYoryoku":"0",
"sMiniUridateYoryoku":"0",
"sOpKaidateYoryokyu":"0",
"sSyoukokinFusokugaku":"0",
"sGenbutuBaibaiDaikin":"0",
"sGenbutuOrderCount":"0",
"sGenbutuYakuzyouCount":"0",
"sSinyouBaibaiDaikin":"0",
"sSinyouOrderCount":"0",
"sSinyouYakuzyouCount":"0",
"sSakiBaibaiDaikin":"0",
"sSakiOrderCount":"0",
"sSakiYakuzyouCount":"0",
"sOpBaibaiDaikin":"0",
"sOpOrderCount":"0",
"sOpYakuzyouCount":"0",
"aHikazeiKouzaList":
[
{
"sHikazeiTekiyouYear":"2023",
"sSeityouTousiKanougaku":"300000"
}
],
"aOisyouHasseiZyoukyouList":"",
"aHosyoukinSeikyuZyoukyouList":""
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMZanKaiSummary |
| sResultCode | 結果コード | CLMKabuNewOrder.sResultCode 参照 |
| sResultText | 結果テキスト | CLMKabuNewOrder.sResultText 参照 |
| sUpdateDate | 更新日時 | YYYYMMDDHHMM |
| sOisyouHasseiFlg | 追証発生フラグ | 1:発生 0:未発生 |
| sTatekaekinHasseiFlg | 立替金発生フラグ | 1:発生 0:未発生 |
| sGenbutuKabuKaituke | 株式現物買付可能額 | 0～9999999999999999 |
| sSinyouSinkidate | 信用新規建可能額 | 0～9999999999999999 |
| sSinyouGenbiki | 信用現引可能額 | 0～9999999999999999 |
| sHosyouKinritu | 委託保証金率(%) | 0～9999999999.99 |
| sNseityouTousiKanougaku | NISA成長投資可能額 | 0～9999999999999999 |
| sTousinKaituke | 投信買付可能額 | 0～9999999999999999 |
| sRuitouKaituke | MMF・中国F買付 | 0～9999999999999999 |
| sIPOKounyu | IPO購入可能額 | 0～9999999999999999 |
| sSyukkin | 出金可能額 | 0～9999999999999999 |
| sFusokugaku | 不足額(入金請求額） | 0～9999999999999999 |
| sLargeKaidateYoryoku | 先物買建 | 0～9999999999999 |
| sMiniKaidateYoryoku | OPプット売建(ミニ) | 0～9999999999999 |
| sLargeUridateYoryoku | 先物売建 | 0～9999999999999 |
| sMiniUridateYoryoku | OPコール売建(ミニ) | 0～9999999999999 |
| sOpKaidateYoryokyu | オプション新規買建 | 0～9999999999999999 |
| sSyoukokinFusokugaku | 証拠金不足額（本日請求額） | 0～9999999999999999 |
| aHikazeiKouzaList | 非課税口座リスト | 以下レコードを配列で設定 |
| sHikazeiTekiyouYear | 適用年（対象年） | YYYY、非課税適用年度 |
| sSeityouTousiKanougaku| 成長投資可能額 | 0～999999999999999 |
| aOisyouHasseiZyoukyouList| 追証発生状況リスト | 以下項目を配列で応答、情報が無い場合は"" |
| aHosyoukinSeikyuZyoukyouList| 保証金請求発生状況リスト | 以下項目を配列で応答、情報が無い場合は"" |


### 13. 可能額推移

#### 1. 要求
```json
{
"sCLMID":"CLMZanKaiKanougakuSuii"
}
```

#### 2. 応答
```json
{
"sCLMID":"CLMZanKaiKanougakuSuii",
"sResultCode":"0",
"sResultText":"",
"sWarningCode":"0",
"sWarningText":"",
"sUpdateDate":"202310201242",
"sNearaiKubun":"0",
"aKanougakuSuiiList":
[
{
"sHituke":"20231020",
"sAzukariKin":"1144578",
"sHattyuZyutoukin":"0",
"sHibakariKousokukin":"0",
"sSonotaKousokukin":"0",
"sGenkinHosyoukin":"1144578",
"sDaiyouHyoukagaku":"202310200849",
"sSasiireHosyoukin":"8707818",
"sSinyouTateHyoukaSon":"8580",
"sSinyouTateHyoukaEki":"0",
"sSinyouTadeSyoukeihi":"24",
"sMiukewatasiKessaiSon":"0",
"sMiukewatasiKessaiEki":"0",
"sUkeireHosyoukin":"8699214",
"sMikessaiTateDaikin":"70000",
"sGenbikiWatasiTateDaikin":"0",
"sHituyouHosyoukin":"23100",
"sHituyouGenkinHosyoukin":"0",
"sHosyoukinYoryoku":"8676114",
"sGenkinHosyoukinYoryoku":"1144578",
"sItakuHosyoukinRitu":"12427.44",
"sHosyoukinHikidasiKousokukin":"308604",
"sHosyoukinHikidasiYoryoku":"8399214",
"sOisyouHituyouHosyoukin":"308604",
"sOisyouYoryoku":"8399214",
"sFusokugaku":"0",
"sGenbutuKaitukeKanougaku":"1144578",
"sSinyouSinkidateKanougaku":"26291254",
"sGenbikiKanougaku":"1144578",
"sTousinKaitukeKanougaku":"1144578",
"sSyukkinKanougaku":"1144578"
}
]
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMZanKaiKanougakuSuii |
| sResultCode | 結果コード | CLMKabuNewOrder.sResultCode 参照 |
| sResultText | 結果テキスト | CLMKabuNewOrder.sResultText 参照 |
| sUpdateDate | 更新日時 | YYYYMMDDHHMM |
| sNearaiKubun | 値洗い区分 | 0:値洗い停止 1:値洗い中 2:値洗い終了 |
| aKanougakuSuiiList | 可能額推移リスト | 配列で応答。:当日営業日 ～:６営業日 |
| sHituke | 日付 | YYYYMMDD |
| sAzukariKin | 預り金 | 0～9999999999999999 |
| sHattyuZyutoukin | 発注済み注文充当金 | 0～9999999999999999 |
| sHibakariKousokukin | 日計り拘束金 | 0～9999999999999999 |
| sSonotaKousokukin | その他拘束金 | 0～9999999999999999 |
| sGenkinHosyoukin | 現金保証金 | -999999999999999～9999999999999999 |
| sDaiyouHyoukagaku | 代用証券評価額 | 0～9999999999999999 |
| sSasiireHosyoukin | 差入保証金 | 0～9999999999999999 |
| sUkeireHosyoukin | 受入保証金 | -999999999999999～9999999999999999 |
| sItakuHosyoukinRitu | 委託保証金率(%) | -999999999.99～9999999999.99 |
| sFusokugaku | 追証/立替金/保証金不足額 | 0～9999999999999999 |
| sGenbutuKaitukeKanougaku| 現物株式買付可能額 | -999999999999999～9999999999999999 |
| sSinyouSinkidateKanougaku| 信用新規建可能額 | -999999999999999～9999999999999999 |
| sGenbikiKanougaku | 信用現引可能額 | -999999999999999～9999999999999999 |
| sTousinKaitukeKanougaku| 投信買付可能額 | -999999999999999～9999999999999999 |
| sSyukkinKanougaku | 出金可能額 | -999999999999999～9999999999999999 |


### 14. 現物株式買付可能額詳細

#### 1. 要求
```json
{
"sCLMID":"CLMZanKaiGenbutuKaitukeSyousai",
"sHitukeIndex":"3"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMZanKaiGenbutuKaitukeSyousai |
| sHitukeIndex | 日付インデックス | 3:第4営業日 4:第5営業日 5:第6営業日 |

#### 2. 応答
```json
{
"sCLMID":"CLMZanKaiGenbutuKaitukeSyousai",
"sResultCode":"0",
"sResultText":"",
"sWarningCode":"0",
"sWarningText":"",
"sHitukeIndex":"3",
"sHituke":"20231030",
"sGenkinHosyoukin":"1383069",
"sHosyoukinGenbutuKaitukeKanouga":"1383069",
"sGenbutuKaitukeKanougaku":"1383069",
"sAzukariKin":"1383069",
"sHattyuZyutoukin":"0",
"sHibakariKousokukin":"0",
"sSonotaKousokukin":"0",
"sHituyouGenkinHosyoukin":"0",
"sDaiyouHyoukagaku":"7257880",
"sTatekabuHyoukaSoneki":"-8940",
"sTatekabuSyoukeihi":"232",
"sMiukewatasiKessaiSon":"0",
"sMiukewatasiKessaiEki":"0",
"sUkeireHosyoukin":"8631777",
"sHituyouHosyoukin":"244658",
"sHosyoukinYoryoku":"8387119"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMZanKaiGenbutuKaitukeSyousai |
| sHitukeIndex | 日付インデックス | 要求設定値 |
| sHituke | 指定日（日付） | YYYYMMDD |
| sGenkinHosyoukin | 現金保証金 | -999999999999999～9999999999999999 |
| sHosyoukinGenbutuKaitukeKanouga | 保証金からの現物株式買付可能額 | 0～9999999999999999 |
| sGenbutuKaitukeKanougaku| 現物株式買付可能額 | -999999999999999～9999999999999999 |
| sUkeireHosyoukin | 受入保証金 | -999999999999999～9999999999999999 |
| sHituyouHosyoukin | 必要保証金 | -999999999999999～9999999999999999 |
| sHosyoukinYoryoku | 保証金余力 | -999999999999999～9999999999999999 |


### 15. 信用新規建て可能額詳細

#### 1. 要求
```json
{
"sCLMID":"CLMZanKaiSinyouSinkidateSyousai",
"sHitukeIndex":"3"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMZanKaiSinyouSinkidateSyousai |
| sHitukeIndex | 日付インデックス | 0:第1営業日 1:第2営業日 2:第3営業日 3:第4営業日 4:第5営業日 5:第6営業日 |

#### 2. 応答
```json
{
"sCLMID":"CLMZanKaiSinyouSinkidateSyousai",
"sResultCode":"0",
"sResultText":"",
"sWarningCode":"0",
"sWarningText":"",
"sHitukeIndex":"3",
"sHituke":"20231030",
"sUkeireHosyoukin":"8631777",
"sHituyouHosyoukin":"244658",
"sHosyoukinYoryoku":"8387119",
"sHosyoukinTyousyuRitu":"33",
"sSinyouSinkidateKanougaku":"25415512",
"sAzukariKin":"1383069",
"sHattyuZyutoukin":"0",
"sSonotaKousokukin":"0",
"sGenkinHosyoukin":"1383069",
"sDaiyouHyoukagaku":"7257880",
"sHattyuDaiyouHyoukagaku":"0",
"sSasiireHosyoukin":"8640949",
"sSinkiTesuryou":"0",
"sHibuGyakuhibuKousokuki":"232",
"sHibuGyakuhibuSyueki":"0",
"sSonotaTateSyokeihi":"0",
"sSinyouTadeSyoukeihi":"232",
"sSinyouTateHyoukaSon":"8940",
"sSinyouTateHyoukaEki":"0",
"sTatekabuHyoukaSoneki":"-8940",
"sMiukewatasiKessaiSon":"0",
"sMiukewatasiKessaiEki":"0",
"sSaiteiHituyouHosyoukin":"300000",
"sHosyoukin":"244658",
"sHattyuHosyoukin":"0",
"sGenbikiWatasiHosyoukin":"0",
"sMikessaiGenkinHosyoukin":"0",
"sHattyuGenkinHosyoukin":"0",
"sGenbwGenkinHosyoukin":"0",
"sHituyouGenkinHosyoukin":"0",
"sHosyoukinRitu":"33",
"sHosyoukinIziRitu":"30",
"sHosyoukinRituIziYoryoku":"8387119",
"sHosyoukinIzirituIziYoryoku":"8409360",
"sMikessaiTateDaikin":"741390",
"sHattyuTateDaikin":"0",
"sGenbikiWatasiTateDaikin":"0",
"sItakuHosyoukinRitu":"1164.26",
"sTouzituKessaiSon":"0",
"sTouzituKessaiEki":"0",
"sKessaiTotalToday":"0",
"sTouzituKessaiZenHyouka":"0",
"sUkewatasiTategyokuSon":"0",
"sUkewatasiTategyokuEki":"0",
"sKessaiTotalSiteibi":"0"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMZanKaiSinyouSinkidateSyousai |
| sHitukeIndex | 日付インデックス | 要求設定値 |
| sHituke | 指定日（日付） | YYYYMMDD |
| sUkeireHosyoukin | 受入保証金 | -999999999999999～9999999999999999 |
| sHituyouHosyoukin | 必要保証金 | -999999999999999～9999999999999999 |
| sHosyoukinYoryoku | 保証金余力 | -999999999999999～9999999999999999 |
| sHosyoukinTyousyuRitu | 保証金徴収率(%) | 0～9999999999999 |
| sSinyouSinkidateKanougaku| 信用新規建可能額 | -999999999999999～9999999999999999 |
| sItakuHosyoukinRitu | 委託保証金率(%) | 0～9999999999.99 |


### 16. リアル保証金率

#### 1. 要求
```json
{
"sCLMID":"CLMZanRealHosyoukinRitu"
}
```

#### 2. 応答
```json
{
"sCLMID":"CLMZanRealHosyoukinRitu",
"sResultCode":"0",
"sResultText":"",
"sWarningCode":"0",
"sWarningText":"",
"sSasiireHosyoukin":"8640949",
"sHyoukaSonEki":"-9172",
"sUkeireHosyoukin":"8631777",
"sTateKabuDaikin":"741390",
"sItakuHosyoukinRitu":"1164.26",
"sOisyouHituyouHosyoukin":"309172",
"sOisyouYoryoku":"8331777",
"sT0SasiireHosyoukin":"8640949",
"sT0HyoukaSonEki":"-9172",
"sT0UkeireHosyoukin":"8631777",
"sT0TateKabuDaikin":"741390",
"sT0ItakuHosyoukinRitu":"1164.26",
"sT0OisyouHituyouHosyoukin":"309172",
"sT0OisyouYoryoku":"8331777",
"sT5SasiireHosyoukin":"8640949",
"sT5HyoukaSonEki":"-9172",
"sT5UkeireHosyoukin":"8631777",
"sT5TateKabuDaikin":"741390",
"sT5ItakuHosyoukinRitu":"1164.26",
"sT5OisyouHituyouHosyoukin":"309172",
"sT5OisyouYoryoku":"8331777"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMZanRealHosyoukinRitu |
| sResultCode | 結果コード | CLMKabuNewOrder.sResultCode 参照 |
| sResultText | 結果テキスト | CLMKabuNewOrder.sResultText 参照 |
| sSasiireHosyoukin | 差入保証金 | 0～9999999999999999 |
| sHyoukaSonEki | 評価損益 | 0～9999999999999999 |
| sUkeireHosyoukin | 受入保証金 | 0～9999999999999999 |
| sTateKabuDaikin | 建株代金 | -999999999999999～9999999999999999 |
| sItakuHosyoukinRitu | 委託保証金率(%) | 0～9999999999999 |
| sOisyouHituyouHosyoukin| 追証必要保証金 | 0～9999999999999999 |
| sOisyouYoryoku | 追証余力 | 0～9999999999999999 |
| sT0SasiireHosyoukin | 差入保証金 (T0) | 0～9999999999999999 |
| sT0ItakuHosyoukinRitu | 委託保証金率(%) (T0) | 0～9999999999999 |
| sT5SasiireHosyoukin | 差入保証金 (T5) | 0～9999999999999999 |
| sT5ItakuHosyoukinRitu | 委託保証金率(%) (T5) | 0～9999999999999 |

## 4. マスタ機能（REQUEST I/F）

### 1. マスタ情報ダウンロード

#### 1. 要求
```json
{
"sCLMID":"CLMEventDownload",
"sTargetCLMID":"CLMIssueMstKabu,CLMDateZyouhou"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMEventDownload |
| sTargetCLMID | 対象機能ＩＤ | 取得したいマスタ情報の機能ＩＤをカンマ区切りで羅列する。未指定「""」時は全マスタ情報。例："CLMIssueMstKabu,CLMDateZyouhou" |

※本要求は配信要求のため応答は返さず、指定されたマスタ情報のレコード（JSON）が配信されます。最後に `CLMEventDownloadComplete` が配信されて終了します。

#### 配信される各マスタデータ（抜粋）

**・システムステータス (CLMSystemStatus)**
```json
{
"sCLMID":"CLMSystemStatus",
"sSystemStatusKey":"001",
"sLoginKyokaKubun":"1",
"sSystemStatus":"1",
"sCreateTime":"",
"sUpdateTime":"",
"sUpdateNumber":"",
"sDeleteFlag":"",
"sDeleteTime":""
}
```

**・日付情報 (CLMDateZyouhou)**
```json
{
"sCLMID":"CLMDateZyouhou",
"sDayKey":"001",
"sMaeEigyouDay_1":"20231031",
"sTheDay":"20231101",
"sYokuEigyouDay_1":"20231102",
"sKabuUkewatasiDay":"20231106",
"sKabuKariUkewatasiDay":"20231107",
"sBondUkewatasiDay":"20231106"
}
```

**・株式銘柄マスタ (CLMIssueMstKabu)**
```json
{
"sCLMID":"CLMIssueMstKabu",
"sIssueCode":"1301",
"sIssueName":"極 洋",
"sTokuteiF":"1",
"sHikazeiC":"1",
"sZyouzyouHakkouKabusu":"10928283",
"sBaibaiTani":"100",
"sGyousyuCode":"0050",
"sGyousyuName":"水産・農林業"
}
```
（※その他、呼値、運用ステータス、株式銘柄市場マスタ、株式銘柄別・市場別規制、先物・オプション銘柄マスタ、代用掛目、保証金マスタ等のマスタ情報が個別の機能IDで配信されます。）


### 2. マスタ情報問合取得

#### 1. 要求
```json
{
"sCLMID":"CLMMfdsGetMasterData",
"sTargetCLMID":"CLMIssueMstKabu,CLMOrderErrReason",
"sTargetColumn":"sIssueCode,sIssueName,sErrReasonCode,sErrReasonText"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMMfdsGetMasterData |
| sTargetCLMID | 対象機能ＩＤ | 取得したいマスタ情報の機能ＩＤをカンマ区切りで羅列する。未指定時は全マスタ。 |
| sTargetColumn| 対象項目 | 各マスタ情報で取得したい項目をカンマ区切りで指定する。未指定時は全項目。 |

#### 2. 応答
```json
{
"sCLMID":"CLMMfdsGetMasterData",
"CLMIssueMstKabu":
[
{
"sIssueCode":"oxox",
"sIssueName":"oxox"
}
],
"CLMOrderErrReason":
[
{
"sErrReasonCode":"oxox",
"sErrReasonText":"oxox"
}
]
}
```


### 3. ニュースヘッダー問合取得

#### 1. 要求
```json
{
"sCLMID":"CLMMfdsGetNewsHead",
"p_CG":"",
"p_IS":"",
"p_DT_FROM":"",
"p_DT_TO":"",
"p_REC_OFST":"",
"p_REC_LIMT":""
}
```

#### 2. 応答
```json
{
"sCLMID":"CLMMfdsGetNewsHead",
"p_REC_MAX":"26359",
"aCLMMfdsNewsHead":
[
{
"p_ID":"20230512153000_OBM9789",
"p_DT":"20230512",
"p_TM":"1530",
"p_CGL":"120",
"p_GNL":"62104",
"p_ISL":"4838",
"p_HDL":"JTNDVERuZXQlM0VB...."
}
]
}
```


### 4. ニュースボディー問合取得

#### 1. 要求
```json
{
"sCLMID":"CLMMfdsGetNewsBody",
"p_ID":"20230315121900_NYU8165"
}
```

#### 2. 応答
```json
{
"sCLMID":"CLMMfdsGetNewsBody",
"aCLMMfdsNewsBody":
[
{
"p_ID":"20230315121900_NYU8165",
"p_DT":"20230315",
"p_TM":"1219",
"p_CGL":"110",
"p_GNL":"60010",
"p_ISL":"8070|3657|6822...",
"p_HDL":"JTNDVERuZXQlM0VB....",
"p_TX":"JTNDVERuZXQlM0VB...."
}
]
}
```


### 5. 銘柄詳細情報問合取得

#### 1. 要求
```json
{
"sCLMID":"CLMMfdsGetIssueDetail",
"sTargetIssueCode":"6501,7203"
}
```

#### 2. 応答
```json
{
"sCLMID":"CLMMfdsGetIssueDetail",
"aCLMMfdsIssueDetail":
[
{
"sIssueCode":"6501",
"pBPSB":"6155.38",
"pCLOE":"2025/03/28",
"pEPSF":"131.06",
"pEXRD":"2024/06/27",
"pIDVE":"2025/09/29",
"pROEL":"10.52",
"pRPER":"30.1",
"pSPBR":"4.99",
"pSPRO":"1.54",
"pSYIE":"1.03",
"pYHPD":"2024/12/05",
"pYHPR":"4145",
"pYLPD":"2024/08/05",
"pYLPR":"2584.0"
}
]
}
```


### 6. 証金残情報問合取得

#### 1. 要求
```json
{
"sCLMID":"CLMMfdsGetSyoukinZan",
"sTargetIssueCode":"6501,7203"
}
```

#### 2. 応答
```json
{
"sCLMID":"CLMMfdsGetSyoukinZan",
"aCLMMfdsSyoukinZan":
[
{
"sIssueCode":"6501",
"pSFC6":"24500",
"pSFD":"2024/12/30",
"pSFD6":"10.3",
"pSFF6":"239600",
"pSFG6":"21300",
"pSFKS":"2",
"pSFL6":"21900",
"pSFN6":"235900",
"pSFP6":"600",
"pSFR6":"64.75",
"pSFS6":"3700",
"pSSG6":"-3200",
"pSSL6":"0",
"pSSP6":"3200"
}
]
}
```


### 7. 信用残情報問合取得

#### 1. 要求
```json
{
"sCLMID":"CLMMfdsGetShinyouZan",
"sTargetIssueCode":"6501,7203"
}
```

#### 2. 応答
```json
{
"sCLMID":"CLMMfdsGetShinyouZan",
"aCLMMfdsShinyouZan":
[
{
"sIssueCode":"6501",
"pMBB3":"1632200",
"pMBB6":"3691900",
"pMBBQ":"5324100",
"pMBC3":"-103700",
"pMBC6":"-215900",
"pMBCQ":"-319600",
"pMBD":"2024/12/20",
"pMBN3":"124600",
"pMBN6":"445800",
"pMBNQ":"570400",
"pMBR3":"5.99",
"pMBR6":"9.26",
"pMBRQ":"7.93",
"pMBS3":"272400",
"pMBS6":"398700",
"pMBSQ":"671100"
}
]
}
```


### 8. 逆日歩情報問合取得

#### 1. 要求
```json
{
"sCLMID":"CLMMfdsGetHibuInfo",
"sTargetIssueCode":"6501,7203"
}
```

#### 2. 応答
```json
{
"sCLMID":"CLMMfdsGetHibuInfo",
"aCLMMfdsHibuInfo":
[
{
"sIssueCode":"6501",
"pBWRQ":"0.05"
}
]
}
```

---

## 5. 時価情報機能（REQUEST I/F）

### 1. 時価情報問合取得

#### 1. 要求
```json
{
"sCLMID":"CLMMfdsGetMarketPrice",
"sTargetIssueCode":"6501,6502,6503",
"sTargetColumn":"pDPP,tDPP:T,pPRP"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMMfdsGetMarketPrice |
| sTargetIssueCode | 対象銘柄コード | 取得したい銘柄コードをカンマ区切りで羅列。最大120銘柄まで指定可能。 |
| sTargetColumn | 対象情報コード | 取得したい情報コードをカンマ区切りで羅列。 |

#### 2. 応答
```json
{
"sCLMID":"CLMMfdsGetMarketPrice",
"aCLMMfdsMarketPrice":
[
{
"sIssueCode":"6501",
"pDPP":"oxox",
"pPRP":"oxox",
"tDPP:T":"oxox"
}
]
}
```


### 2. 蓄積情報問合取得

#### 1. 要求
```json
{
"sCLMID":"CLMMfdsGetMarketPriceHistory",
"sIssueCode":"6501",
"sSizyouC":"00"
}
```

| 項目名 | 名称 | 内容 |
| ------ | ------ | ------ |
| sCLMID | 機能ＩＤ | CLMMfdsGetMarketPriceHistory |
| sIssueCode | 銘柄コード | １要求１銘柄指定。 |
| sSizyouC | 市場 | 00：東証（引数省略可能。デフォルト＝東証） |

#### 2. 応答
```json
{
"sCLMID":"CLMMfdsGetMarketPriceHistory",
"sIssueCode":"6501",
"sSizyouC":"00",
"aCLMMfdsMarketPriceHistory":
[
{
"sDate":"YYYYMMDD",
"pDOP":"oxox",
"pDHP":"oxox",
"pDLP":"oxox",
"pDPP":"oxox",
"pDV":"oxox",
"pDOPxK":"oxox",
"pDHPxK":"oxox",
"pDLPxK":"oxox",
"pDPPxK":"oxox",
"pDVxK":"oxox",
"pSPUO":"oxox",
"pSPUC":"oxox",
"pSPUK":"oxox"
}
]
}
```

---

## 6. 注文約定通知（EVENT I/F）
時価及び注文約定通知（EVENT I/F 利用時）については、別途「立花証券・ｅ支店・ＡＰＩ（ｖ４ｒ７）、EVENT I/F 利用方法、データ仕様」の資料を参照下さい。

## 7. 結果コード、警告コード表 (1/4)

| コード | 分類 | テキスト | 理由 |
| ------ | ------ | ------ | ------ |
| 10001 | ログイン | ユーザーID、パスワードに誤りがあります | ユーザＩＤ不正 |
| 10002 | ログイン | ユーザーID、パスワードに誤りがあります | パスワード不正 |
| 10003 | ログイン | ユーザータイプに誤りがあります。 | ユーザタイプ不正 |
| 10004 | ログイン | 接続に誤りがあります | チャネル不正 |
| 10005 | ログイン | ＩＰアドレスに誤りがあります | ＩＰアドレス不正 |
| 10006 | ログイン | セッションＩＤに誤りがあります | セッションＩＤ不正 |
| 10007 | ログイン | ユーザエージェントに誤りがあります | ユーザエージェント不正 |
| 10008 | ログイン | ユーザーIDに誤りがあります | ユーザＩＤ不正 |
| 10009 | ログイン | パスキー認証でエラーが発生しました | パスキー認証エラー |
| 10020 | ログイン | 通信に問題があります | ダミーセッションファイル障害 |
| 10021 | ログイン | 通信に問題があります | ダミーセッションレコードなし |
| 10030 | ログイン | ログインに問題があります | ユーザー管理ファイル障害 |
| 10031 | ログイン | ログイン前に行う電話番号認証が認証されない、または、ユーザID、暗証番号をお間違えです。 | ユーザー管理レコードなし |
| 10032 | ログイン | ログインに問題があります | ユーザー管理レコード更新エラー |
| 10033 | ログイン | 電話番号認証が認証されない、ユーザID、暗証番号のご入力間違いが弊社規程回数を超えたため、現在ログイン停止中です。(ログイン停止の解除は、コールセンターまでお電話下さい。) | ユーザー管理ログインロック |
| 10034 | ログイン | ログインできません | ユーザー管理ログインＮＧ |
| 10035 | ログイン | ユーザIDか暗証番号をお間違えです。ご確認の上、再度ご入力下さい。 | ユーザー管理パスワード不一致 |
| 10036 | ログイン | ご入力された口座では当システムをご利用いただけません。ご利用のサービスをご確認ください。 | システム口座区分不一致 |
| 10037 | ログイン | システム口座区分が不正のため金商法交付ドキュメントが取得できません。 | システム口座区分不正 |
| 10038 | ログイン | ユーザIDか暗証番号をお間違えです。ご確認の上、再度ご入力下さい。 | マイページUSER管理顧客登録番号データなし |
| 10039 | ログイン | ユーザIDか暗証番号をお間違えです。ご確認の上、再度ご入力下さい。 | マイページUSER管理社員IDデータなし |
| 10040 | ログイン | システム設定ファイルで問題が発生しました | システム設定ファイル障害 |
| 10041 | ログイン | システム設定レコードがありません | システム設定レコードなし |
| 10045 | ログイン | BAD_IPファイルで問題が発生しました | BAD_IPファイル障害 |
| 10046 | ログイン | BAD_IPレコード作成でエラーが発生しました | BAD_IPレコード作成エラー |
| 10047 | ログイン | BAD_IP許可件数が制限を超えています | BAD_IP許可件数オーバー |
| 10050 | ログイン | セッション情報ファイルで問題が発生しました | セッション情報ファイル障害 |
| 10051 | ログイン | セッション情報レコードがありません | セッション情報レコードなし |
| 10052 | ログイン | セッション情報レコード更新でエラーが発生しました | セッション情報レコード更新エラー |
| 10053 | ログイン | セッション情報レコード作成でエラーが発生しました | セッション情報レコード作成エラー |
| 10054 | ログイン | セッションＩＤ生成でエラーが発生しました | セッションＩＤ生成エラー |
| 10055 | ログイン | 顧客マスタファイルで問題が発生しました | 顧客マスタファイル障害 |
| 10056 | ログイン | 顧客マスタレコードがありません | 顧客マスタレコードなし |
| 10057 | ログイン | 社員属性ファイルで問題が発生しました | 社員属性ファイル障害 |
| 10058 | ログイン | 社員属性レコードがありません | 社員属性レコードなし |
| 10060 | ログイン | 顧客情報ファイルで問題が発生しました | 顧客情報ファイル障害 |
| 10061 | ログイン | 顧客情報レコードがありません | 顧客情報レコードなし |
| 10062 | ログイン | 顧客情報レコード更新でエラーが発生しました | 顧客情報レコード更新エラー |
| 10063 | ログイン | お客様はパスキー未登録のため、通常ログインを行ってください。 | FIDO認証未登録 |
| 10064 | ログイン | お客様はパスキー登録済みのため、パスキーログインを行ってください。 | FIDO認証登録済み |
| 10065 | ログイン | 口座管理ファイルで問題が発生しました | 口座管理ファイル障害 |
| 10066 | ログイン | 口座管理レコードがありません | 口座管理レコードなし |
| 10067 | ログイン | 日付情報ファイルで問題が発生しました | 日付情報ファイル障害 |
| 10068 | ログイン | 日付情報レコードがありません | 日付情報レコードなし |
| 10069 | ログイン | 翌年用口座管理ファイルで問題が発生しました | 翌年用口座管理ファイル障害 |
| 10070 | ログイン | 現在時刻取得でエラーが発生しました | 現在時刻取得エラー |
| 10071 | ログイン | 当日日付取得でエラーが発生しました | 当日日付取得エラー |
| 10072 | ログイン | FIDO認証制御マスタファイルに問題が発生しました。 | FIDO認証制御マスタファイル障害 |
| 10073 | ログイン | FIDO設定解除客情報ファイルに問題が発生しました。 | FIDO設定解除客情報ファイル障害 |
| 10074 | ログイン | FIDO認証登録情報ファイルに問題が発生しました。 | FIDO認証登録取得情報 |
| 10075 | ログイン | 部店管理ファイルで問題が発生しました | 部店管理ファイル障害 |
| 10076 | ログイン | 部店管理レコードがありません | 部店管理レコードなし |
| 10077 | ログイン | FIDO認証制御マスタにレコードがありません。 | FIDO認証制御マスタレコードなし |
| 10078 | ログイン | 現在、パスキー認証利用停止中。電話番号認証してからログインしてください。 | FIDO認証未実施 |
| 10079 | ログイン | パスキー認証利用停止中。画面更新後、電話番号認証してから出金依頼してください。 | 顧客別FIDO認証未実施 |
| 10080 | ログイン | 情報サービス利用客ファイルに問題が発生しました | 情報サービス利用客ファイル障害 |
| 10081 | ログイン | 利用期限が切れております。引き続きご利用の場合には、証券口座ログイン後[市況・情報]-[情報サービス]からお申し込みください。 |  |
| 10082 | ログイン | ロック顧客ファイルに問題があります | ロック顧客ファイル障害 |
| 10083 | ログイン | マネロンリスク評価ファイルに問題があります | マネロンリスク評価ファイル障害 |
| 10084 | ログイン | 電話認証制御マスタにレコードがありません。 | 電話認証制御マスタレコードなし |
| 10085 | ログイン | 電話認証電話番号マスタにデータがありません。 | 電話認証電話番号マスタレコードなし |
| 10086 | ログイン | 顧客情報にデータがありません。 | 顧客情報取得失敗 |
| 10087 | ログイン | 当社に登録の電話番号から認証電話番号へかけた後にログインしてください。 | 電話番号未登録 |
| 10088 | ログイン | 当社に登録の電話番号から認証電話番号へかけた後にログインしてください。 | 着信なし、電話認証エラー |
| 10089 | ログイン | 当社に登録の電話番号から認証電話番号へかけた後、3分以内にログインしてください。 | 制限時間内の着信なし、電話認証エラー |
| 10091 | ログイン | 対象書面の閲覧日時の取得でエラーが発生しました。改めて書面をクリックして、再表示してください。 | 交付日時取得エラー |
| 10097 | ログイン | ネットワークでエラーが発生しました | ネットワークエラー |
| 10098 | ログイン | ＤＢでエラーが発生しました | ＤＢエラー |
| 10099 | ログイン | セッションがタイムアウトしました | タイムアウト |
| 10101 | お知らせ | 選択されたメッセージは削除済みのため、表示する事ができません。 | メッセージ削除済み |
| 10102 | お知らせ | 選択されたメッセージは存在しません。 | 該当メッセージなし |
| 10103 | お知らせ | 選択されたメッセージは削除済みのため、削除する事ができません。 | メッセージ削除済み |
| 10201 | お問い合わせ | お問い合わせ内容が長すぎます。2000文字以内にしてください。 | お問い合わせ内容文字列超過 |
| 10998 | HOME | ＤＢでエラーが発生しました | ＤＢエラー |
| 10999 | HOME | セッションがタイムアウトしました | タイムアウト |
| 11001 | 株式新規注文 | 注文種別に誤りがあります | 注文種別不正 |
| 11002 | 株式新規注文 | 親注文番号に誤りがあります | 親注文番号不正 |
| 11003 | 株式新規注文 | システム口座区分に誤りがあります | システム口座区分不正 |
| 11004 | 株式新規注文 | 部店コードに誤りがあります | 部店コード不正 |
| 11005 | 株式新規注文 | 顧客登録番号に誤りがあります | 顧客登録Ｎ不正 |
| 11006 | 株式新規注文 | 譲渡益課税区分に誤りがあります | 譲渡益課税区分不正 |
| 11007 | 株式新規注文 | 銘柄コードに誤りがあります | 銘柄コード不正 |
| 11008 | 株式新規注文 | 市場に誤りがあります | 市場不正 |
| 11009 | 株式新規注文 | 売買区分に誤りがあります | 売買区分不正 |
| 11010 | 株式新規注文 | 執行条件に誤りがあります | 執行条件不正 |
| 11011 | 株式新規注文 | 注文値段区分に誤りがあります | 注文値段区分不正 |
| 11012 | 株式新規注文 | 注文値段に誤りがあります | 注文値段不正 |
| 11013 | 株式新規注文 | 注文数量に誤りがあります | 注文数量不正 |
| 11014 | 株式新規注文 | 現金信用区分に誤りがあります | 現金信用区分不正 |
| 11015 | 株式新規注文 | 空売り符号に誤りがあります | 空売り符号不正 |
| 11016 | 株式新規注文 | 注文期日に誤りがあります | 注文期日不正 |
| 11017 | 株式新規注文 | 逆指値注文種別に誤りがあります | 逆指値注文種別不正 |
| 11018 | 株式新規注文 | 逆指値条件に誤りがあります | 逆指値条件不正 |
| 11019 | 株式新規注文 | 逆指値値段区分に誤りがあります | 逆指値値段区分不正 |
| 11020 | 株式新規注文 | 逆指値値段に誤りがあります | 逆指値値段不正 |
| 11021 | 株式新規注文 | 接続に誤りがあります | チャネル不正 |
| 11022 | 株式新規注文 | 接続に誤りがあります | チャネル詳細不正 |
| 11023 | 株式新規注文 | ＩＰアドレスに誤りがあります | ＩＰアドレス不正 |
| 11024 | 株式新規注文 | 建日種類に誤りがあります | 建日種類不正（建日種類を1：個別指定以外に指定して、返済リストに建玉を列挙した場合にも発生） |
| 11025 | 株式新規注文 | 建玉番号に誤りがあります | 建玉番号不正 |
| 11026 | 株式新規注文 | 建玉順位に誤りがあります | 建玉順位不正 |
| 11027 | 株式新規注文 | 建玉数量に誤りがあります | 建玉数量不正 |
| 11028 | 株式新規注文 | 返済数量に誤りがあります | 返済数量不正 |
| 11029 | 株式新規注文 | 第二暗証番号省略フラグに誤りがあります | 第二暗証番号省略フラグ不正 |
| 11030 | 株式新規注文 | 第二暗証番号に誤りがあります | 第二暗証番号不正 |
| 11031 | 株式新規注文 |  | チェックのみフラグ不正 |
| 11032 | 株式新規注文 | 不成注文に成行が指定されています | 不成注文に成行が指定されています |
| 11033 | 株式新規注文 | 注文期限を指定する場合は、執行条件は｢無条件｣を指定して下さい。 | 期限付注文執行条件エラー |
| 11034 | 株式新規注文 | 逆指値注文執行条件でエラーが発生しました | 逆指値注文執行条件エラー |
| 11035 | 株式新規注文 | 通常＋逆指値注文執行条件でエラーが発生しました | 通常＋逆指値注文執行条件エラー |
| 11036 | 株式新規注文 | 子注文に執行条件でエラーが発生しました | 子注文に執行条件エラー |
| 11037 | 株式新規注文 | 子注文に注文期限でエラーが発生しました | 子注文に注文期限エラー |
| 11039 | 株式新規注文 | 端株に指値は指定出来ない | 端株に指値は指定出来ない |
| 11040 | 株式新規注文 | 非課税口座チャネルでエラーが発生しました | 非課税口座チャネルエラー |
| 11041 | 株式新規注文 | 非課税口座取引でエラーが発生しました（現物のみ） | 非課税口座取引エラー（現物のみ） |
| 11042 | 株式新規注文 | 非課税口座執行条件でエラーが発生しました（指定なしのみ） | 非課税口座執行条件エラー（指定なしのみ） |
| 11043 | 株式新規注文 | 非課税口座値段区分でエラーが発生しました（指値のみ） | 非課税口座値段区分エラー（指値のみ） |
| 11044 | 株式新規注文 | 非課税口座注文期限でエラーが発生しました（当日中のみ） | 非課税口座注文期限エラー（当日中のみ） |
| 11045 | 株式新規注文 | 非課税口座特殊注文でエラーが発生しました（特殊注文はできません） | 非課税口座特殊注文エラー（特殊注文禁止） |
| 11046 | 株式新規注文 | 弁済区分が選択されていません | 弁済区分不正 |
| 11047 | 株式新規注文 | ｢成行｣を指定されていますが、注文単価も入力されています。指値の場合は｢指値｣に印を付けて下さい。 | 成行指値同時指定 |
| 11048 | 株式新規注文 | 通常＋逆指値値段でエラーが発生しました | 通常＋逆指値値段エラー |
| 11100 | 株式新規注文 | 運用ステータス(注文)にデータがありません | 運用ステータス(注文)レコードなし |
| 11101 | 株式新規注文 | 運用ステータス(採用値幅)にデータがありません | 運用ステータス(採用値幅)レコードなし |
| 11102 | 株式新規注文 | 只今の時間帯は受付できません | 運用ステータス(注文).受付停止 |
| 11103 | 株式新規注文 | 日付情報にデータがありません | 日付情報レコードなし |
| 11104 | 株式新規注文 | 銘柄がありません | 銘柄マスタレコードなし |
| 11105 | 株式新規注文 | 当該銘柄は売買停止中です | 銘柄マスタ.売買停止エラー |
| 11106 | 株式新規注文 | 当該銘柄は市場に直接お取り次ぎすることができません | 銘柄マスタ.場伝票出力有無エラー |
| 11107 | 株式新規注文 | 当該銘柄はNISA口座への買付ができません | 銘柄マスタ.非課税口座エラー |
| 11108 | 株式新規注文 | 銘柄市場マスタにデータがありません | 銘柄市場マスタレコードなし |
| 11109 | 株式新規注文 | 当該銘柄は前日終値がないため成行はできません | 銘柄市場マスタ.前日終値なし(成行禁止) |
| 11110 | 株式新規注文 | 当該銘柄は上場終了しています | 銘柄市場マスタ.上場廃止日エラー |
| 11111 | 株式新規注文 | 当該銘柄は上場前です | 銘柄市場マスタ.新規上場日エラー |
| 11112 | 株式新規注文 | 当該銘柄の売買単位の整数倍の数量を入力してください | 銘柄マスタ.売買単位エラー |
| 11113 | 株式新規注文 | 当該銘柄の値幅制限内の単価を入力してください | 銘柄市場マスタ.値幅エラー |
| 11114 | 株式新規注文 |  | 銘柄市場マスタ.制度信用エラー |
| 11115 | 株式新規注文 | 当該銘柄では新規売建はお取り扱いできません | 銘柄市場マスタ.信用売建エラー |
| 11116 | 株式新規注文 | 当該銘柄ではご指定の弁済区分での新規売建はお取り扱いできません | 一般信用売建エラー |
| 11117 | 株式新規注文 | 呼値にデータがありません | 呼値レコードなし |
| 11118 | 株式新規注文 | 正しい呼値の単位で単価を入力してください | 呼値エラー |
| 11119 | 株式新規注文 | 当該銘柄の信用属性でエラーが発生しました | 銘柄市場マスタ.信用属性エラー |
| 11120 | 株式新規注文 | 注文期日でエラーが発生しました | 注文期日エラー |
| 11121 | 株式新規注文 | 逆指値段には当該銘柄の値幅制限内の単価を入力してください | 逆指値段値幅エラー |
| 11122 | 株式新規注文 | 逆指値段呼値にデータがありません | 逆指値段呼値レコードなし |
| 11123 | 株式新規注文 | 正しい呼値の単位で逆指値段を入力してください | 逆指値段呼値エラー |
| 11124 | 株式新規注文 | 執行単価が0以下です | 執行値段マイナスエラー |
| 11125 | 株式新規注文 | システム別設定にデータがありません | システム別設定レコードなし |
| 11126 | 株式新規注文 | このサービスは取り扱っておりません | システム別設定.現物未実施 |
| 11127 | 株式新規注文 | このサービスは取り扱っておりません | システム別設定.制度信用未実施 |
| 11128 | 株式新規注文 | このサービスは取り扱っておりません | システム別設定.一般信用未実施 |
| 11129 | 株式新規注文 | システム市場弁済別取扱条件にデータがありません | システム市場弁済別取扱条件レコードなし |
| 11130 | 株式新規注文 | このサービスは取り扱っておりません | サービス別取扱.現物買付取扱不可 |
| 11131 | 株式新規注文 | このサービスは取り扱っておりません | サービス別取扱.現物売付取扱不可 |
| 11132 | 株式新規注文 | このサービスは取り扱っておりません | サービス別取扱.信用新規取扱不可 |
| 11133 | 株式新規注文 | このサービスは取り扱っておりません | サービス別取扱.信用返済取扱不可 |
| 11134 | 株式新規注文 | このサービスは取り扱っておりません | サービス別取扱.現受現渡取扱不可 |
| 11135 | 株式新規注文 | 当該市場ではお取引できません | 市場別設定.取引不可 |
| 11136 | 株式新規注文 | 寄付注文はできません | 商品市場別設定.執行条件寄付不可 |
| 11137 | 株式新規注文 | 引け注文はできません | 商品市場別設定.執行条件引け不可 |
| 11138 | 株式新規注文 | 不成注文はできません | 商品市場別設定.執行条件不成不可 |
| 11139 | 株式新規注文 | 連続注文はできません | 商品市場別設定.連続注文不可 |
| 11140 | 株式新規注文 | 出来るまで注文はできません | 商品市場別設定.出来るまで注文不可 |
| 11141 | 株式新規注文 | 当該銘柄はお取引できません | 銘柄別市場別規制.停止区分取引禁止 |
| 11142 | 株式新規注文 | 当該銘柄の成行注文はできません | 銘柄別市場別規制.停止区分成行禁止 |
| 11143 | 株式新規注文 | 当該銘柄の買付の注文はできません | 銘柄別市場別規制.現物買付取引禁止 |
| 11144 | 株式新規注文 | 当該銘柄の買付の成行注文はできません | 銘柄別市場別規制.現物買付成行禁止 |
| 11145 | 株式新規注文 | 当該銘柄の売付の注文はできません | 銘柄別市場別規制.現物売付取引禁止 |
| 11146 | 株式新規注文 | 当該銘柄の売付の成行注文はできません | 銘柄別市場別規制.現物売付成行禁止 |
| 11147 | 株式新規注文 | 当該銘柄の制度信用の新規買建注文はできません | 銘柄別市場別規制.制度信用買建取引禁止 |
| 11148 | 株式新規注文 | 当該銘柄の制度信用の新規買建の成行注文はできません | 銘柄別市場別規制.制度信用買建成行禁止 |
| 11149 | 株式新規注文 | 当該銘柄の制度信用の新規売建注文はできません | 銘柄別市場別規制.制度信用売建取引禁止 |
| 11150 | 株式新規注文 | 当該銘柄の制度信用の新規売建の成行注文はできません | 銘柄別市場別規制.制度信用売建成行禁止 |
| 11151 | 株式新規注文 | 当該銘柄の制度信用の買返済注文はできません | 銘柄別市場別規制.制度信用買返済取引禁止 |
| 11152 | 株式新規注文 | 当該銘柄の制度信用の買返済の成行注文はできません | 銘柄別市場別規制.制度信用買返済成行禁止 |
| 11153 | 株式新規注文 | 当該銘柄の制度信用の売返済注文はできません | 銘柄別市場別規制.制度信用売返済取引禁止 |
| 11154 | 株式新規注文 | 当該銘柄の制度信用の売返済の成行注文はできません | 銘柄別市場別規制.制度信用売返済成行禁止 |
| 11155 | 株式新規注文 | 当該銘柄の一般信用の新規買建注文はできません | 銘柄別市場別規制.一般信用買建取引禁止 |
| 11156 | 株式新規注文 | 当該銘柄の一般信用の新規買建の成行注文はできません | 銘柄別市場別規制.一般信用買建成行禁止 |
| 11157 | 株式新規注文 | 当該銘柄の一般信用の新規売建注文はできません | 銘柄別市場別規制.一般信用売建取引禁止 |
| 11158 | 株式新規注文 | 当該銘柄の一般信用の新規売建の成行注文はできません | 銘柄別市場別規制.一般信用売建成行禁止 |
| 11159 | 株式新規注文 | 当該銘柄の一般信用の買返済注文はできません | 銘柄別市場別規制.一般信用買返済取引禁止 |
| 11160 | 株式新規注文 | 当該銘柄の一般信用の買返済の成行注文はできません | 銘柄別市場別規制.一般信用買返済成行禁止 |
| 11161 | 株式新規注文 | 当該銘柄の一般信用の売返済注文はできません | 銘柄別市場別規制.一般信用売返済取引禁止 |
| 11162 | 株式新規注文 | 当該銘柄の一般信用の売返済の成行注文はできません | 銘柄別市場別規制.一般信用売返済成行禁止 |
| 11163 | 株式新規注文 | 当該銘柄の事前調整取引はできません | 銘柄別市場別規制.事前調整取引禁止 |
| 11164 | 株式新規注文 | 当該銘柄の即日入金取引はできません | 銘柄別市場別規制.即日入金取引禁止 |
| 11165 | 株式新規注文 | 当該銘柄の即日入金取引の成行注文はできません | 銘柄別市場別規制.即日入金取引成行禁止 |
| 11166 | 株式新規注文 | 当該銘柄の制度信用の現渡注文はできません | 銘柄別市場別規制.制度信用現渡取引禁止 |
| 11167 | 株式新規注文 | 当該銘柄の制度信用の現引注文はできません | 銘柄別市場別規制.制度信用現引取引禁止 |
| 11168 | 株式新規注文 | 当該銘柄の一般信用の現渡注文はできません | 銘柄別市場別規制.一般信用現渡済取引禁止 |
| 11169 | 株式新規注文 | 当該銘柄の一般信用の現引注文はできません | 銘柄別市場別規制.一般信用現引取引禁止 |
| 11170 | 株式新規注文 | サービス別取扱レコードがありません | サービス別取扱レコードなし |
| 11171 | 株式新規注文 | 市場別設定レコードがありません | 市場別設定レコードなし |
| 11172 | 株式新規注文 | 商品市場別設定レコードがありません | 商品市場別設定レコードなし |
| 11173 | 株式新規注文 | 当該銘柄の即日入金取引の期限付き注文はできません | 銘柄別市場別規制.即日入金期限付き注文禁止 |
| 11174 | 株式新規注文 | 当該銘柄では特定口座でのお取り扱いはできません | 銘柄マスタ.特定口座対象Ｃエラー |
| 11175 | 株式新規注文 | 上場投信信託（ETF）は、上場日当日の８：００頃より、ご注文の入力が可能となります | 銘柄市場マスタ.値幅ゼロ |
| 11176 | 株式新規注文 | 当該銘柄の端株買付の注文はできません | 銘柄別市場別規制.端株買付取引禁止 |
| 11177 | 株式新規注文 | 当該銘柄の端株売付の注文はできません | 銘柄別市場別規制.端株売付取引禁止 |
| 11245 | 株式新規注文 | システム状態にデータがありません | システム状態レコードなし |
| 11246 | 株式新規注文 | システムが受付可能時間外です。 | システム状態.ログイン不許可 |
| 11247 | 株式新規注文 | システムが受付可能時間外です。 | システム状態.閉局 |
| 11288 | 株式新規注文 | 子注文同一銘柄でエラーが発生しました | 子注文同一銘柄エラー |
| 11289 | 株式新規注文 | 子注文件数が制限を超えています | 子注文件数オーバー |
| 11299 | 株式新規注文 | 顧客マスタファイルに問題があります | 顧客マスタファイル障害 |
| 11300 | 株式新規注文 | 顧客マスタにデータがありません | 顧客マスタレコードなし |
| 11301 | 株式新規注文 | 顧客マスタ.精算理由でエラーが発生しました | 顧客マスタ.精算理由エラー |
| 11302 | 株式新規注文 | 顧客情報ファイルに問題があります | 顧客情報ファイル障害 |
| 11303 | 株式新規注文 | 顧客情報にデータがありません | 顧客情報レコードなし |
| 11304 | 株式新規注文 | 第二暗証番号が誤っています | 顧客マスタ.第二パスワード不一致 |
| 11305 | 株式新規注文 | 口座管理ファイルに問題があります | 口座管理ファイル障害 |
| 11306 | 株式新規注文 | 口座管理にデータがありません | 口座管理レコードなし |
| 11307 | 株式新規注文 | 特定口座が未開設です | 口座管理.特定口座未開設 |
| 11308 | 株式新規注文 | NISA口座が未開設です | 口座管理.非課税口座未開設 |
| 11309 | 株式新規注文 | 信用口座が未開設です | 口座管理.信用口座未開設 |
| 11310 | 株式新規注文 | 外国口座が未開設です | 口座管理.外国口座未開設 |
| 11311 | 株式新規注文 | ロック顧客ファイルに問題があります | ロック顧客ファイル障害 |
| 11312 | 株式新規注文 | 現在、お客様の口座には、お取引制限がかかっています。コールセンターまでお問い合わせ下さい。 | ロック顧客該当エラー |
| 11313 | 株式新規注文 | インサイダファイルに問題があります | インサイダファイル障害 |
| 11314 | 株式新規注文 | 当該注文はインサイダー情報に基づく注文ではない同意が無い為受付できません | インサイダチェックエラー |
| 11315 | 株式新規注文 | 特定投資家契約マスタファイルに問題があります | 特定投資家契約マスタファイル障害 |
| 11316 | 株式新規注文 | 特定投資家契約マスタチェックでエラーが発生しました | 特定投資家契約マスタチェックエラー |
| 11317 | 株式新規注文 | 金商法交付書面ファイルに問題があります | 金商法交付書面ファイル障害 |
| 11318 | 株式新規注文 | 金商法交付書面(当日分)ファイルに問題があります | 金商法交付書面(当日更新分)ファイル障害 |
| 11319 | 株式新規注文 | 金商法交付書面チェックでエラーが発生しました | 金商法交付書面チェックエラー |
| 11320 | 株式新規注文 | 顧客銘柄別取引停止ファイルに問題があります | 顧客銘柄別取引停止ファイル障害 |
| 11321 | 株式新規注文 | 顧客銘柄別取引停止にデータがありません | 顧客銘柄別取引停止レコードなし |
| 11322 | 株式新規注文 | お客様の当該銘柄における現物買付のお取引を停止させていただいております | 顧客銘柄別取引停止.現物買付停止 |
| 11323 | 株式新規注文 | お客様の当該銘柄における現物売付のお取引を停止させていただいております | 顧客銘柄別取引停止.現物売付停止 |
| 11324 | 株式新規注文 | お客様の当該銘柄における信用新規買建のお取引を停止させていただいております | 顧客銘柄別取引停止.信用新規買建停止 |
| 11325 | 株式新規注文 | お客様の当該銘柄における信用新規売建のお取引を停止させていただいております | 顧客銘柄別取引停止.信用新規売建停止 |
| 11326 | 株式新規注文 | お客様の当該銘柄における信用買返済のお取引を停止させていただいております | 顧客銘柄別取引停止.信用買返済停止 |
| 11327 | 株式新規注文 | お客様の当該銘柄における信用売返済のお取引を停止させていただいております | 顧客銘柄別取引停止.信用売返済停止 |
| 11328 | 株式新規注文 | お客様の当該銘柄における信用現引のお取引を停止させていただいております | 顧客銘柄別取引停止.信用現引停止 |
| 11329 | 株式新規注文 | お客様の当該銘柄における信用現渡のお取引を停止させていただいております | 顧客銘柄別取引停止.信用現渡停止 |
| 11340 | 株式新規注文 | 市場別特殊執行注文取扱停止ファイルに問題があります | 市場別特殊執行注文取扱停止ファイル障害 |
| 11341 | 株式新規注文 | 当該市場での逆指値注文の受付を停止しています | 市場別特殊執行注文取扱停止.逆指値停止 |
| 11342 | 株式新規注文 | 当該市場での通常＋逆指値注文の受付を停止しています | 市場別特殊執行注文取扱停止.通常＋逆指値停止 |
| 11343 | 株式新規注文 | 商品別特殊執行注文取扱停止ファイルに問題があります | 商品別特殊執行注文取扱停止ファイル障害 |
| 11344 | 株式新規注文 | 逆指値注文の受付を停止しています | 商品別特殊執行注文取扱停止.逆指値停止 |
| 11345 | 株式新規注文 | 通常＋逆指値注文の受付を停止しています | 商品別特殊執行注文取扱停止.通常＋逆指値停止 |
| 11346 | 株式新規注文 | 銘柄別特殊執行注文取扱停止ファイルに問題があります | 銘柄別特殊執行注文取扱停止ファイル障害 |
| 11347 | 株式新規注文 | 当該銘柄は逆指値注文はできません | 銘柄別特殊執行注文取扱停止.逆指値停止 |
| 11348 | 株式新規注文 | 当該銘柄は通常＋逆指値はできません | 銘柄別特殊執行注文取扱停止.通常＋逆指値停止 |
| 11349 | 株式新規注文 | 当該銘柄は特殊執行注文はできません | 銘柄別特殊執行注文取扱停止.特殊執行注文不可 |
| 11350 | 株式新規注文 | ハードリミット市場別ファイルに問題があります | ハードリミット市場別ファイル障害 |
| 11351 | 株式新規注文 | ハードリミット市場別にデータがありません | ハードリミット市場別レコードなし |
| 11352 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合1 |
| 11353 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合2 |
| 11354 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合3 |
| 11355 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合4 |
| 11356 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合5 |
| 11360 | 株式新規注文 | ハードリミットファイルに問題があります | ハードリミットファイル障害 |
| 11361 | 株式新規注文 | ハードリミットにデータがありません | ハードリミットレコードなし |
| 11362 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量買上限 |
| 11363 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量売上限 |
| 11364 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量買新規上限 |
| 11365 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量売新規上限 |
| 11366 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量買返済上限 |
| 11367 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量売返済上限 |
| 11368 | 株式新規注文 | 発注金額が弊社規定の制限を越えています | ハードリミット.発注金額上限 |
| 11369 | 株式新規注文 | 建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ハードリミット.建玉総数量上限 |
| 11370 | 株式新規注文 | 売建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ハードリミット.売建玉総数量上限 |
| 11371 | 株式新規注文 | 建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ハードリミット.建玉総金額上限 |
| 11372 | 株式新規注文 | 当該銘柄の建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ハードリミット.建玉銘柄総金額上限 |
| 11380 | 株式新規注文 | 注文できません。個別ファイルに問題があります | ソフトリミット個別ファイル障害 |
| 11381 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量買上限 |
| 11382 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量売上限 |
| 11383 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量買新規上限 |
| 11384 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量売新規上限 |
| 11385 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量買返済上限 |
| 11386 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量売返済上限 |
| 11387 | 株式新規注文 | 発注金額が弊社規定の制限を越えています | ソフトリミット個別.発注金額上限 |
| 11388 | 株式新規注文 | 建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット個別.建玉総数量上限 |
| 11389 | 株式新規注文 | 売建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット個別.売建玉総数量上限 |
| 11390 | 株式新規注文 | 建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット個別.建玉総金額上限 |
| 11391 | 株式新規注文 | 当該銘柄の建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット個別.建玉銘柄総金額上限 |
| 11400 | 株式新規注文 | 注文できません。通常ファイルに問題があります | ソフトリミット通常ファイル障害 |
| 11401 | 株式新規注文 | 注文できません。通常ファイルにデータがありません | ソフトリミット通常レコードなし |
| 11402 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量買上限 |
| 11403 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量売上限 |
| 11404 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量買新規上限 |
| 11405 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量売新規上限 |
| 11406 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量買返済上限 |
| 11407 | 株式新規注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量売返済上限 |
| 11408 | 株式新規注文 | 発注金額が弊社規定の制限を越えています | ソフトリミット通常.発注金額上限 |
| 11409 | 株式新規注文 | 建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット通常.建玉総数量上限 |
| 11410 | 株式新規注文 | 売建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット通常.売建玉総数量上限 |
| 11411 | 株式新規注文 | 建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット通常.建玉総金額上限 |
| 11412 | 株式新規注文 | 当該銘柄の建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット通常.建玉銘柄総金額上限 |
| 11415 | 株式新規注文 | 空売り注文はできません | 空売り規制(注文不可) |
| 11416 | 株式新規注文 | 空売り成行注文はできません | 空売り規制(成行不可) |
| 11417 | 株式新規注文 | 空売り期限付き注文はできません | 空売り規制(期限付き不可) |
| 11420 | 株式新規注文 | 買付可能額が不足しています | 買付可能額不足 |
| 11421 | 株式新規注文 | 売付可能数量が不足しています | 売付可能数量不足 |
| 11422 | 株式新規注文 | 新規建余力は、%s 円です。ご注文に対して現金保証金が、%s 円不足しています。 | 信用新規建可能額不足 |
| 11423 | 株式新規注文 | 現引可能額が不足しています | 現引可能額不足 |
| 11424 | 株式新規注文 | この注文単価では%s株までしか入力できません | 売付可能数量不足 |
| 11425 | 株式新規注文 | この注文単価ではこの注文株数は入力は出来ません | 売付可能数量不足 |
| 11426 | 株式新規注文 | この銘柄は日計りの対象となっているため、現在売付可能株数は%s株となります。ご注文された%s株を発注するためには、お預かり金が%s円不足しております。 | 売付可能数量不足 |
| 11427 | 株式新規注文 | この銘柄は日計りの対象となっているため、余力が不足します。 | 売付可能数量不足 |
| 11428 | 株式新規注文 | 増担保の現金必要保証金が不足します。 | 現金必要保証金不足 |
| 11429 | 株式新規注文 | 信用新規建可能額が不足しています | 信用新規建可能額不足 |
| 11430 | 株式新規注文 | 現在、この銘柄の買付可能額は、%s 円です。%s 円不足しているため、このご注文はお受けできません。(日計り取引銘柄等、銘柄によっては可能額の算出方法が異なるケースがあります。) | 買付可能額不足 |
| 11440 | 株式新規注文 | 非課税口座管理ファイルに問題があります | 非課税口座管理ファイル障害 |
| 11441 | 株式新規注文 | 非課税口座管理にデータがありません | 非課税口座管理レコードなし |
| 11442 | 株式新規注文 | 非課税口座管理更新でエラーが発生しました | 非課税口座管理更新エラー |
| 11443 | 株式新規注文 | 非課税口座可能額が不足しています | 非課税口座可能額不足 |
| 11450 | 株式新規注文 | 顧客手数料ファイルに問題があります | 顧客手数料ファイル障害 |
| 11451 | 株式新規注文 |  | 顧客手数料レコードなし |
| 11460 | 株式新規注文 | 手数料マスタファイルに問題があります | 手数料マスタファイル障害 |
| 11461 | 株式新規注文 |  | 手数料マスタレコードなし |
| 11470 | 株式新規注文 | 課税率マスタファイルに問題があります | 課税率マスタファイル障害 |
| 11471 | 株式新規注文 |  | 課税率マスタレコードなし |
| 11480 | 株式新規注文 | 保管顧客課税別ファイルに問題があります | 保管顧客課税別ファイル障害 |
| 11481 | 株式新規注文 | 選択した口座区分がお預かり銘柄と不一致のため、このご注文はお受けできません。 | 保管顧客課税別にデータがありません |
| 11482 | 株式新規注文 | 売付可能な株数が不足しているため、このご注文はお受けできません。売却可能株数や注文一覧画面をご確認ください。 | 保管顧客課税別にデータがありません |
| 11483 | 株式新規注文 | 保管顧客課税別にデータ作成でエラーが発生しました | 保管顧客課税別レコード作成エラー |
| 11484 | 株式新規注文 | 保管顧客課税別のデータ更新でエラーが発生しました | 保管顧客課税別レコード更新エラー |
| 11490 | 株式新規注文 | 信用建玉明細ファイルに問題があります | 信用建玉明細ファイル障害 |
| 11491 | 株式新規注文 | 信用建玉明細にデータがありません | 信用建玉明細レコードなし |
| 11492 | 株式新規注文 | 信用建玉明細有効数量がありません | 信用建玉明細有効数量なし |
| 11493 | 株式新規注文 | 信用建玉明細のデータ更新でエラーが発生しました | 信用建玉明細レコード更新エラー |
| 11494 | 株式新規注文 | 信用建玉残ファイルに問題があります | 信用建玉残ファイル障害 |
| 11495 | 株式新規注文 | 信用建玉残にデータがありません | 信用建玉残レコードなし |
| 11496 | 株式新規注文 | 信用建玉残にデータ作成でエラーが発生しました | 信用建玉残レコード作成エラー |
| 11497 | 株式新規注文 | 信用建玉残のデータ更新でエラーが発生しました | 信用建玉残レコード更新エラー |
| 11500 | 株式新規注文 | 顧客金銭ファイルに問題があります | 顧客金銭ファイル障害 |
| 11501 | 株式新規注文 | 顧客金銭にデータがありません | 顧客金銭レコードなし |
| 11502 | 株式新規注文 | 顧客拘束金ファイルに問題があります | 顧客拘束金ファイル障害 |
| 11503 | 株式新規注文 | 顧客拘束金にデータがありません | 顧客拘束金レコードなし |
| 11504 | 株式新規注文 |  | 顧客拘束金レコードレコード作成エラー |
| 11505 | 株式新規注文 |  | 顧客拘束金レコード更新エラー |
| 11509 | 株式新規注文 | 保証金率取得でエラーが発生しました | 保証金率取得エラー |
| 11510 | 株式新規注文 | 代用掛目ファイルに問題があります | 代用掛目ファイル障害 |
| 11511 | 株式新規注文 | 日付情報ファイルに問題があります | 日付情報ファイル障害 |
| 11512 | 株式新規注文 | 保管顧客別残ファイルに問題があります | 保管顧客別残ファイル障害 |
| 11513 | 株式新規注文 | 保管顧客別残にデータがありません | 保管顧客別残レコードなし |
| 11514 | 株式新規注文 | 保管顧客別残にデータ作成でエラーが発生しました | 保管顧客別残レコード作成エラー |
| 11515 | 株式新規注文 | 保管顧客別残のデータ更新でエラーが発生しました | 保管顧客別残レコード更新エラー |
| 11516 | 株式新規注文 | 保証金推移ファイルに問題があります | 保証金推移ファイル障害 |
| 11517 | 株式新規注文 | 保証金推移にデータがありません | 保証金推移レコードなし |
| 11518 | 株式新規注文 | 保証金推移にデータ作成でエラーが発生しました | 保証金推移レコード作成エラー |
| 11519 | 株式新規注文 | 保証金推移のデータ更新でエラーが発生しました | 保証金推移レコード更新エラー |
| 11520 | 株式新規注文 | 顧客当日取引情報ファイルに問題があります | 顧客当日取引情報ファイル障害 |
| 11521 | 株式新規注文 | 顧客当日取引情報にデータがありません | 顧客当日取引情報レコードなし |
| 11522 | 株式新規注文 | 顧客当日取引情報にデータ作成でエラーが発生しました | 顧客当日取引情報レコード作成エラー |
| 11523 | 株式新規注文 | 顧客当日取引情報のデータ更新でエラーが発生しました | 顧客当日取引情報レコード更新エラー |
| 11524 | 株式新規注文 | 差金決済管理明細ファイルに問題があります | 差金決済管理明細ファイル障害 |
| 11525 | 株式新規注文 | 差金決済管理明細にデータがありません | 差金決済管理明細レコードなし |
| 11526 | 株式新規注文 | 差金決済管理明細にデータ作成でエラーが発生しました | 差金決済管理明細レコード作成エラー |
| 11527 | 株式新規注文 | 差金決済管理明細のデータ更新でエラーが発生しました | 差金決済管理明細レコード更新エラー |
| 11530 | 株式新規注文 | 譲渡益台帳ファイルに問題が発生しました | 譲渡益台帳ファイル障害 |
| 11531 | 株式新規注文 | 譲渡益台帳レコードがありません | 譲渡益台帳レコードなし |
| 11532 | 株式新規注文 | 譲渡益台帳レコード作成でエラーが発生しました | 譲渡益台帳レコード作成エラー |
| 11534 | 株式新規注文 | 譲渡益台帳レコード更新でエラーが発生しました | 譲渡益台帳レコード更新エラー |
| 11600 | 株式新規注文 | 注文番号（株式）ファイルに問題があります | 注文番号（株式）ファイル障害 |
| 11601 | 株式新規注文 | 注文番号（株式）にデータがありません | 注文番号（株式）レコードなし |
| 11602 | 株式新規注文 | 建玉番号（株式）ファイルに問題があります | 建玉番号（株式）ファイル障害 |
| 11603 | 株式新規注文 | 建玉番号（株式）にデータがありません | 建玉番号（株式）レコードなし |
| 11604 | 株式新規注文 | 親注文株式サマリファイルに問題があります | 親注文株式サマリファイル障害 |
| 11605 | 株式新規注文 | 親注文株式サマリにデータがありません | 親注文株式サマリレコードなし |
| 11606 | 株式新規注文 | 親注文株式サマリ有効数量がありません | 親注文株式サマリ有効数量なし |
| 11607 | 株式新規注文 | 親注文株式サマリ数量を超えています | 親注文株式サマリ数量オーバー |
| 11608 | 株式新規注文 |  | 親注文株式サマリレコード更新エラー |
| 11609 | 株式新規注文 | 株式サマリファイルに問題があります | 株式サマリファイル障害 |
| 11610 | 株式新規注文 | 株式サマリにデータがありません | 株式サマリレコードなし |
| 11611 | 株式新規注文 | 株式サマリにデータ作成でエラーが発生しました | 株式サマリレコード作成エラー |
| 11612 | 株式新規注文 | 株式サマリのデータ更新でエラーが発生しました | 株式サマリレコード更新エラー |
| 11613 | 株式新規注文 | 株式明細ファイルに問題があります | 株式明細ファイル障害 |
| 11614 | 株式新規注文 | 株式明細にデータがありません | 株式明細レコードなし |
| 11615 | 株式新規注文 | 株式明細更新でエラーが発生しました | 株式明細更新エラー |
| 11616 | 株式新規注文 | 株式明細にデータ作成でエラーが発生しました | 株式明細レコード作成エラー |
| 11617 | 株式新規注文 | 株式注文約定履歴ファイルに問題があります | 株式注文約定履歴ファイル障害 |
| 11618 | 株式新規注文 | 株式注文約定履歴作成でエラーが発生しました | 株式注文約定履歴作成エラー |
| 11621 | 株式新規注文 | 株式返済予約ファイルに問題があります | 株式返済予約ファイル障害 |
| 11622 | 株式新規注文 | 株式返済予約にデータがありません | 株式返済予約レコードなし |
| 11623 | 株式新規注文 | 株式返済予約にデータ作成でエラーが発生しました | 株式返済予約レコード作成エラー |
| 11624 | 株式新規注文 | 株式返済予約のデータ更新でエラーが発生しました | 株式返済予約レコード更新エラー |
| 11625 | 株式新規注文 | 株式返済明細ファイルに問題があります | 株式返済明細ファイル障害 |
| 11626 | 株式新規注文 | 株式返済明細にデータがありません | 株式返済明細レコードなし |
| 11627 | 株式新規注文 | 株式返済明細にデータ作成でエラーが発生しました | 株式返済明細レコード作成エラー |
| 11628 | 株式新規注文 | 株式返済明細のデータ更新でエラーが発生しました | 株式返済明細レコード更新エラー |
| 11640 | 株式新規注文 | 株式約定失効ファイルに問題があります | 株式約定失効ファイル障害 |
| 11641 | 株式新規注文 | 株式約定失効にデータがありません | 株式約定失効レコードなし |
| 11642 | 株式新規注文 | 株式約定失効にデータ作成でエラーが発生しました | 株式約定失効レコード作成エラー |
| 11643 | 株式新規注文 | 株式約定失効のデータ更新でエラーが発生しました | 株式約定失効レコード更新エラー |
| 11645 | 株式新規注文 | システム別設定ファイルに問題があります | システム別設定ファイル障害 |
| 11646 | 株式新規注文 | 銘柄マスタ（株式）ファイルに問題があります | 銘柄マスタ（株式）ファイル障害 |
| 11647 | 株式新規注文 | 銘柄市場マスタ（株式）ファイルに問題があります | 銘柄市場マスタ（株式）ファイル障害 |
| 11648 | 株式新規注文 | 銘柄別・市場別規制（株式）ファイルに問題があります | 銘柄別・市場別規制（株式）ファイル障害 |
| 11700 | 株式新規注文 | 運用ステータス(申告)にデータがありません | 運用ステータス(申告)レコードなし |
| 11701 | 株式新規注文 | 只今の時間帯は受付できません | 運用ステータス(申告).受付停止 |
| 11702 | 株式新規注文 | 運用ステータス(連続注文)にデータがありません | 運用ステータス(連続注文)レコードなし |
| 11703 | 株式新規注文 | 只今の時間帯は受付できません | 運用ステータス(連続注文).受付停止 |
| 11800 | 株式新規注文 | 余力制御ファイルに問題があります | 余力制御ファイル障害 |
| 11802 | 株式新規注文 | お客様のお取引を停止させていただいております | 余力制御.取引停止 |
| 11803 | 株式新規注文 | お客様の信用新規建のお取引を停止させていただいております | 余力制御.信用新規建停止 |
| 11806 | 株式新規注文 | お客様のその他商品買付のお取引を停止させていただいております | 余力制御.その他商品買付停止 |
| 11807 | 株式新規注文 | 追証で未入金があります | 余力制御.追証未入金あり |
| 11808 | 株式新規注文 | お客様の現引、現渡のお取引を停止させていただいております | 余力制御.現引現渡停止 |
| 11810 | 株式新規注文 | 二階建チェックファイルに問題があります | 二階建チェックファイル障害 |
| 11811 | 株式新規注文 | 二階建チェックでエラーが発生しました | 二階建チェックエラー |
| 11820 | 株式新規注文 | この銘柄には増担保規制が適用されております。増担保ファイルに問題があります。 | 増担保ファイル障害 |
| 11821 | 株式新規注文 | この銘柄には増担保規制が適用されております。規制銘柄新規建余力は、%s円です。ご注文に対して現金保証金が、%s円不足しています。 | 増担保現金チェックエラー |
| 11822 | 株式新規注文 | この銘柄には増担保規制が適用されております。規制銘柄新規建余力は、%s円です。ご注文に対して現金保証金が、%s円不足しています。 | 増担保現金チェックエラー |
| 11823 | 株式新規注文 | この銘柄には増担保規制が適用されております。ご注文に対して現金保証金が不足しています。 | 増担保現金チェックエラー |
| 11824 | 株式新規注文 | この銘柄には増担保規制が適用されております。規制銘柄新規建余力が不足しています。 | 増担保保証金チェックエラー |
| 11825 | 株式新規注文 | ご注文に対して現金保証金が不足しています。 | 増担保現金チェックエラー |
| 11826 | 株式新規注文 | 新規建余力は0円です。(最低保証金割れ) | 最低保証金割れエラー |
| 11830 | 株式新規注文 | 一極集中ファイル障害 | 一極集中ファイル障害 |
| 11831 | 株式新規注文 | 一極集中銘柄規制に抵触します。 | 一極集中チェックエラー |
| 11832 | 株式新規注文 | 保証金率チェックファイル障害 | 保証金率チェックファイル障害 |
| 11833 | 株式新規注文 | 当社運用規制の為、注文は受付られません | 保証金率チェックエラー |
| 11834 | 株式新規注文 | NISA注文抑止チェックファイル障害 | NISA買付注文抑止チェックファイル障害 |
| 11835 | 株式新規注文 | ＮＩＳＡロールオーバー期間の為、買付注文停止中です。 | NISA買付注文抑止チェックエラー |
| 11836 | 株式新規注文 | NISA注文抑止チェックファイル障害 | NISA売付注文抑止チェックファイル障害 |
| 11837 | 株式新規注文 | ＮＩＳＡロールオーバー期間の為、対象年のお預りがある銘柄の売付注文停止中です。 | NISA売付注文抑止チェックエラー |
| 11900 | 株式新規注文 | 現物買付可能額取得でエラーが発生しました | 現物買付可能額取得エラー |
| 11901 | 株式新規注文 | 差金決済売付可能数量取得でエラーが発生しました | 差金決済売付可能数量取得エラー |
| 11902 | 株式新規注文 | 信用新規建可能額取得でエラーが発生しました | 信用新規建可能額取得エラー |
| 11903 | 株式新規注文 | 現引可能額取得でエラーが発生しました | 現引可能額取得エラー |
| 11904 | 株式新規注文 | 日計り拘束金取得でエラーが発生しました | 日計り拘束金取得エラー |
| 11991 | 株式新規注文 | セッション情報レコードがありません | セッション情報レコードなし |
| 11992 | 株式新規注文 | セッション情報レコードファイルに問題が発生しました | セッション情報レコードファイル障害 |
| 11993 | 株式新規注文 | セッション情報レコード更新でエラーが発生しました | セッション情報レコード更新エラー |
| 11994 | 株式新規注文 | ボタンが２回以上押されたた可能性があります。注文状況照会をご確認下さい。 | 注文二重送信エラー |
| 11997 | 株式新規注文 | ネットでエラーが発生しました | ネットエラー |
| 11998 | 株式新規注文 | ＤＢ接続でエラーが発生しました | ＤＢエラー |
| 11999 | 株式新規注文 | サーバからの応答がありません。結果をご確認下さい。 | タイムアウト |
| 12001 | 株式訂正注文 | 注文番号に誤りがあります | 注文番号不正 |
| 12002 | 株式訂正注文 | 営業日に誤りがあります | 営業日不正 |
| 12003 | 株式訂正注文 | 市場に誤りがあります | 市場不正 |
| 12004 | 株式訂正注文 | 執行条件訂正フラグに誤りがあります | 執行条件訂正フラグ不正 |
| 12005 | 株式訂正注文 | 執行条件に誤りがあります | 執行条件不正 |
| 12006 | 株式訂正注文 | 注文値段訂正フラグに誤りがあります | 注文値段訂正フラグ不正 |
| 12007 | 株式訂正注文 | 注文値段区分に誤りがあります | 注文値段区分不正 |
| 12008 | 株式訂正注文 | 注文値段に誤りがあります | 注文値段不正 |
| 12009 | 株式訂正注文 | 注文数量訂正フラグに誤りがあります | 注文数量訂正フラグ不正 |
| 12010 | 株式訂正注文 | 注文数量に誤りがあります | 注文数量不正 |
| 12011 | 株式訂正注文 | 注文期日訂正フラグに誤りがあります | 注文期日訂正フラグ不正 |
| 12012 | 株式訂正注文 | 注文期日訂正フラグに誤りがあります | 注文期日訂正フラグ不正 |
| 12013 | 株式訂正注文 | 逆指値値段条件訂正フラグに誤りがあります | 逆指値値段条件訂正フラグ不正 |
| 12014 | 株式訂正注文 | 逆指値条件に誤りがあります | 逆指値条件不正 |
| 12015 | 株式訂正注文 | 逆指値値段区分訂正フラグに誤りがあります | 逆指値値段区分訂正フラグ不正 |
| 12016 | 株式訂正注文 | 逆指値値段区分に誤りがあります | 逆指値値段区分不正 |
| 12017 | 株式訂正注文 | 逆指値値段に誤りがあります | 逆指値値段不正 |
| 12018 | 株式訂正注文 | 接続チャネルに誤りがあります | チャネル不正 |
| 12019 | 株式訂正注文 | 接続チャネル詳細に誤りがあります | チャネル詳細不正 |
| 12020 | 株式訂正注文 | オペレータに誤りがあります | オペレータ不正 |
| 12021 | 株式訂正注文 | ＩＰアドレスに誤りがあります | ＩＰアドレス不正 |
| 12022 | 株式訂正注文 | 第二暗証番号省略フラグに誤りがあります | 第二暗証番号省略フラグ不正 |
| 12023 | 株式訂正注文 | 第二暗証番号に誤りがあります | 第二暗証番号不正 |
| 12024 | 株式訂正注文 | チェックのみフラグ不正 | チェックのみフラグ不正 |
| 12032 | 株式訂正注文 | 不成注文に成行が指定されています | 不成注文に成行が指定されています |
| 12033 | 株式訂正注文 | 期限付き注文に執行条件の訂正はできません。 | 期限付注文執行条件エラー |
| 12034 | 株式訂正注文 | トリガー前の逆指値注文の執行条件は訂正できません。 | 逆指値注文執行条件エラー |
| 12035 | 株式訂正注文 | トリガー前の通常＋逆指値注文の執行条件は訂正できません。 | 通常＋逆指値注文執行条件エラー |
| 12036 | 株式訂正注文 | 子注文に執行条件でエラーが発生しました | 子注文に執行条件エラー |
| 12037 | 株式訂正注文 | 子注文に注文期限でエラーが発生しました | 子注文に注文期限エラー |
| 12039 | 株式訂正注文 | 端株に指値は指定できません | 端株に指値は指定出来ない |
| 12040 | 株式訂正注文 | ｢成行｣を指定されていますが、注文単価も入力されています。指値の場合は｢指値｣に印を付けて下さい。 | 成行指値同時指定 |
| 12042 | 株式訂正注文 | 非課税口座執行条件で無条件以外のエラーが発生しました | 非課税口座執行条件エラー（指定なしのみ） |
| 12043 | 株式訂正注文 | 非課税口座値段区分でエラーが発生しました（指値のみ） | 非課税口座値段区分エラー（指値のみ） |
| 12044 | 株式訂正注文 | 非課税口座注文期限でエラーが発生しました（当日中のみ） | 非課税口座注文期限エラー（当日中のみ） |
| 12048 | 株式訂正注文 | 通常＋逆指値値段でエラーが発生しました | 通常＋逆指値値段エラー |
| 12050 | 株式訂正注文 | 通常注文の逆指値条件は訂正できません。 | 通常注文逆指値条件エラー |
| 12051 | 株式訂正注文 | 通常注文の逆指値注文値段は訂正できません。 | 通常注文逆指値注文値段エラー |
| 12052 | 株式訂正注文 | トリガー前の逆指値注文の注文値段は訂正できません。 | 逆指値注文値段エラー |
| 12053 | 株式訂正注文 | トリガー前の通常＋逆指値注文の注文値段は訂正できません。 | 通常＋逆指値注文値段エラー |
| 12110 | 株式訂正注文 | 執行条件変更がありません | 執行条件変更なし |
| 12111 | 株式訂正注文 | 注文数量変更がありません | 注文数量変更なし |
| 12112 | 株式訂正注文 | 注文値段変更がありません | 注文値段変更なし |
| 12113 | 株式訂正注文 | 注文期日変更がありません | 注文期日変更なし |
| 12114 | 株式訂正注文 | 逆指値条件変更がありません | 逆指値条件変更なし |
| 12115 | 株式訂正注文 | 逆指値注文値段変更なし | 逆指値注文値段変更なし |
| 12116 | 株式訂正注文 | 変更項目がありません | 変更項目なし |
| 12120 | 株式訂正注文 | 運用ステータス(注文)にデータがありません | 運用ステータス(注文)レコードなし |
| 12121 | 株式訂正注文 | 運用ステータス(採用値幅)にデータがありません | 運用ステータス(採用値幅)レコードなし |
| 12122 | 株式訂正注文 | 只今の時間帯は受付できません | 運用ステータス(注文).受付停止 |
| 12130 | 株式訂正注文 | 日付情報にデータがありません | 日付情報レコードなし |
| 12140 | 株式訂正注文 | 銘柄マスタにデータがありません | 銘柄マスタレコードなし |
| 12151 | 株式訂正注文 | 銘柄市場マスタにデータがありません | 銘柄市場マスタレコードなし |
| 12152 | 株式訂正注文 | 銘柄市場マスタ.前日終値がありません | 銘柄市場マスタ.前日終値なし |
| 12153 | 株式訂正注文 | 当該銘柄の売買単位の整数倍の数量を入力してください | 銘柄マスタ.売買単位エラー |
| 12154 | 株式訂正注文 | 当該銘柄の値幅制限内の単価を入力してください | 銘柄市場マスタ.値幅エラー |
| 12160 | 株式訂正注文 | 呼値にデータがありません | 呼値レコードなし |
| 12161 | 株式訂正注文 | 正しい呼値の単位で単価を入力してください | 呼値エラー |
| 12170 | 株式訂正注文 | 数量の増加はできません | 増株訂正エラー |
| 12180 | 株式訂正注文 | 注文期日でエラーが発生しました | 注文期日エラー |
| 12191 | 株式訂正注文 | 逆指値段には当該銘柄の値幅制限内の単価を入力してください | 逆指値段値幅エラー |
| 12192 | 株式訂正注文 | 逆指値段呼値にデータがありません | 逆指値段呼値レコードなし |
| 12193 | 株式訂正注文 | 正しい呼値の単位で逆指値段を入力してください | 逆指値段呼値エラー |
| 12194 | 株式訂正注文 | 執行単価が0以下です | 執行値段マイナスエラー |
| 12199 | 株式訂正注文 | サービス別取扱レコードがありません | サービス別取扱レコードなし |
| 12200 | 株式訂正注文 | このサービスは取り扱っておりません | サービス別取扱.現物訂正取扱不可 |
| 12201 | 株式訂正注文 | このサービスは取り扱っておりません | サービス別取扱.信用訂正取扱不可 |
| 12209 | 株式訂正注文 | 商品市場別設定レコードがありません | 商品市場別設定レコードなし |
| 12210 | 株式訂正注文 | 数量と値段の同時訂正はできません | 商品市場別設定.同時訂正不可 |
| 12211 | 株式訂正注文 | 寄付への訂正はできません | 商品市場別設定.執行条件寄付不可 |
| 12212 | 株式訂正注文 | 引けへの訂正はできません | 商品市場別設定.執行条件引け不可 |
| 12213 | 株式訂正注文 | 不成への訂正はできません | 商品市場別設定.執行条件不成不可 |
| 12214 | 株式訂正注文 | 連続注文の訂正はできません | 商品市場別設定.連続注文不可 |
| 12215 | 株式訂正注文 | 出来るまで注文の訂正はできません | 商品市場別設定.出来るまで注文不可 |
| 12220 | 株式訂正注文 | 当該銘柄はお取引できません | 銘柄別市場別規制.停止区分取引禁止 |
| 12221 | 株式訂正注文 | 当該銘柄の現物買付の成行注文はできません | 銘柄別市場別規制.現物買付成行禁止 |
| 12222 | 株式訂正注文 | 当該銘柄の現物売付の成行注文はできません | 銘柄別市場別規制.現物売付成行禁止 |
| 12223 | 株式訂正注文 | 当該銘柄の制度信用の新規買建の成行注文はできません | 銘柄別市場別規制.制度信用買建成行禁止 |
| 12224 | 株式訂正注文 | 当該銘柄の制度信用の新規売建の成行注文はできません | 銘柄別市場別規制.制度信用売建成行禁止 |
| 12225 | 株式訂正注文 | 当該銘柄の制度信用の買返済の成行注文はできません | 銘柄別市場別規制.制度信用買返済成行禁止 |
| 12226 | 株式訂正注文 | 当該銘柄の制度信用の売返済の成行注文はできません | 銘柄別市場別規制.制度信用売返済成行禁止 |
| 12227 | 株式訂正注文 | 当該銘柄の一般信用の新規買建の成行注文はできません | 銘柄別市場別規制.一般信用買建成行禁止 |
| 12228 | 株式訂正注文 | 当該銘柄の一般信用の新規売建の成行注文はできません | 銘柄別市場別規制.一般信用売建成行禁止 |
| 12229 | 株式訂正注文 | 当該銘柄の一般信用の買返済の成行注文はできません | 銘柄別市場別規制.一般信用買返済成行禁止 |
| 12299 | 株式訂正注文 | 顧客マスタファイルに問題があります | 顧客マスタファイル障害 |
| 12300 | 株式訂正注文 | 顧客マスタにデータがありません | 顧客マスタレコードなし |
| 12301 | 株式訂正注文 | 顧客マスタ.精算理由でエラーが発生しました | 顧客マスタ.精算理由エラー |
| 12302 | 株式訂正注文 | 顧客情報ファイルに問題があります | 顧客情報ファイル障害 |
| 12303 | 株式訂正注文 | 顧客情報にデータがありません | 顧客情報レコードなし |
| 12304 | 株式訂正注文 | 第二暗証番号が誤っています | 顧客マスタ.第二パスワード不一致 |
| 12305 | 株式訂正注文 | 口座管理ファイルに問題があります | 口座管理ファイル障害 |
| 12306 | 株式訂正注文 | 口座管理にデータがありません | 口座管理レコードなし |
| 12307 | 株式訂正注文 | 口座管理.特定口座が未開設です | 口座管理.特定口座未開設 |
| 12308 | 株式訂正注文 | 口座管理.非課税口座が未開設です | 口座管理.非課税口座未開設 |
| 12309 | 株式訂正注文 | 口座管理.信用口座が未開設です | 口座管理.信用口座未開設 |
| 12311 | 株式訂正注文 | ロック顧客ファイルに問題があります | ロック顧客ファイル障害 |
| 12312 | 株式訂正注文 | 現在、お客様の口座には、お取引制限がかかっています。コールセンターまでお問い合わせ下さい。 | ロック顧客該当エラー |
| 12313 | 株式訂正注文 | インサイダファイルに問題があります | インサイダファイル障害 |
| 12314 | 株式訂正注文 | 当該注文はインサイダー情報に基づく注文ではない同意が無い為受付できません | インサイダチェックエラー |
| 12315 | 株式訂正注文 | 特定投資家契約マスタファイルに問題があります | 特定投資家契約マスタファイル障害 |
| 12316 | 株式訂正注文 | 特定投資家契約マスタチェックでエラーが発生しました | 特定投資家契約マスタチェックエラー |
| 12317 | 株式訂正注文 | 金商法交付書面ファイルに問題があります | 金商法交付書面ファイル障害 |
| 12318 | 株式訂正注文 | 金商法交付書面(当日分)ファイルに問題があります | 金商法交付書面(当日更新分)ファイル障害 |
| 12319 | 株式訂正注文 | 金商法交付書面チェックでエラーが発生しました | 金商法交付書面チェックエラー |
| 12320 | 株式訂正注文 | 顧客銘柄別取引停止ファイルに問題があります | 顧客銘柄別取引停止ファイル障害 |
| 12321 | 株式訂正注文 | 顧客銘柄別取引停止にデータがありません | 顧客銘柄別取引停止レコードなし |
| 12322 | 株式訂正注文 | お客様の当該銘柄における現物買付のお取引を停止させていただいております | 顧客銘柄別取引停止.現物買付停止 |
| 12323 | 株式訂正注文 | お客様の当該銘柄における現物売付のお取引を停止させていただいております | 顧客銘柄別取引停止.現物売付停止 |
| 12324 | 株式訂正注文 | お客様の当該銘柄における信用新規買建のお取引を停止させていただいております | 顧客銘柄別取引停止.信用新規買建停止 |
| 12325 | 株式訂正注文 | お客様の当該銘柄における信用新規売建のお取引を停止させていただいております | 顧客銘柄別取引停止.信用新規売建停止 |
| 12326 | 株式訂正注文 | お客様の当該銘柄における信用買返済のお取引を停止させていただいております | 顧客銘柄別取引停止.信用買返済停止 |
| 12327 | 株式訂正注文 | お客様の当該銘柄における信用売返済のお取引を停止させていただいております | 顧客銘柄別取引停止.信用売返済停止 |
| 12328 | 株式訂正注文 | お客様の当該銘柄における信用現引のお取引を停止させていただいております | 顧客銘柄別取引停止.信用現引停止 |
| 12329 | 株式訂正注文 | お客様の当該銘柄における信用現渡のお取引を停止させていただいております | 顧客銘柄別取引停止.信用現渡停止 |
| 12340 | 株式訂正注文 | 市場別特殊執行注文取扱停止ファイルに問題があります | 市場別特殊執行注文取扱停止ファイル障害 |
| 12341 | 株式訂正注文 | 当該市場での逆指値注文の受付を停止しています | 市場別特殊執行注文取扱停止.逆指値停止 |
| 12342 | 株式訂正注文 | 当該市場での通常＋逆指値注文の受付を停止しています | 市場別特殊執行注文取扱停止.通常＋逆指値停止 |
| 12343 | 株式訂正注文 | 商品別特殊執行注文取扱停止ファイルに問題があります | 商品別特殊執行注文取扱停止ファイル障害 |
| 12344 | 株式訂正注文 | 逆指値注文の受付を停止しています | 商品別特殊執行注文取扱停止.逆指値停止 |
| 12345 | 株式訂正注文 | 通常＋逆指値注文の受付を停止しています | 商品別特殊執行注文取扱停止.通常＋逆指値停止 |
| 12346 | 株式訂正注文 | 銘柄別特殊執行注文取扱停止ファイルに問題があります | 銘柄別特殊執行注文取扱停止ファイル障害 |
| 12347 | 株式訂正注文 | 当該銘柄は逆指値注文はできません | 銘柄別特殊執行注文取扱停止.逆指値停止 |
| 12348 | 株式訂正注文 | 当該銘柄は通常＋逆指値はできません | 銘柄別特殊執行注文取扱停止.通常＋逆指値停止 |
| 12349 | 株式訂正注文 | 当該銘柄は特殊執行注文はできません | 銘柄別特殊執行注文取扱停止.特殊執行注文不可 |
| 12350 | 株式訂正注文 | ハードリミット市場別ファイルに問題があります | ハードリミット市場別ファイル障害 |
| 12351 | 株式訂正注文 | ハードリミット市場別にデータがありません | ハードリミット市場別レコードなし |
| 12352 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合1 |
| 12353 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合2 |
| 12354 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合3 |
| 12355 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合4 |
| 12356 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合5 |
| 12360 | 株式訂正注文 | ハードリミットファイルに問題があります | ハードリミットファイル障害 |
| 12361 | 株式訂正注文 | ハードリミットにデータがありません | ハードリミットレコードなし |
| 12362 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量買上限 |
| 12363 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量売上限 |
| 12364 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量買新規上限 |
| 12365 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量売新規上限 |
| 12366 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量買返済上限 |
| 12367 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量売返済上限 |
| 12368 | 株式訂正注文 | 発注金額が弊社規定の制限を越えています | ハードリミット.発注金額上限 |
| 12369 | 株式訂正注文 | 建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ハードリミット.建玉総数量上限 |
| 12370 | 株式訂正注文 | 売建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ハードリミット.売建玉総数量上限 |
| 12371 | 株式訂正注文 | 建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ハードリミット.建玉総金額上限 |
| 12372 | 株式訂正注文 | 当該銘柄の建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ハードリミット.建玉銘柄総金額上限 |
| 12380 | 株式訂正注文 | 注文できません。個別ファイルに問題があります | ソフトリミット個別ファイル障害 |
| 12381 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量買上限 |
| 12382 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量売上限 |
| 12383 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量買新規上限 |
| 12384 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量売新規上限 |
| 12385 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量買返済上限 |
| 12386 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量売返済上限 |
| 12387 | 株式訂正注文 | 発注金額が弊社規定の制限を越えています | ソフトリミット個別.発注金額上限 |
| 12388 | 株式訂正注文 | 建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット個別.建玉総数量上限 |
| 12389 | 株式訂正注文 | 売建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット個別.売建玉総数量上限 |
| 12390 | 株式訂正注文 | 建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット個別.建玉総金額上限 |
| 12391 | 株式訂正注文 | 当該銘柄の建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット個別.建玉銘柄総金額上限 |
| 12400 | 株式訂正注文 | 注文できません。通常ファイルに問題があります | ソフトリミット通常ファイル障害 |
| 12401 | 株式訂正注文 | 注文できません。通常にデータがありません | ソフトリミット通常レコードなし |
| 12402 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量買上限 |
| 12403 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量売上限 |
| 12404 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量買新規上限 |
| 12405 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量売新規上限 |
| 12406 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量買返済上限 |
| 12407 | 株式訂正注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量売返済上限 |
| 12408 | 株式訂正注文 | 発注金額が弊社規定の制限を越えています | ソフトリミット通常.発注金額上限 |
| 12409 | 株式訂正注文 | 建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット通常.建玉総数量上限 |
| 12410 | 株式訂正注文 | 売建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット通常.売建玉総数量上限 |
| 12411 | 株式訂正注文 | 建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット通常.建玉総金額上限 |
| 12412 | 株式訂正注文 | 当該銘柄の建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット通常.建玉銘柄総金額上限 |
| 12415 | 株式訂正注文 | 空売り注文はできません | 空売り規制(注文不可) |
| 12416 | 株式訂正注文 | 空売り注文は成行に訂正できません | 空売り規制(成行不可) |
| 12417 | 株式訂正注文 | 空売り期限付き注文はできません | 空売り規制(期限付き不可) |
| 12420 | 株式訂正注文 | 買付可能額が不足しています | 買付可能額不足 |
| 12421 | 株式訂正注文 | 売付可能株数が不足しているため、この後注文はお受けできません | 売付可能数量不足 |
| 12422 | 株式訂正注文 | 新規建余力は、%s 円です。ご注文に対して現金保証金が、%s 円不足しています。 | 信用新規建可能額不足 |
| 12423 | 株式訂正注文 | 現引可能額が不足しています | 現引可能額不足 |
| 12426 | 株式訂正注文 | この銘柄は日計りの対象となっているため、この訂正により、お預かり金が不足します。 | 売付可能数量不足 |
| 12428 | 株式訂正注文 | 増担保の現金必要保証金が不足します。 | 現金必要保証金不足 |
| 12429 | 株式訂正注文 | 信用新規建可能額が不足しています | 信用新規建可能額不足 |
| 12430 | 株式訂正注文 | 現在、この銘柄の買付可能額は、%s 円です。%s 円不足しているため、このご注文はお受けできません。(日計り取引銘柄等、銘柄によっては可能額の算出方法が異なるケースがあります。) | 買付可能額不足 |
| 12440 | 株式訂正注文 | 非課税口座管理ファイルに問題があります | 非課税口座管理ファイル障害 |
| 12441 | 株式訂正注文 | 非課税口座管理にデータがありません | 非課税口座管理レコードなし |
| 12442 | 株式訂正注文 | 非課税口座管理更新でエラーが発生しました | 非課税口座管理更新エラー |
| 12443 | 株式訂正注文 | 非課税口座可能額が不足しています | 非課税口座可能額不足 |
| 12450 | 株式訂正注文 | 顧客手数料ファイルに問題があります | 顧客手数料ファイル障害 |
| 12451 | 株式訂正注文 | 顧客手数料レコードなし | 顧客手数料レコードなし |
| 12460 | 株式訂正注文 | 手数料マスタファイルに問題があります | 手数料マスタファイル障害 |
| 12461 | 株式訂正注文 | 手数料マスタレコードなし | 手数料マスタレコードなし |
| 12470 | 株式訂正注文 | 課税率マスタファイルに問題があります | 課税率マスタファイル障害 |
| 12471 | 株式訂正注文 | 課税率マスタレコードなし | 課税率マスタレコードなし |
| 12480 | 株式訂正注文 | 保管顧客課税別ファイルに問題があります | 保管顧客課税別ファイル障害 |
| 12481 | 株式訂正注文 | 選択した口座区分がお預かり銘柄と不一致のため、このご注文はお受けできません。 | 保管顧客課税別にデータがありません |
| 12482 | 株式訂正注文 | 売付可能な株数が不足しているため、このご注文はお受けできません。売却可能株数や注文一覧画面をご確認ください。 | 保管顧客課税別有効数量がありません |
| 12483 | 株式訂正注文 | 保管顧客課税別にデータ作成でエラーが発生しました | 保管顧客課税別レコード作成エラー |
| 12484 | 株式訂正注文 | 保管顧客課税別のデータ更新でエラーが発生しました | 保管顧客課税別レコード更新エラー |
| 12490 | 株式訂正注文 | 信用建玉明細ファイルに問題があります | 信用建玉明細ファイル障害 |
| 12491 | 株式訂正注文 | 信用建玉明細にデータがありません | 信用建玉明細レコードなし |
| 12492 | 株式訂正注文 | 信用建玉明細有効数量がありません | 信用建玉明細有効数量なし |
| 12493 | 株式訂正注文 | 信用建玉明細のデータ更新でエラーが発生しました | 信用建玉明細レコード更新エラー |
| 12494 | 株式訂正注文 | 信用建玉残ファイルに問題があります | 信用建玉残ファイル障害 |
| 12495 | 株式訂正注文 | 信用建玉残にデータがありません | 信用建玉残レコードなし |
| 12496 | 株式訂正注文 | 信用建玉残にデータ作成でエラーが発生しました | 信用建玉残レコード作成エラー |
| 12497 | 株式訂正注文 | 信用建玉残のデータ更新でエラーが発生しました | 信用建玉残レコード更新エラー |
| 12500 | 株式訂正注文 | 顧客金銭ファイルに問題があります | 顧客金銭ファイル障害 |
| 12501 | 株式訂正注文 | 顧客金銭にデータがありません | 顧客金銭レコードなし |
| 12502 | 株式訂正注文 | 顧客拘束金ファイルに問題があります | 顧客拘束金ファイル障害 |
| 12503 | 株式訂正注文 | 顧客拘束金にデータがありません | 顧客拘束金レコードなし |
| 12504 | 株式訂正注文 | 顧客拘束金レコードレコード作成エラー | 顧客拘束金レコード作成エラー |
| 12505 | 株式訂正注文 | 顧客拘束金レコード更新エラー | 顧客拘束金レコード更新エラー |
| 12509 | 株式訂正注文 | 保証金率取得でエラーが発生しました | 保証金率取得エラー |
| 12510 | 株式訂正注文 | 代用掛目ファイルに問題があります | 代用掛目ファイル障害 |
| 12511 | 株式訂正注文 | 日付情報ファイルに問題があります | 日付情報ファイル障害 |
| 12512 | 株式訂正注文 | 保管顧客別残ファイルに問題があります | 保管顧客別残ファイル障害 |
| 12513 | 株式訂正注文 | 保管顧客別残にデータがありません | 保管顧客別残レコードなし |
| 12514 | 株式訂正注文 | 保管顧客別残にデータ作成でエラーが発生しました | 保管顧客別残レコード作成エラー |
| 12515 | 株式訂正注文 | 保管顧客別残のデータ更新でエラーが発生しました | 保管顧客別残レコード更新エラー |
| 12516 | 株式訂正注文 | 保証金推移ファイルに問題があります | 保証金推移ファイル障害 |
| 12517 | 株式訂正注文 | 保証金推移にデータがありません | 保証金推移レコードなし |
| 12518 | 株式訂正注文 | 保証金推移にデータ作成でエラーが発生しました | 保証金推移レコード作成エラー |
| 12519 | 株式訂正注文 | 保証金推移のデータ更新でエラーが発生しました | 保証金推移レコード更新エラー |
| 12520 | 株式訂正注文 | 顧客当日取引情報ファイルに問題があります | 顧客当日取引情報ファイル障害 |
| 12521 | 株式訂正注文 | 顧客当日取引情報にデータがありません | 顧客当日取引情報レコードなし |
| 12522 | 株式訂正注文 | 顧客当日取引情報にデータ作成でエラーが発生しました | 顧客当日取引情報レコード作成エラー |
| 12523 | 株式訂正注文 | 顧客当日取引情報のデータ更新でエラーが発生しました | 顧客当日取引情報レコード更新エラー |
| 12524 | 株式訂正注文 | 差金決済管理明細ファイルに問題があります | 差金決済管理明細ファイル障害 |
| 12525 | 株式訂正注文 | 差金決済管理明細にデータがありません | 差金決済管理明細レコードなし |
| 12526 | 株式訂正注文 | 差金決済管理明細にデータ作成でエラーが発生しました | 差金決済管理明細レコード作成エラー |
| 12527 | 株式訂正注文 | 差金決済管理明細のデータ更新でエラーが発生しました | 差金決済管理明細レコード更新エラー |
| 12600 | 株式訂正注文 | 注文番号（株式）ファイルに問題があります | 注文番号（株式）ファイル障害 |
| 12601 | 株式訂正注文 | 注文番号（株式）にデータがありません | 注文番号（株式）レコードなし |
| 12602 | 株式訂正注文 | 建玉番号（株式）ファイルに問題があります | 建玉番号（株式）ファイル障害 |
| 12603 | 株式訂正注文 | 建玉番号（株式）にデータがありません | 建玉番号（株式）レコードなし |
| 12604 | 株式訂正注文 | 親注文株式サマリファイルに問題があります | 親注文株式サマリファイル障害 |
| 12605 | 株式訂正注文 | 親注文株式サマリにデータがありません | 親注文株式サマリレコードなし |
| 12606 | 株式訂正注文 | 親注文株式サマリ有効数量がありません | 親注文株式サマリ有効数量なし |
| 12607 | 株式訂正注文 | 親注文株式サマリ数量を超えています | 親注文株式サマリ数量オーバー |
| 12608 | 株式訂正注文 | 親注文株式サマリレコード更新エラー | 親注文株式サマリレコード更新エラー |
| 12609 | 株式訂正注文 | 株式サマリファイルに問題があります | 株式サマリファイル障害 |
| 12610 | 株式訂正注文 | 株式サマリにデータがありません | 株式サマリレコードなし |
| 12611 | 株式訂正注文 | 株式サマリにデータ作成でエラーが発生しました | 株式サマリレコード作成エラー |
| 12612 | 株式訂正注文 | 株式サマリのデータ更新でエラーが発生しました | 株式サマリレコード更新エラー |
| 12613 | 株式訂正注文 | 株式明細ファイルに問題があります | 株式明細ファイル障害 |
| 12614 | 株式訂正注文 | 株式明細にデータがありません | 株式明細レコードなし |
| 12615 | 株式訂正注文 | 株式明細更新でエラーが発生しました | 株式明細更新エラー |
| 12616 | 株式訂正注文 | 株式明細にデータ作成でエラーが発生しました | 株式明細レコード作成エラー |
| 12617 | 株式訂正注文 | 株式注文約定履歴ファイルに問題があります | 株式注文約定履歴ファイル障害 |
| 12618 | 株式訂正注文 | 株式注文約定履歴作成でエラーが発生しました | 株式注文約定履歴作成エラー |
| 12621 | 株式訂正注文 | 株式返済予約ファイルに問題があります | 株式返済予約ファイル障害 |
| 12622 | 株式訂正注文 | 株式返済予約にデータがありません | 株式返済予約レコードなし |
| 12623 | 株式訂正注文 | 株式返済予約にデータ作成でエラーが発生しました | 株式返済予約レコード作成エラー |
| 12624 | 株式訂正注文 | 株式返済予約のデータ更新でエラーが発生しました | 株式返済予約レコード更新エラー |
| 12625 | 株式訂正注文 | 株式返済明細ファイルに問題があります | 株式返済明細ファイル障害 |
| 12626 | 株式訂正注文 | 株式返済明細にデータがありません | 株式返済明細レコードなし |
| 12627 | 株式訂正注文 | 株式返済明細にデータ作成でエラーが発生しました | 株式返済明細レコード作成エラー |
| 12628 | 株式訂正注文 | 株式返済明細のデータ更新でエラーが発生しました | 株式返済明細レコード更新エラー |
| 12640 | 株式訂正注文 | 株式約定失効ファイルに問題があります | 株式約定失効ファイル障害 |
| 12641 | 株式訂正注文 | 株式約定失効にデータがありません | 株式約定失効レコードなし |
| 12642 | 株式訂正注文 | 株式約定失効にデータ作成でエラーが発生しました | 株式約定失効レコード作成エラー |
| 12643 | 株式訂正注文 | 株式約定失効のデータ更新でエラーが発生しました | 株式約定失効レコード更新エラー |
| 12645 | 株式訂正注文 | システム別設定ファイルに問題があります | システム別設定ファイル障害 |
| 12646 | 株式訂正注文 | 銘柄マスタ（株式）ファイルに問題があります | 銘柄マスタ（株式）ファイル障害 |
| 12647 | 株式訂正注文 | 銘柄市場マスタ（株式）ファイルに問題があります | 銘柄市場マスタ（株式）ファイル障害 |
| 12700 | 株式訂正注文 | 運用ステータス(申告)にデータがありません | 運用ステータス(申告)レコードなし |
| 12701 | 株式訂正注文 | 只今の時間帯は受付できません | 運用ステータス(申告).受付停止 |
| 12702 | 株式訂正注文 | 運用ステータス(連続注文)にデータがありません | 運用ステータス(連続注文)レコードなし |
| 12703 | 株式訂正注文 | 只今の時間帯は受付できません | 運用ステータス(連続注文).受付停止 |
| 12800 | 株式訂正注文 | 余力制御ファイルに問題があります | 余力制御ファイル障害 |
| 12802 | 株式訂正注文 | 余力制御.取引制限のため受付を停止しています | 余力制御.取引停止 |
| 12803 | 株式訂正注文 | 余力制御.取引制限のため信用新規建の受付を停止しています | 余力制御.信用新規建停止 |
| 12806 | 株式訂正注文 | 余力制御.取引制限のためその他商品買付の受付を停止しています | 余力制御.その他商品買付停止 |
| 12807 | 株式訂正注文 | 追証の未入金があるため受付できません | 余力制御.追証未入金あり |
| 12810 | 株式訂正注文 | 二階建チェックファイルに問題があります | 二階建チェックファイル障害 |
| 12811 | 株式訂正注文 | 二階建チェックでエラーが発生しました | 二階建チェックエラー |
| 12820 | 株式訂正注文 | 増担保ファイルに問題があります | 増担保ファイル障害 |
| 12821 | 株式訂正注文 | 規制銘柄新規建余力は、%s円です。ご注文に対して現金保証金が、%s円不足しています。 | 増担保現金チェックエラー |
| 12822 | 株式訂正注文 | 規制銘柄新規建余力は、%s円です。ご注文に対して現金保証金が、%s円不足しています。 | 増担保現金チェックエラー |
| 12823 | 株式訂正注文 | ご注文に対して現金保証金が不足しています。 | 増担保現金チェックエラー |
| 12824 | 株式訂正注文 | 規制銘柄新規建余力が不足しています。 | 増担保保証金チェックエラー |
| 12826 | 株式訂正注文 | 新規建余力は0円です。(最低保証金割れ) | 最低保証金割れエラー |
| 12830 | 株式訂正注文 | 一極集中ファイル障害 | 一極集中ファイル障害 |
| 12831 | 株式訂正注文 | 一極集中銘柄規制に抵触します。 | 一極集中チェックエラー |
| 12832 | 株式訂正注文 | 保証金率ファイル障害 | 保証金率ファイル障害 |
| 12833 | 株式訂正注文 | 当社運用規制の為、注文は受付られません | 保証金率チェックエラー |
| 12900 | 株式訂正注文 | 現物買付可能額取得でエラーが発生しました | 現物買付可能額取得エラー |
| 12901 | 株式訂正注文 | 差金決済売付可能数量取得でエラーが発生しました | 差金決済売付可能数量取得エラー |
| 12902 | 株式訂正注文 | 信用新規建可能額取得でエラーが発生しました | 信用新規建可能額取得エラー |
| 12903 | 株式訂正注文 | 現引可能額取得でエラーが発生しました | 現引可能額取得エラー |
| 12991 | 株式訂正注文 | セッション情報レコードがありません | セッション情報レコードなし |
| 12992 | 株式訂正注文 | セッション情報レコードファイルに問題が発生しました | セッション情報レコードファイル障害 |
| 12993 | 株式訂正注文 | セッション情報レコード更新でエラーが発生しました | セッション情報レコード更新エラー |
| 12994 | 株式訂正注文 | ボタンが２回以上押されたた可能性があります。注文状況照会をご確認下さい。 | 注文二重送信エラー |
| 12997 | 株式訂正注文 | ネットでエラーが発生しました | ネットエラー |
| 12998 | 株式訂正注文 | ＤＢ接続でエラーが発生しました | ＤＢエラー |
| 12999 | 株式訂正注文 | サーバからの応答がありません。結果をご確認下さい。 | タイムアウト |
| 13001 | 株式取消注文 | 注文番号に誤りがあります | 注文番号不正 |
| 13002 | 株式取消注文 | 営業日に誤りがあります | 営業日不正 |
| 13018 | 株式取消注文 | 接続に誤りがあります | チャネル不正 |
| 13019 | 株式取消注文 | 接続詳細に誤りがあります | チャネル詳細不正 |
| 13020 | 株式取消注文 | オペレータに誤りがあります | オペレータ不正 |
| 13021 | 株式取消注文 | ＩＰアドレスに誤りがあります | ＩＰアドレス不正 |
| 13022 | 株式取消注文 | 第二暗証番号省略フラグに誤りがあります | 第二暗証番号省略フラグ不正 |
| 13023 | 株式取消注文 | 第二暗証番号に誤りがあります | 第二暗証番号不正 |
| 13100 | 株式取消注文 | 株式サマリにデータがありません | 株式サマリレコードなし |
| 13120 | 株式取消注文 | 運用ステータス(注文)にデータがありません | 運用ステータス(注文)レコードなし |
| 13122 | 株式取消注文 | 只今の時間帯は受付できません | 運用ステータス(注文).受付停止 |
| 13130 | 株式取消注文 | 日付情報にデータがありません | 日付情報レコードなし |
| 13199 | 株式取消注文 | サービス別取扱レコードがありません | サービス別取扱レコードなし |
| 13200 | 株式取消注文 | このサービスは取り扱っておりません | サービス別取扱.現物取消取扱不可 |
| 13201 | 株式取消注文 | このサービスは取り扱っておりません | サービス別取扱.信用取消取扱不可 |
| 13245 | 株式取消注文 | システム状態にデータがありません | システム状態レコードなし |
| 13246 | 株式取消注文 | システムが受付可能時間外です。 | システム状態.ログイン不許可 |
| 13247 | 株式取消注文 | システムが受付可能時間外です。 | システム状態.閉局 |
| 13290 | 株式取消注文 | 注文処理中です。 | 注文処理中 |
| 13291 | 株式取消注文 | 訂正中です | 訂正中 |
| 13292 | 株式取消注文 | 取消中です | 取消中 |
| 13293 | 株式取消注文 | 失効済みです | 失効済み |
| 13294 | 株式取消注文 | 約定済みです | 約定済み |
| 13295 | 株式取消注文 | 取消済みです | 取消済み |
| 13296 | 株式取消注文 | トリガー発動済みです | トリガー発動済み |
| 13297 | 株式取消注文 | 強制決済中ですので、この操作はできません | 強制決済中 |
| 13299 | 株式取消注文 | 顧客マスタファイルに問題があります | 顧客マスタファイル障害 |
| 13300 | 株式取消注文 | 顧客マスタにデータがありません | 顧客マスタレコードなし |
| 13301 | 株式取消注文 | 顧客マスタ.精算理由でエラーが発生しました | 顧客マスタ.精算理由エラー |
| 13302 | 株式取消注文 | 顧客情報ファイルに問題があります | 顧客情報ファイル障害 |
| 13303 | 株式取消注文 | 顧客情報にデータがありません | 顧客情報レコードなし |
| 13304 | 株式取消注文 | 第二暗証番号が誤っています | 顧客マスタ.第二パスワード不一致 |
| 13305 | 株式取消注文 | 口座管理ファイルに問題があります | 口座管理ファイル障害 |
| 13306 | 株式取消注文 | 口座管理にデータがありません | 口座管理レコードなし |
| 13307 | 株式取消注文 | 口座管理.特定口座が未開設です | 口座管理.特定口座未開設 |
| 13308 | 株式取消注文 | 口座管理.非課税口座が未開設です | 口座管理.非課税口座未開設 |
| 13309 | 株式取消注文 | 口座管理.信用口座が未開設です | 口座管理.信用口座未開設 |
| 13311 | 株式取消注文 | ロック顧客ファイルに問題があります | ロック顧客ファイル障害 |
| 13312 | 株式取消注文 | 現在、お客様の口座には、お取引制限がかかっています。コールセンターまでお問い合わせ下さい。 | ロック顧客該当エラー |
| 13313 | 株式取消注文 | インサイダファイルに問題があります | インサイダファイル障害 |
| 13314 | 株式取消注文 | 当該注文はインサイダー情報に基づく注文ではない同意が無い為受付できません | インサイダチェックエラー |
| 13315 | 株式取消注文 | 特定投資家契約マスタファイルに問題があります | 特定投資家契約マスタファイル障害 |
| 13316 | 株式取消注文 | 特定投資家契約マスタチェックでエラーが発生しました | 特定投資家契約マスタチェックエラー |
| 13317 | 株式取消注文 | 金商法交付書面ファイルに問題があります | 金商法交付書面ファイル障害 |
| 13318 | 株式取消注文 | 金商法交付書面(当日分)ファイルに問題があります | 金商法交付書面(当日分)ファイル障害 |
| 13319 | 株式取消注文 | 金商法交付書面チェックでエラーが発生しました | 金商法交付書面チェックエラー |
| 13320 | 株式取消注文 | 顧客銘柄別取引停止ファイルに問題があります | 顧客銘柄別取引停止ファイル障害 |
| 13321 | 株式取消注文 | 顧客銘柄別取引停止にデータがありません | 顧客銘柄別取引停止レコードなし |
| 13322 | 株式取消注文 | お客様の当該銘柄における現物買付のお取引を停止させていただいております | 顧客銘柄別取引停止.現物買付停止 |
| 13323 | 株式取消注文 | お客様の当該銘柄における現物売付のお取引を停止させていただいております | 顧客銘柄別取引停止.現物売付停止 |
| 13324 | 株式取消注文 | お客様の当該銘柄における信用新規買建のお取引を停止させていただいております | 顧客銘柄別取引停止.信用新規買建停止 |
| 13325 | 株式取消注文 | お客様の当該銘柄における信用新規売建のお取引を停止させていただいております | 顧客銘柄別取引停止.信用新規売建停止 |
| 13326 | 株式取消注文 | お客様の当該銘柄における信用買返済のお取引を停止させていただいております | 顧客銘柄別取引停止.信用買返済停止 |
| 13327 | 株式取消注文 | お客様の当該銘柄における信用売返済のお取引を停止させていただいております | 顧客銘柄別取引停止.信用売返済停止 |
| 13328 | 株式取消注文 | お客様の当該銘柄における信用現引のお取引を停止させていただいております | 顧客銘柄別取引停止.信用現引停止 |
| 13329 | 株式取消注文 | お客様の当該銘柄における信用現渡のお取引を停止させていただいております | 顧客銘柄別取引停止.信用現渡停止 |
| 13340 | 株式取消注文 | 市場別特殊執行注文取扱停止ファイルに問題があります | 市場別特殊執行注文取扱停止ファイル障害 |
| 13341 | 株式取消注文 | 当該市場での逆指値注文の受付を停止しています | 市場別特殊執行注文取扱停止.逆指値停止 |
| 13342 | 株式取消注文 | 当該市場での通常＋逆指値注文の受付を停止しています | 市場別特殊執行注文取扱停止.通常＋逆指値停止 |
| 13343 | 株式取消注文 | 商品別特殊執行注文取扱停止ファイルに問題があります | 商品別特殊執行注文取扱停止ファイル障害 |
| 13344 | 株式取消注文 | 逆指値注文の受付を停止しています | 商品別特殊執行注文取扱停止.逆指値停止 |
| 13345 | 株式取消注文 | 通常＋逆指値注文の受付を停止しています | 商品別特殊執行注文取扱停止.通常＋逆指値停止 |
| 13346 | 株式取消注文 | 銘柄別特殊執行注文取扱停止ファイルに問題があります | 銘柄別特殊執行注文取扱停止ファイル障害 |
| 13347 | 株式取消注文 | 当該銘柄は逆指値注文はできません | 銘柄別特殊執行注文取扱停止.逆指値停止 |
| 13348 | 株式取消注文 | 当該銘柄は通常＋逆指値はできません | 銘柄別特殊執行注文取扱停止.通常＋逆指値停止 |
| 13349 | 株式取消注文 | 当該銘柄は特殊執行注文はできません | 銘柄別特殊執行注文取扱停止.特殊執行注文不可 |
| 13350 | 株式取消注文 | ハードリミット市場別ファイルに問題があります | ハードリミット市場別ファイル障害 |
| 13351 | 株式取消注文 | ハードリミット市場別にデータがありません | ハードリミット市場別レコードなし |
| 13352 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合1 |
| 13353 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合2 |
| 13354 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合3 |
| 13355 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合4 |
| 13356 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ハードリミット市場別.発注割合5 |
| 13360 | 株式取消注文 | ハードリミットファイルに問題があります | ハードリミットファイル障害 |
| 13361 | 株式取消注文 | ハードリミットにデータがありません | ハードリミットレコードなし |
| 13362 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量買上限 |
| 13363 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量売上限 |
| 13364 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量買新規上限 |
| 13365 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量売新規上限 |
| 13366 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量買返済上限 |
| 13367 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ハードリミット.発注数量売返済上限 |
| 13368 | 株式取消注文 | 発注金額が弊社規定の制限を越えています | ハードリミット.発注金額上限 |
| 13369 | 株式取消注文 | 建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ハードリミット.建玉総数量上限 |
| 13370 | 株式取消注文 | 売建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ハードリミット.売建玉総数量上限 |
| 13371 | 株式取消注文 | 建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ハードリミット.建玉総金額上限 |
| 13372 | 株式取消注文 | 当該銘柄の建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ハードリミット.建玉銘柄総金額上限 |
| 13380 | 株式取消注文 | 注文できません。個別ファイルに問題があります | ソフトリミット個別ファイル障害 |
| 13381 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量買上限 |
| 13382 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量売上限 |
| 13383 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量買新規上限 |
| 13384 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量売新規上限 |
| 13385 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量買返済上限 |
| 13386 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット個別.発注数量売返済上限 |
| 13387 | 株式取消注文 | 発注金額が弊社規定の制限を越えています | ソフトリミット個別.発注金額上限 |
| 13388 | 株式取消注文 | 建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット個別.建玉総数量上限 |
| 13389 | 株式取消注文 | 売建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット個別.売建玉総数量上限 |
| 13390 | 株式取消注文 | 建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット個別.建玉総金額上限 |
| 13391 | 株式取消注文 | 当該銘柄の建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット個別.建玉銘柄総金額上限 |
| 13400 | 株式取消注文 | 注文できません。通常ファイルに問題があります | ソフトリミット通常ファイル障害 |
| 13401 | 株式取消注文 | 注文できません。通常にデータがありません | ソフトリミット通常レコードなし |
| 13402 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量買上限 |
| 13403 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量売上限 |
| 13404 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量買新規上限 |
| 13405 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量売新規上限 |
| 13406 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量買返済上限 |
| 13407 | 株式取消注文 | 発注数量が弊社規定の制限を越えています | ソフトリミット通常.発注数量売返済上限 |
| 13408 | 株式取消注文 | 発注金額が弊社規定の制限を越えています | ソフトリミット通常.発注金額上限 |
| 13409 | 株式取消注文 | 建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット通常.建玉総数量上限 |
| 13410 | 株式取消注文 | 売建玉の総数量(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット通常.売建玉総数量上限 |
| 13411 | 株式取消注文 | 建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット通常.建玉総金額上限 |
| 13412 | 株式取消注文 | 当該銘柄の建玉の総金額(既存分＋今回ご注文分)が弊社規定の制限を越えています | ソフトリミット通常.建玉銘柄総金額上限 |
| 13415 | 株式取消注文 | 空売り注文はできません | 空売り規制(注文不可) |
| 13416 | 株式取消注文 | 空売り成行注文はできません | 空売り規制(成行不可) |
| 13417 | 株式取消注文 | 空売り期限付き注文はできません | 空売り規制(期限付き不可) |
| 13420 | 株式取消注文 | 買付可能額が不足しています | 買付可能額不足 |
| 13421 | 株式取消注文 | 売付可能数量が不足しています | 売付可能数量不足 |
| 13422 | 株式取消注文 | 信用新規建可能額が不足しています | 信用新規建可能額不足 |
| 13423 | 株式取消注文 | 現引可能額が不足しています | 現引可能額不足 |
| 13440 | 株式取消注文 | 非課税口座管理ファイルに問題があります | 非課税口座管理ファイル障害 |
| 13441 | 株式取消注文 | 非課税口座管理にデータがありません | 非課税口座管理レコードなし |
| 13442 | 株式取消注文 | 非課税口座管理更新でエラーが発生しました | 非課税口座管理更新エラー |
| 13443 | 株式取消注文 | 非課税口座可能額が不足しています | 非課税口座可能額不足 |
| 13450 | 株式取消注文 | 顧客手数料ファイルに問題があります | 顧客手数料ファイル障害 |
| 13451 | 株式取消注文 | 顧客手数料レコードなし | 顧客手数料レコードなし |
| 13460 | 株式取消注文 | 手数料マスタファイルに問題があります | 手数料マスタファイル障害 |
| 13461 | 株式取消注文 | 手数料マスタレコードなし | 手数料マスタレコードなし |
| 13470 | 株式取消注文 | 課税率マスタファイルに問題があります | 課税率マスタファイル障害 |
| 13471 | 株式取消注文 | 課税率マスタレコードなし | 課税率マスタレコードなし |
| 13480 | 株式取消注文 | 保管顧客課税別ファイルに問題があります | 保管顧客課税別ファイル障害 |
| 13481 | 株式取消注文 | 選択した口座区分がお預かり銘柄と不一致のため、このご注文はお受けできません。 | 保管顧客課税別有効数量がありません |
| 13482 | 株式取消注文 | 売り超過になります。取消できません。 | 保管顧客課税別有効数量なし |
| 13483 | 株式取消注文 | 保管顧客課税別にデータ作成でエラーが発生しました | 保管顧客課税別レコード作成エラー |
| 13484 | 株式取消注文 | 保管顧客課税別のデータ更新でエラーが発生しました | 保管顧客課税別レコード更新エラー |
| 13490 | 株式取消注文 | 信用建玉明細ファイルに問題があります | 信用建玉明細ファイル障害 |
| 13491 | 株式取消注文 | 信用建玉明細にデータがありません | 信用建玉明細レコードなし |
| 13492 | 株式取消注文 | 信用建玉明細有効数量がありません | 信用建玉明細有効数量なし |
| 13493 | 株式取消注文 | 信用建玉明細のデータ更新でエラーが発生しました | 信用建玉明細レコード更新エラー |
| 13494 | 株式取消注文 | 信用建玉残ファイルに問題があります | 信用建玉残ファイル障害 |
| 13495 | 株式取消注文 | 信用建玉残にデータがありません | 信用建玉残レコードなし |
| 13496 | 株式取消注文 | 信用建玉残にデータ作成でエラーが発生しました | 信用建玉残レコード作成エラー |
| 13497 | 株式取消注文 | 信用建玉残のデータ更新でエラーが発生しました | 信用建玉残レコード更新エラー |
| 13500 | 株式取消注文 | 顧客金銭ファイルに問題があります | 顧客金銭ファイル障害 |
| 13501 | 株式取消注文 | 顧客金銭にデータがありません | 顧客金銭レコードなし |
| 13502 | 株式取消注文 | 顧客拘束金ファイルに問題があります | 顧客拘束金ファイル障害 |
| 13503 | 株式取消注文 | 顧客拘束金にデータがありません | 顧客拘束金レコードなし |
| 13504 | 株式取消注文 | 顧客拘束金レコードレコード作成エラー | 顧客拘束金レコード作成エラー |
| 13505 | 株式取消注文 | 顧客拘束金レコード更新エラー | 顧客拘束金レコード更新エラー |
| 13509 | 株式取消注文 | 保証金率取得でエラーが発生しました | 保証金率取得エラー |
| 13510 | 株式取消注文 | 代用掛目ファイルに問題があります | 代用掛目ファイル障害 |
| 13511 | 株式取消注文 | 日付情報ファイルに問題があります | 日付情報ファイル障害 |
| 13512 | 株式取消注文 | 保管顧客別残ファイルに問題があります | 保管顧客別残ファイル障害 |
| 13513 | 株式取消注文 | 保管顧客別残にデータがありません | 保管顧客別残レコードなし |
| 13514 | 株式取消注文 | 保管顧客別残にデータ作成でエラーが発生しました | 保管顧客別残レコード作成エラー |
| 13515 | 株式取消注文 | 保管顧客別残のデータ更新でエラーが発生しました | 保管顧客別残レコード更新エラー |
| 13516 | 株式取消注文 | 保証金推移ファイルに問題があります | 保証金推移ファイル障害 |
| 13517 | 株式取消注文 | 保証金推移にデータがありません | 保証金推移レコードなし |
| 13518 | 株式取消注文 | 保証金推移にデータ作成でエラーが発生しました | 保証金推移レコード作成エラー |
| 13519 | 株式取消注文 | 保証金推移のデータ更新でエラーが発生しました | 保証金推移レコード更新エラー |
| 13520 | 株式取消注文 | 顧客当日取引情報ファイルに問題があります | 顧客当日取引情報ファイル障害 |
| 13521 | 株式取消注文 | 顧客当日取引情報にデータがありません | 顧客当日取引情報レコードなし |
| 13522 | 株式取消注文 | 顧客当日取引情報にデータ作成でエラーが発生しました | 顧客当日取引情報レコード作成エラー |
| 13523 | 株式取消注文 | 顧客当日取引情報のデータ更新でエラーが発生しました | 顧客当日取引情報レコード更新エラー |
| 13524 | 株式取消注文 | 差金決済管理明細ファイルに問題があります | 差金決済管理明細ファイル障害 |
| 13525 | 株式取消注文 | 差金決済管理明細にデータがありません | 差金決済管理明細レコードなし |
| 13526 | 株式取消注文 | 差金決済管理明細にデータ作成でエラーが発生しました | 差金決済管理明細レコード作成エラー |
| 13527 | 株式取消注文 | 差金決済管理明細のデータ更新でエラーが発生しました | 差金決済管理明細レコード更新エラー |
| 13530 | 株式取消注文 | 譲渡益台帳ファイルに問題が発生しました | 譲渡益台帳ファイル障害 |
| 13531 | 株式取消注文 | 譲渡益台帳レコードがありません | 譲渡益台帳レコードなし |
| 13532 | 株式取消注文 | 譲渡益台帳レコード作成でエラーが発生しました | 譲渡益台帳レコード作成エラー |
| 13534 | 株式取消注文 | 譲渡益台帳レコード更新でエラーが発生しました | 譲渡益台帳レコード更新エラー |
| 13600 | 株式取消注文 | 注文番号（株式）ファイルに問題があります | 注文番号（株式）ファイル障害 |
| 13601 | 株式取消注文 | 注文番号（株式）にデータがありません | 注文番号（株式）レコードなし |
| 13602 | 株式取消注文 | 建玉番号（株式）ファイルに問題があります | 建玉番号（株式）ファイル障害 |
| 13603 | 株式取消注文 | 建玉番号（株式）にデータがありません | 建玉番号（株式）レコードなし |
| 13604 | 株式取消注文 | 親注文株式サマリファイルに問題があります | 親注文株式サマリファイル障害 |
| 13605 | 株式取消注文 | 親注文株式サマリにデータがありません | 親注文株式サマリレコードなし |
| 13606 | 株式取消注文 | 親注文株式サマリ有効数量がありません | 親注文株式サマリ有効数量なし |
| 13607 | 株式取消注文 | 親注文株式サマリ数量を超えています | 親注文株式サマリ数量オーバー |
| 13608 | 株式取消注文 | 親注文株式サマリレコード更新エラー | 親注文株式サマリレコード更新エラー |
| 13609 | 株式取消注文 | 株式サマリファイルに問題があります | 株式サマリファイル障害 |
| 13610 | 株式取消注文 | 株式サマリにデータがありません | 株式サマリレコードなし |
| 13611 | 株式取消注文 | 株式サマリにデータ作成でエラーが発生しました | 株式サマリレコード作成エラー |
| 13612 | 株式取消注文 | 株式サマリのデータ更新でエラーが発生しました | 株式サマリレコード更新エラー |
| 13613 | 株式取消注文 | 株式明細ファイルに問題があります | 株式明細ファイル障害 |
| 13614 | 株式取消注文 | 株式明細にデータがありません | 株式明細レコードなし |
| 13615 | 株式取消注文 | 株式明細更新でエラーが発生しました | 株式明細更新エラー |
| 13616 | 株式取消注文 | 株式明細にデータ作成でエラーが発生しました | 株式明細レコード作成エラー |
| 13617 | 株式取消注文 | 株式注文約定履歴ファイルに問題があります | 株式注文約定履歴ファイル障害 |
| 13618 | 株式取消注文 | 株式注文約定履歴作成でエラーが発生しました | 株式注文約定履歴作成エラー |
| 13621 | 株式取消注文 | 株式返済予約ファイルに問題があります | 株式返済予約ファイル障害 |
| 13622 | 株式取消注文 | 株式返済予約にデータがありません | 株式返済予約レコードなし |
| 13623 | 株式取消注文 | 株式返済予約にデータ作成でエラーが発生しました | 株式返済予約レコード作成エラー |
| 13624 | 株式取消注文 | 株式返済予約のデータ更新でエラーが発生しました | 株式返済予約レコード更新エラー |
| 13625 | 株式取消注文 | 株式返済明細ファイルに問題があります | 株式返済明細ファイル障害 |
| 13626 | 株式取消注文 | 株式返済明細にデータがありません | 株式返済明細レコードなし |
| 13627 | 株式取消注文 | 株式返済明細にデータ作成でエラーが発生しました | 株式返済明細レコード作成エラー |
| 13628 | 株式取消注文 | 株式返済明細のデータ更新でエラーが発生しました | 株式返済明細レコード更新エラー |
| 13640 | 株式取消注文 | 株式約定失効ファイルに問題があります | 株式約定失効ファイル障害 |
| 13641 | 株式取消注文 | 株式約定失効にデータがありません | 株式約定失効レコードなし |
| 13642 | 株式取消注文 | 株式約定失効にデータ作成でエラーが発生しました | 株式約定失効レコード作成エラー |
| 13643 | 株式取消注文 | 株式約定失効のデータ更新でエラーが発生しました | 株式約定失効レコード更新エラー |
| 13645 | 株式取消注文 | システム別設定ファイルに問題があります | システム別設定ファイル障害 |
| 13646 | 株式取消注文 | 銘柄マスタ（株式）ファイルに問題があります | 銘柄マスタ（株式）ファイル障害 |
| 13647 | 株式取消注文 | 銘柄市場マスタ（株式）ファイルに問題があります | 銘柄市場マスタ（株式）ファイル障害 |
| 13700 | 株式取消注文 | 運用ステータス(申告)にデータがありません | 運用ステータス(申告)レコードなし |
| 13701 | 株式取消注文 | 只今の時間帯は受付できません | 運用ステータス(申告).受付停止 |
| 13702 | 株式取消注文 | 運用ステータス(連続注文)にデータがありません | 運用ステータス(連続注文)レコードなし |
| 13703 | 株式取消注文 | 只今の時間帯は受付できません | 運用ステータス(連続注文).受付停止 |
| 13800 | 株式取消注文 | 余力制御ファイルに問題があります | 余力制御ファイル障害 |
| 13802 | 株式取消注文 | 余力制御.取引停止 | 余力制御.取引停止 |
| 13803 | 株式取消注文 | 余力制御.信用新規建停止 | 余力制御.信用新規建停止 |
| 13806 | 株式取消注文 | 余力制御.その他商品買付停止 | 余力制御.その他商品買付停止 |
| 13807 | 株式取消注文 | 追証の未入金があります | 余力制御.追証未入金あり |
| 13991 | 株式取消注文 | セッション情報レコードがありません | セッション情報レコードなし |
| 13992 | 株式取消注文 | セッション情報レコードファイルに問題が発生しました | セッション情報レコードファイル障害 |
| 13993 | 株式取消注文 | セッション情報レコード更新でエラーが発生しました | セッション情報レコード更新エラー |
| 13994 | 株式取消注文 | ボタンが２回以上押されたた可能性があります。注文状況照会をご確認下さい。 | 注文二重送信エラー |
| 13997 | 株式取消注文 | ネットワークでエラーが発生しました | ネットエラー |
| 13998 | 株式取消注文 | ＤＢ接続でエラーが発生しました | ＤＢエラー |
| 13999 | 株式取消注文 | サーバからの応答がありません。結果をご確認下さい。 | タイムアウト |