# 評価データ

## BeME 手作りケース

`evaluation_cases.jsonl` は BeME 独自の 14 件（文脈なし 12 件、文脈付き 2 件）です。既存の NDCG / MRR / Top-1 評価に使います。

## AJIMEE-Bench

- ファイル: `ajimee_bench_jwtd_v2_v1.json`（200 件、upstream の未改変コピー）
- 出所: [azooKey/AJIMEE-Bench](https://github.com/azooKey/AJIMEE-Bench)、`JWTD_v2/v1/evaluation_items.json`
- 固定版: [`401666cd56d1a570c2021798b64b6da4396bfd45`](https://github.com/azooKey/AJIMEE-Bench/commit/401666cd56d1a570c2021798b64b6da4396bfd45)
- SHA-256: `e9eb668fd6aa14b1e26436f429b5550108af0a1dfd443b8cea0bcb3ab3028fca`
- ライセンス: **CC BY-SA 3.0**。詳細・帰属は `licenses/AJIMEE-BENCH-CC-BY-SA-3.0.txt`

依頼時の調査レポートでは MIT とされていましたが、upstream README は `JWTD_v2/v1` のデータを「元データに準拠し CC BY-SA 3.0」と明記しています。このため、MIT とは表記せず upstream のデータライセンスに従います。評価コード `utils.py` の CC0 表記はデータには適用されません。

入力フィールドはローマ字ではなく全角カタカナの読みです。BeME は全角カタカナをひらがなへ変換し、句読点・数字・記号をそのまま通過させ、ひらがな連続区間ごとの K-best を合成します。`context_text` は左文脈としてリランカへ渡します。各項目では複数の `expected_output` のうち最小となる編集距離と CER（編集距離 / 正解文字数）を採用し、全 200 件の平均、完全一致率、項目別結果を出力します。

## Mozc quality regression test

- ファイル: `mozc_oss.tsv`（500 件、upstream の未改変コピー）
- 出所: [google/mozc](https://github.com/google/mozc)、`src/data/test/quality_regression_test/oss.tsv`
- 固定版: [`851c3fe33060d2a6090363e4d7ec44fafde2c03d`](https://github.com/google/mozc/commit/851c3fe33060d2a6090363e4d7ec44fafde2c03d)
- SHA-256: `053a43e72d174da43b351278cadc471abe08cea9fe98829dafe3d152580fcff0`
- ライセンス: **BSD 3-Clause**。著作権表示・条件・免責条項は `licenses/MOZC-BSD-3-Clause.txt`

判定は Mozc の conversion コマンドに合わせます。

- `Conversion Expected [N]`: 正解全文が 0 始まりの順位 `N` 以内に存在する（省略時は Top-1）。
- `Conversion Match`: Top-1 に指定部分文字列が存在する。
- `Conversion Not Match`: 既知の誤変換部分文字列が Top-1 に存在しない。
- suggestion / prediction 系: BeME に同モードがないため合否へ混ぜず、理由付き `skipped` とする。現在の `oss.tsv` では `Suggestion Not Expected` 2 件が該当する。

`Conversion Not Match` は「変換不能」扱いで黙って通すのではなく、禁止すべき誤変換が出力に混入していないことを検査します。正例が辞書にない場合は失敗として残るため、回帰データが辞書・候補生成の不足を可視化します。
