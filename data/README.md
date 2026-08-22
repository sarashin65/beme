# 評価データ

## BeME 既定辞書（MeCab IPADIC 取り込み）

`dictionary.tsv` は MeCab IPADIC `2.7.0-20070801` の原典 CSV から生成した BeME の既定辞書です。手作りの優先語は `dictionary_manual.tsv` に分離し、生成時に上書きマージします。IPADIC 由来のカタカナ surface には P0 のコスト補正を適用済みです。

- 出所: [MeCab IPADIC](https://taku910.github.io/mecab/)、Debian source package `mecab-ipadic 2.7.0-20070801+main-3`
- 固定ソース: [`mecab-ipadic_2.7.0-20070801+main.orig.tar.gz`](https://deb.debian.org/debian/pool/main/m/mecab-ipadic/mecab-ipadic_2.7.0-20070801+main.orig.tar.gz)
- ソース SHA-256: `b62f527d881c504576baed9c6ef6561554658b175ce6ae0096a60307e49e3523`
- 生成物: 349,043 エントリ、14,583,715 bytes
- 生成物 SHA-256: `efcdf5a1baeaeb74f8ca6c85552d6fac335d40ad274f1e77181c21c8f37f617b`
- ライセンス: NAIST / ICOT の IPADIC ライセンス。著作権表示、再配布条件、無保証条項の全文は `licenses/IPADIC.txt`

### v2 辞書（Task 3）

追加属性を保持する `data/dictionary_v2.tsv` は次のコマンドで生成します（原典の SHA-256 を検証します）。

```sh
uv run python scripts/build_ipadic_dictionary.py \
  mecab-ipadic_2.7.0-20070801+main.orig.tar.gz \
  --schema v2 --output data/dictionary_v2.tsv \
  --expect-sha256 b62f527d881c504576baed9c6ef6561554658b175ce6ae0096a60307e49e3523
```

列は必須 4 列（`reading`, `surface`, `pos`, `cost`）に、`pos_detail`, `conjugation`, `left_id`, `right_id`, `source`, `raw_cost` を加えた 10 列です。IPADIC の重複キーは最小コストの行を採用し、その採用行の属性も採用します。IPADIC 由来の `source` は `ipadic`、overlay 上書きは `overlay` です。

- 生成物: 349,043 エントリ、31,566,919 bytes
- 生成物 SHA-256: `d4f5446d3d79481fb1736eac0cc994496d50c3f924b0ba35b9e820c4cb067903`

WSL2 では root 権限を必要とする apt の代わりに、開発依存 `mecab-python3>=1.0.10` と `ipadic>=1.0.0` をプロジェクトの `.venv` に導入します。実測環境では MeCab `0.996`、`mecab-python3 1.0.12`、`ipadic 1.0.0` を使用しました。`mecab-python3` / MeCab は BSD・LGPL・GPL の選択ライセンスで配布される変換ツールで、生成 TSV にそのコードは含まれません。生成スクリプトは全語彙を列挙できないコンパイル済み `sys.dic` ではなく、固定した原典 CSV を直接読みます。

再生成手順:

```sh
uv sync --group dev
curl -fLO https://deb.debian.org/debian/pool/main/m/mecab-ipadic/mecab-ipadic_2.7.0-20070801+main.orig.tar.gz
sha256sum mecab-ipadic_2.7.0-20070801+main.orig.tar.gz
python scripts/build_ipadic_dictionary.py \
  mecab-ipadic_2.7.0-20070801+main.orig.tar.gz \
  --expect-sha256 b62f527d881c504576baed9c6ef6561554658b175ce6ae0096a60307e49e3523
```

変換規則:

- EUC-JP の IPADIC CSV から表記、階層 POS、読み、単語コストを取得し、カタカナ読みをひらがなへ正規化する。
- IPADIC の階層 POS を BeME の `noun` / `proper_noun` / `verb` / `particle` 等へ集約する。
- 記号 208 行とひらがな読みを作れない 9 行を除外し、同じ `(reading, surface, pos)` は IPADIC コストが最小の行へ集約する。頻度、固有名詞、読み長による削減は行わず、活用形と固有名詞を含む語彙を維持する。
- IPADIC コスト $c$ は低いほど良い順序を維持したまま、$1.75 + 3.50 \times \mathrm{clamp}((c-3000)/7000, 0, 1)$ へ線形変換する。手作り辞書の最大コスト `1.65` より取り込み語の下限 `1.75` を高くし、「きしゃ」「こうしょう」等の既存優先順位を保証する。
- NFKC 正規化後の surface に全角または半角のカタカナ文字が 1 文字以上含まれる IPADIC 由来エントリは、写像後コストへ固定で `+1.0` する。形態素解析用 IPADIC のカタカナ表層が IME の生成候補へ過剰に出る問題への対症療法であり、手作り overlay には適用しない。AJIMEE 200 件では `+1.0` が最良で、`+1.5..3.0` は正しいカタカナ外来語を抑えて劣化したため採用しない。

生成物 14.6 MB はリポジトリへ同梱する。固定ソースアーカイブ 12.2 MB は生成物と重複し、上記 URL と SHA-256 から再取得できるため同梱しない。Intel N100 での実測は、TSV ロード 1.851 秒、14 手作り読みのラティス構築平均 0.228 ms、クリーンな `beme convert きしゃ` プロセスの最大 RSS 159,524 KiB だった。従来の文字単位 Python trie は大辞書で不要なノードを大量生成するため、入力の終端候補を exact-reading index へ照会する方式へ変更した。

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
