# S3 事前調査

## D1 manifest

`data/d1/manifest.json` の `source` は次の固定 snapshot を示している。

| field | value |
|---|---|
| `dataset` | `wikimedia/wikipedia` |
| `config` | `20231101.ja` |
| `revision` | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` |
| `license` | `CC BY-SA 3.0 and GFDL` |

manifest では D1 の記事数は `train=696`, `dev=65`, `test=129` のユニーク記事、処理記事数は 5,104 と記録されている。記事本文の実体は manifest には埋め込まれておらず、記事 JSONL または固定 revision の Hugging Face `train` split から読み込む必要がある。D1 の split は manifest の `seed=20260822` と範囲 `train<7700`, `dev<8500`, `test<10000` を使う。

## `scripts/build_d1_data.py`

文境界は次で固定されている。

```python
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？])|[\r\n]+")
```

`article_groups()` はこの正規表現で記事本文を分割し、各断片を `strip()` して空断片を捨てる。MeCab/fugashi の `sentence_nodes()` は node surface の位置を左から探索し、`node_reading()` は MeCab feature の第 8 欄をカタカナからひらがなへ変換する。Wikipedia の取得は `datasets.load_dataset(dataset, config, split="train", streaming=True, revision=revision)` で行い、各 row の `id` と `text` を `Article` にする。ローカル確認用には `iter_jsonl_articles()` が `id`/`text`/任意の `title`,`url` を読む。

S3 では同じ境界規則と全文 article hash を使うが、D1 の局所 span 候補、converter 候補順位、`balanced_select()` は使わない。全文を一つの candidate とし、MeCab の読みを暫定 gold path として保存する。

## S1 調査結果との接続

`S1-code-survey.md` では、E0 は文単位の固定 gold、長さ層ごとの hash 順選択、article 上限、D1/AJIMEE/handmade 文書のリーク監査を別々に行う方針になっている。特に scorer で gold path を再選択せず、入力読みと原文 surface の offset をそのまま保存すること、`balanced_select()` を再利用しないことが明記されている。S3 実装では S2 の `build_e0_leakage_index()` と `check_e0_leakage()` を抽出前後の gate に用いる。

