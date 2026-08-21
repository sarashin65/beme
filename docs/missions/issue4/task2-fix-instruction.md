# Task 2 修正: apply_cost_learning.py の MeCab 初期化失敗対応

## 問題

`scripts/apply_cost_learning.py` を実行すると、mecab-python3 の Tagger 初期化で失敗する:

```
RuntimeError: Failed initializing MeCab.
arguments: -d <venv>/site-packages/ipadic/dicdir
[ifs] no such file or directory: /usr/local/etc/mecabrc
```

この環境には mecabrc が存在しない（/usr/local/etc/mecabrc・/etc/mecabrc・~/.mecabrc すべて無し）。mecab-python3 はシステム mecabrc を要求するため、この環境では Tagger が初期化できない。

## 修正方針

**mecab-python3 をやめ、`fugashi` に切り替える**（当初プランでも fugashi 想定・pure wheel で mecabrc 不要・ipadic と直接統合）:

- `pyproject.toml` の dev 依存に `fugashi>=1.0.0` を追加（mecab-python3 は使わないなら外してよい）
- `scripts/apply_cost_learning.py` の解析部分を `fugashi.Tagger()` に書き換える
  - 例: `from fugashi import Tagger; tagger = Tagger('-Owakati')` または `Tagger()`（ipadic 同梱）
  - 各 token から surface / pos を取得する方法に合わせて実装を調整
- 頻度集計の粒度 `(reading, surface, pos)` は維持（`src/beme/ipadic.py` の `katakana_to_hiragana` / `map_ipadic_pos` を利用）

## 完了条件

- `uv run python scripts/apply_cost_learning.py --corpus data/corpus.txt --dictionary /tmp/test_dict.tsv --workers 2 --chunk-lines 10 --json` が exit 0 で完了する（/tmp への出力はシェル側でコピーして検証）
- 更新された辞書の diff を確認できる（高頻度語のコストが減算されている）
- 既存テストがパスする（`uv run pytest -q`）

## 制約

- git commit / push はしない（作業ツリーに変更を残す）
- 対象外ファイルを変更しない
- 実装後は自己確認（実行・diff・テスト）して報告
