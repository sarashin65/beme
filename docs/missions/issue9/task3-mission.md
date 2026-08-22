# Task 3: IPADIC ビルダ拡張と `dictionary_v2.tsv` 生成（issue #9 Phase 1）

`docs/missions/issue9/PLAN.md` のタスク 3 を実装してください。§5 タスク3（および関連する §4 の設計）を読んでから着手すること。

**要ネットワーク**: IPADIC 原典（12.2 MB）をダウンロードする必要があります（この環境はネットワーク接続 OK 確認済み）。

## 実装内容（PLAN.md タスク3 の要点）

1. `scripts/build_ipadic_dictionary.py` の `_load_imported_entries` で IPADIC CSV の 左文脈ID(1) / 右文脈ID(2) / 原コスト(3) / 詳細POS(5,6,7) / 活用型(8) / 活用形(9) を保持する
2. 重複統合（`(reading, surface, pos)` の最小コスト採用）で「採用した行の属性を採用する」と実装・ドキュメント両方に明記
3. `source` 列: IPADIC 由来 = `ipadic`、overlay 上書き = `overlay`
4. 出力を `--schema v2` で切り替え（既定は現行 4 列のまま・既存の再生成手順を壊さない）
5. `data/dictionary_v2.tsv` を生成する
6. `data/README.md` に生成手順・列定義・SHA-256・エントリ数・バイト数を追記

## 完了条件（PLAN.md より）

- [ ] `--schema v2` で 6 列追加の TSV が生成でき、`Dictionary.from_tsv` で読める
- [ ] 既定（v1）出力が従来と**バイト単位で同一**（`data/dictionary.tsv` を再生成して `sha256sum` が `efcdf5a1…` と一致）
- [ ] `data/dictionary_v2.tsv` のエントリ数が 349,043 と一致
- [ ] 実測ゲート 2 項目の数値が報告されている（読み込み時間 ≤ 2.0× / 最大 RSS ≤ 1.6×。超えたら判断を報告に書く・その場でリファクタしない）
- [ ] `uv run pytest -q` 全通（79 件 + 新規）

## 検証コマンド（PLAN.md より）

```bash
curl -fLO https://deb.debian.org/debian/pool/main/m/mecab-ipadic/mecab-ipadic_2.7.0-20070801+main.orig.tar.gz
sha256sum mecab-ipadic_2.7.0-20070801+main.orig.tar.gz  # b62f527d... と一致確認
uv run python scripts/build_ipadic_dictionary.py mecab-ipadic_*.tar.gz --output /tmp/dict_v1_check.tsv \
  --expect-sha256 b62f527d881c504576baed9c6ef6561554658b175ce6ae0096a60307e49e3523
sha256sum /tmp/dict_v1_check.tsv data/dictionary.tsv   # 一致すること
uv run python scripts/build_ipadic_dictionary.py mecab-ipadic_*.tar.gz --schema v2 --output data/dictionary_v2.tsv
uv run python -c "from beme.dictionary import Dictionary; d=Dictionary.from_tsv('data/dictionary_v2.tsv'); print(len(d), d.entries[0])"
/usr/bin/time -v uv run beme convert きしゃ 2>&1 | grep -i maximum
```

## 制約

- git commit / push しない
- 作業はこの worktree（`~/work/beme-9`）内で行う
- `src/beme/lattice.py` は変更しない
- 自己確認した出力（sha256 一致・エントリ数・実測ゲート数値・pytest 結果）を貼って報告すること
