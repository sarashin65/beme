# S1: d1 artifact の復元・固定

作成日: 2026-09-08  
対象 checkout: `/home/sarashin/work/beme-9p4`  
参照 manifest: `data/d1/manifest.json`

## 結論

採用する d1 artifact 一式は、既存 worktree `/home/sarashin/work/beme/data/d1/` にある版である。この版は現 checkout の `data/d1/manifest.json` が記録する全 artifact の SHA-256 と一致し、manifest の件数とも一致する。

現 checkout には JSONL 本体は存在しない。ファイル作成制約により、本 checkout へ JSONL を複製せず、S1 では発見場所・hash・件数をこの文書で固定する。S2 以降が現 checkout から実行される場合は、採用ディレクトリを入力として明示するか、別途 artifact 復元手順を実施すること。manifest の hash と異なるファイルを補正・採用してはならない。

## manifest の記録

`data/d1/manifest.json` の `counts` は次のとおりである。

| split/artifact | manifest の件数・行数 | manifest SHA-256 |
|---|---:|---|
| `train.jsonl` | 10,000 | `b9609547a66d8b15ab72d416b1f964b62881fa1d77d3ddd2cd900d7d3c23baa8` |
| `dev.jsonl` | 1,000 | `8a6aea13dcc1caf61708cf94f9db836e14ba3544fa5c4c522d9668d418182f47` |
| `test.jsonl` | 2,000 | `7b0ad8ff3f9abf11ccfb9488214db0199ecd9b06d07928dea0fb6c8681f6f2ff` |
| `articles.jsonl` | manifest に行数フィールドはない（実測 890） | `7858a39ff39817edb4fc379dc03364fa1746bce8e6507ad003813e7842a20a6e` |
| `audit.jsonl` | manifest に行数フィールドはない（実測 200） | `71d584f5b1b74e1fae67e8165147ce15638f5aa143f7a9693a812c257eed3ee3` |

manifest の source は Wikipedia `wikimedia/wikipedia`、config `20231101.ja`、revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`、seed `20260822` を記録している。したがって、依頼文にある 12,000/2,000/2,000 ではなく、実 artifact と manifest の 10,000/1,000/2,000 を採用する。

## 発見場所と照合

確認した worktree は `/home/sarashin/work/beme`、`beme-19`、`beme-21`、`beme-4`、`beme-9`、`beme-9p2`、`beme-9p3`、`beme-9p4`、`beme-kexp`、`beme-p2` である。

- `/home/sarashin/work/beme/data/d1/`: 5 artifact が存在し、上表の hash と全て一致。採用版。
- `/home/sarashin/work/beme-19/data/d1/`: train/dev/test は存在し、3つとも同じ hash と行数。ただし `articles.jsonl` と `audit.jsonl` がないため、一式としては採用しない。
- その他の確認対象 worktree: d1 JSONL 一式は見つからなかった。現在の checkout も manifest、`data_gate.json`、`qwen_token_count.json` のみで JSONL 本体はない。

照合は各ファイルに対する `sha256sum` と `wc -l` で行った。採用版の実測は次のとおりである。

```text
b9609547a66d8b15ab72d416b1f964b62881fa1d77d3ddd2cd900d7d3c23baa8  train.jsonl  10000 lines
8a6aea13dcc1caf61708cf94f9db836e14ba3544fa5c4c522d9668d418182f47  dev.jsonl    1000 lines
7b0ad8ff3f9abf11ccfb9488214db0199ecd9b06d07928dea0fb6c8681f6f2ff  test.jsonl   2000 lines
7858a39ff39817edb4fc379dc03364fa1746bce8e6507ad003813e7842a20a6e  articles.jsonl 890 lines
71d584f5b1b74e1fae67e8165147ce15638f5aa143f7a9693a812c257eed3ee3  audit.jsonl 200 lines
```

この実行環境から別の ws machine は参照できないため、ws machine 上の追加 artifact の有無は検証していない。少なくとも共有された `/home/sarashin` 配下で採用可能な完全な一式は上記だけである。

## S2 以降への入力契約

1. `train.jsonl`、`dev.jsonl`、`test.jsonl` の期待件数と SHA-256 は上表から変更しない。
2. `articles.jsonl` は d1 の記事本文 manifest として必須であり、`beme-19` のように欠けた版で E0 のリーク監査を開始しない。
3. artifact の実体を現 checkout に配置する場合も、配置後に上表の hash を再計算し、失敗時は fail closed とする。
4. d1 の件数を依頼文側の 12,000/2,000/2,000 に合わせて再生成・推測補完しない。再生成が必要になった場合は別版として新しい manifest/hash を発行する。
