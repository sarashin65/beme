# BeME — Beam + IME

LLM と検索技術を活用する自作日本語 IME。ひらがな入力の K-best 候補生成に加え、前後の文脈で候補を再ランキングする Phase 2 モックを実装しています。LLM はまだ接続していません。

## 状態

2026-08-21: Phase 2 の文脈入力・リランカ契約・キーワードベースのモックリランカ・文脈付き評価を実装済み。

- 自作ミニ辞書: `data/dictionary.tsv`
- 文脈なし 12 件、文脈付き 2 件の評価ケース: `data/evaluation_cases.jsonl`
- 現在の固定データに対する実測値: NDCG@5 `0.992972`、MRR@5 `1.000000`
- 文脈付き 2 件はいずれも期待する候補が Top-1

評価ケースと辞書を同時に手作りした小規模なベースライン値であり、一般的な日本語 IME 品質を示す値ではありません。

## アーキテクチャ

1. **辞書引き**: 入力位置ごとに読みの前方一致を検索する。
2. **ラティス構築**: 一致した辞書エントリを辺にし、一致がない位置には一文字の未知語辺を追加する。
3. **K-best Viterbi**: 語コストと POS 遷移コストを加算し、各位置で上位 `beam_width` 経路を保持する。
4. **候補整形**: 表記を一意化し、コスト、表記の順で決定的に並べる。
5. **文脈リランク**: 前文・後方文字列がある場合、Phase 1 の全候補をモックリランカで採点してから `top_k` 件に絞る。
6. **確信度判定**: リランク前の第2候補と第1候補のコスト差を確信度とし、しきい値未満なら将来の LLM リランク対象にする。

## セットアップ

Python 3.11 以上が必要です。ランタイム依存は標準ライブラリだけです。

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install pytest
```

## 変換 CLI

```sh
python -m beme convert きしゃは --top-k 5
python -m beme convert きしゃは --top-k 5 --cascade-threshold 2.0 --json
python -m beme convert きしゃは --dictionary path/to/dictionary.tsv
python -m beme convert こうしょう --preceding-context "歴史を研究する" --top-k 3
python -m beme convert こうしょう --following-context "について商談する" --json
```

テキスト出力例:

```text
reading: きしゃは
1. 記者は  cost=1.200000
2. 汽車は  cost=1.300000
3. 帰社は  cost=1.500000
confidence: 0.100000
cascade_threshold: 2
needs_rerank: yes
```

主なオプション:

- `--top-k`: 出力候補数。既定値は 5。
- `--beam-width`: 各位置で保持する経路数。既定値は 64。
- `--cascade-threshold`: リランク対象とする確信度の上限。初期値は 2.0。
- `--dictionary`: 辞書 TSV。省略時は `data/dictionary.tsv`。
- `--preceding-context`: 変換対象より前の確定済み文・入力履歴。既定値は空文字列。
- `--following-context`: カーソルより後ろの文字列。既定値は空文字列。
- `--json`: 候補のトークン内訳を含む JSON を出力する。

入力は空でないひらがなに限り、長音符 `ー` も許可します。不正な入力や辞書は CLI エラーになります。辞書 TSV のヘッダーは次のとおりです。

```text
reading<TAB>surface<TAB>pos<TAB>cost
```

`cost` は有限の非負数です。

## 精度評価

評価ケースは 1 行 1 JSON の JSONL です。`judgments` の正整数が表記ごとの関連度です。

```json
{"id":"kisha","reading":"きしゃ","judgments":{"記者":3,"汽車":2,"帰社":1}}
{"id":"koushou-history","reading":"こうしょう","preceding_context":"歴史資料を調べる","following_context":"","judgments":{"考証":3}}
```

```sh
python scripts/evaluate.py \
  --dictionary data/dictionary.tsv \
  --cases data/evaluation_cases.jsonl \
  --top-k 5 \
  --beam-width 64 \
  --cascade-threshold 2.0

# 機械可読出力
python scripts/evaluate.py --json
```

レポートにはクエリ数、文脈付きケースを含む NDCG@k / MRR@k、Top-1 accuracy、カスケード対象率、実際の文脈リランク率を出力します。NDCG はゲイン $2^{relevance}-1$ と `log2(rank + 1)` の割引を使い、MRR は最初の関連候補の逆順位です。`preceding_context` と `following_context` は任意で、省略時は空文字列として扱います。

## テスト

```sh
pytest -q
```

辞書検証、ラティス接続、未知語フォールバック、複数分割、候補の一意性と決定性、確信度の境界、CLI、リランカ契約、前文・後方文脈による順位変更、文脈付き NDCG/MRR をテストします。

## 制約と今後

- 辞書は本番用の MeCab/IPADIC ではなく、小規模な自作データです。
- 文脈リランカは説明可能なキーワード一致モックです。候補ごとの関連語辞書を使い、LLM や Cross-Encoder はまだ接続していません。
- ビーム幅が小さすぎると、将来有望な経路を途中で枝刈りする可能性があります。
- 確信度しきい値 2.0 は初期値です。将来はホールドアウトデータ上の誤り率とリランク率の曲線から調整します。
- Phase 2 の `Reranker` 契約は実装済みです。今後は同じ契約のローカル LLM/Cross-Encoder 実装へ差し替えます。

詳細な要件と Phase 1・Phase 2 の設計判断は `docs/spec.md` を参照してください。
