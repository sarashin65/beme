# BeME — Beam + IME

LLM と検索技術を活用する自作日本語 IME。現在は、ひらがな入力を辞書ラティスで分割し、ビーム付き K-best Viterbi で複数候補を返す Phase 1 モックです。LLM リランカーはまだ含みません。

## 状態

2026-08-20: Phase 1 の辞書ベース変換・確信度判定・評価基盤を実装済み。

- 自作ミニ辞書: `data/dictionary.tsv`
- 同音異義語を含む評価ケース 12 件: `data/evaluation_cases.jsonl`
- 現在の固定データに対する実測値: NDCG@5 `0.991800`、MRR@5 `1.000000`

評価ケースと辞書を同時に手作りした小規模なベースライン値であり、一般的な日本語 IME 品質を示す値ではありません。

## アーキテクチャ

1. **辞書引き**: 入力位置ごとに読みの前方一致を検索する。
2. **ラティス構築**: 一致した辞書エントリを辺にし、一致がない位置には一文字の未知語辺を追加する。
3. **K-best Viterbi**: 語コストと POS 遷移コストを加算し、各位置で上位 `beam_width` 経路を保持する。
4. **候補整形**: 表記を一意化し、コスト、表記の順で決定的に並べる。
5. **確信度判定**: `第2候補のコスト - 第1候補のコスト` を確信度とし、しきい値未満なら将来のリランク対象にする。

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

レポートにはクエリ数、NDCG@k、MRR@k、Top-1 accuracy、リランク率を出力します。NDCG はゲイン $2^{relevance}-1$ と `log2(rank + 1)` の割引を使い、MRR は最初の関連候補の逆順位です。

## テスト

```sh
pytest -q
```

辞書検証、ラティス接続、未知語フォールバック、複数分割、候補の一意性と決定性、確信度の境界、CLI、NDCG/MRR をテストします。

## 制約と今後

- 辞書は本番用の MeCab/IPADIC ではなく、小規模な自作データです。
- 語コストと POS 遷移コストは静的です。入力履歴、文全体、カーソル後方の文脈はまだ使いません。
- ビーム幅が小さすぎると、将来有望な経路を途中で枝刈りする可能性があります。
- 確信度しきい値 2.0 は初期値です。将来はホールドアウトデータ上の誤り率とリランク率の曲線から調整します。
- Phase 2 ではローカル LLM/Cross-Encoder による文脈リランクを追加する構想です。

詳細な要件と Phase 1 の設計判断は `docs/spec.md` を参照してください。
