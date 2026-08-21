# BeME — Beam + IME

LLM と検索技術を活用する自作日本語 IME。ひらがな入力の K-best 候補生成と、前後の文脈を使うキーワードモック / Qwen2.5-0.5B logits リランカを実装しています。

## 状態

2026-08-21: Phase 2 のモデル組み込みに加え、評価データ拡充ステップ 1 を実装済み。

- 自作ミニ辞書: `data/dictionary.tsv`
- BeME 手作り評価: 文脈なし 12 件、文脈付き 2 件を `data/evaluation_cases.jsonl` に維持
- AJIMEE-Bench: 文章変換 200 件を `data/ajimee_bench_jwtd_v2_v1.json` に導入
- Mozc quality regression: コーナーケース 500 件を `data/mozc_oss.tsv` に導入
- 手作り 14 件のモック実測値: Top-1 `1.000000`、NDCG@5 `0.992972`
- 手作り 14 件の LLM 実測値: Top-1 `0.928571`、NDCG@5 `0.966609`

手作り評価はケースと辞書を同時に作った小規模なベースラインであり、一般的な日本語 IME 品質を示しません。外部データは独立した品質測定と不足語彙・既知の誤変換の検出に使います。

## アーキテクチャ

1. **辞書引き**: 入力位置ごとに読みの前方一致を検索する。
2. **ラティス構築**: 一致した辞書エントリを辺にし、一致がない位置には一文字の未知語辺を追加する。
3. **K-best Viterbi**: 語コストと POS 遷移コストを加算し、各位置で上位 `beam_width` 経路を保持する。
4. **候補整形**: 表記を一意化し、コスト、表記の順で決定的に並べる。
5. **文脈リランク**: 前文・後方文字列がある場合、Phase 1 の全候補を選択した `Reranker`（`mock` / `llm`）で採点してから `top_k` 件に絞る。
6. **確信度判定**: リランク前の第2候補と第1候補のコスト差を確信度とし、しきい値未満なら将来の LLM リランク対象にする。

## セットアップ

Python 3.11 以上が必要です。`llama-cpp-python` は通常のランタイム依存です。

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install pytest
```

LLM リランカの既定モデルは QuantFactory が量子化した Qwen2.5-0.5B base の `Q4_K_M`（元モデル・ライセンスは Apache 2.0）です。GGUF はリポジトリ外の `~/.cache/beme/models/` に置きます。

```sh
python -m pip install huggingface-hub
python -c "import os; from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='QuantFactory/Qwen2.5-0.5B-GGUF', filename='Qwen2.5-0.5B.Q4_K_M.gguf', local_dir=os.path.expanduser('~/.cache/beme/models'))"
```

## 変換 CLI

```sh
python -m beme convert きしゃは --top-k 5
python -m beme convert きしゃは --top-k 5 --cascade-threshold 2.0 --json
python -m beme convert きしゃは --dictionary path/to/dictionary.tsv
python -m beme convert こうしょう --preceding-context "歴史を研究する" --top-k 3
python -m beme convert こうしょう --following-context "について商談する" --json
python -m beme convert こうしょう \
  --preceding-context "歴史を研究する" \
  --reranker llm \
  --top-k 3
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
- `--reranker`: 文脈リランカ。`mock`（既定）または `llm`。
- `--model-path`: LLM の GGUF。既定値は `~/.cache/beme/models/Qwen2.5-0.5B.Q4_K_M.gguf`。
- `--llm-weight` / `--viterbi-weight`: 最終スコアの重み。両方の既定値は `1.0`。
- `--llm-context-size`: llama.cpp のコンテキスト長。既定値は 256 トークン。
- `--json`: 候補のトークン内訳を含む JSON を出力する。

入力は空でないひらがなに限り、長音符 `ー` も許可します。不正な入力や辞書は CLI エラーになります。辞書 TSV のヘッダーは次のとおりです。

```text
reading<TAB>surface<TAB>pos<TAB>cost
```

`cost` は有限の非負数です。

LLM の最終スコア（小さいほど上位）は次式です。

$$
\text{score}
= w_\text{viterbi}\,\text{cost}
- w_\text{llm}\frac{1}{N}\sum_{i=1}^{N}\log P(t_i\mid \text{preceding},t_{<i})
$$

前文を一度プレフィルして `save_state()` し、候補ごとに `load_state()` して KV cache を復元します。後方文字列がある場合の評価トークン列は `candidate + following` です。コンテキスト上限を超える場合は後方文字列の末尾側、次に前文の先頭側を切り詰めます。モデルファイル欠落、`llama-cpp-python` import 失敗、モデルロード失敗時は警告を標準エラーへ出してモックへフォールバックします。

## 精度評価

評価ケースは 1 行 1 JSON の JSONL です。`judgments` の正整数が表記ごとの関連度です。

```json
{"id":"kisha","reading":"きしゃ","judgments":{"記者":3,"汽車":2,"帰社":1}}
{"id":"koushou-history","reading":"こうしょう","preceding_context":"歴史資料を調べる","following_context":"","judgments":{"考証":3}}
```

```sh
# 既定: 手作り 14 件のランキング評価
python scripts/evaluate.py \
  --suite handmade \
  --dictionary data/dictionary.tsv \
  --cases data/evaluation_cases.jsonl \
  --top-k 5 \
  --beam-width 64 \
  --cascade-threshold 2.0

# 外部データを個別または一括評価
python scripts/evaluate.py --suite ajimee
python scripts/evaluate.py --suite mozc --top-k 5
python scripts/evaluate.py --suite all --top-k 5

# 機械可読出力（項目別結果を含む）
python scripts/evaluate.py --suite all --reranker mock --json
python scripts/evaluate.py --suite handmade --reranker llm --json
```

レポートにはクエリ数、文脈付きケースを含む NDCG@k / MRR@k、Top-1 accuracy、カスケード対象率、実際の文脈リランク率、リランカの候補 1 件あたりレイテンシを出力します。レイテンシはモデルロードを除く各 `rerank()` 全体（共通文脈のプレフィルと状態保存を含む）を候補数で割った値です。NDCG はゲイン $2^{relevance}-1$ と `log2(rank + 1)` の割引を使い、MRR は最初の関連候補の逆順位です。`preceding_context` と `following_context` は任意で、省略時は空文字列として扱います。

2026-08-21、Intel N100、CPU 版 llama.cpp、`top_k=5`、`beam_width=64`、コンテキスト長 256、文脈リランク対象 82 候補での 1 回の実測:

| リランカ | Top-1 | NDCG@5 | MRR@5 | 候補レイテンシ |
|---|---:|---:|---:|---:|
| mock | 1.000000 | 0.992972 | 1.000000 | 0.003 ms |
| llm | 0.928571 | 0.966609 | 0.964286 | 150.891 ms |

このデータでは LLM はモックより精度・速度ともに劣ります。特に一般的な「交渉」を好む頻度バイアスが残るため、評価ケースの拡充と重み校正が必要です。

### 外部評価データと判定方法

詳細な出所、固定 commit、SHA-256、ライセンス、データごとの判定規則は [`data/README.md`](data/README.md) を参照してください。

**AJIMEE-Bench** の `JWTD_v2/v1` は 200 件の全角カタカナ読みと複数の許容出力を持ちます（ローマ字入力ではありません）。全角カタカナをひらがなへ正規化し、句読点・数字・記号はリテラルとして保持します。リテラルで区切られたひらがな区間を BeME で変換し、区間の K-best を合成します。`context_text` は左文脈に使います。項目ごとに許容出力のうち最小の Levenshtein 編集距離と $\mathrm{CER}=\mathrm{distance}/|\mathrm{reference}|$ を採用し、平均 CER、平均編集距離、完全一致率を集計します。

**Mozc `oss.tsv`** は conversion 系 498 件を回帰判定し、BeME に存在しない suggestion 系 2 件を理由付きでスキップします。`Conversion Expected [N]` は正解全文の順位（`N` は 0 始まり）、`Conversion Match` は Top-1 の必須部分文字列、`Conversion Not Match` は Top-1 に含めてはいけない既知の誤変換として判定します。`--top-k` より深い期待順位は観測できないため、必要な深さを指定して評価してください。

ライセンスは upstream の一次情報に従います。

- AJIMEE-Bench `JWTD_v2/v1`: **CC BY-SA 3.0**。依頼時資料の MIT 記載とは異なり、upstream README がデータは元データに準拠して CC BY-SA 3.0 と明記しています。帰属: azooKey/AJIMEE-Bench contributors および日本語 Wikipedia 入力誤りデータセット v2。`data/licenses/AJIMEE-BENCH-CC-BY-SA-3.0.txt` を参照。
- Mozc `oss.tsv`: **BSD 3-Clause**、Copyright 2010-2018 Google Inc. `data/licenses/MOZC-BSD-3-Clause.txt` に著作権表示、条件、免責条項を収録。

## テスト

```sh
pytest -q
```

辞書検証、ラティス接続、未知語フォールバック、複数分割、候補の一意性と決定性、確信度の境界、CLI、リランカ契約、前文・後方文脈による順位変更、文脈付き NDCG/MRR に加え、カタカナ→ひらがな入力アダプタ、句読点保持、Levenshtein/CER、Mozc の正例・禁止誤変換・順位・スキップ規則、導入データ件数（200/500）、手作り14件の維持をテストします。LLM 用テストは実モデルをロードせず、決定的な fake backend で平均尤度、線形結合、`save_state` / `load_state`、ロード失敗時のフォールバックを検証します。

## 制約と今後

- 辞書は本番用の MeCab/IPADIC ではなく、小規模な自作データです。
- キーワードモックは小規模評価データに合わせた手作りルールです。
- 0.5B Q4_K_M でも N100 CPU 上の候補レイテンシは約 151 ms であり、目標の数十 ms を満たしません。
- ビーム幅が小さすぎると、有望な経路を途中で枝刈りする可能性があります。
- 確信度しきい値 2.0 と LLM/Viterbi 重み 1.0 は初期値です。今後は独立したホールドアウトデータで調整します。

詳細な要件と Phase 1・Phase 2 の設計判断は `docs/spec.md` を参照してください。
