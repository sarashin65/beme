# BeME — Beam + IME

LLM と検索技術を活用する自作日本語 IME。ひらがな入力の K-best 候補生成と、前後の文脈を使うキーワードモック / Qwen2.5-0.5B logits / Cross-Encoder リランカを実装しています。

## 状態

2026-08-21: 評価データ拡充ステップ 1、辞書拡充 B（MeCab IPADIC 取り込み）、P0（カタカナ表層の抑制）まで実装済み。

- 既定辞書: IPADIC 由来 349,043 エントリを `data/dictionary.tsv` に同梱
- 手作り優先語: 133 エントリを `data/dictionary_manual.tsv` に維持し、生成時に上書き
- BeME 手作り評価: 文脈なし 12 件、文脈付き 2 件を `data/evaluation_cases.jsonl` に維持
- AJIMEE-Bench: 文章変換 200 件を `data/ajimee_bench_jwtd_v2_v1.json` に導入
- Mozc quality regression: コーナーケース 500 件を `data/mozc_oss.tsv` に導入
- P0 適用後 mock: 手作り Top-1 / NDCG@5 は `1.000000 / 0.992972` を維持、AJIMEE CER は IPADIC 取り込み直後の `0.341433` から `0.290080`、Mozc pass は `81/498` から `88/498` へ改善

手作り評価は優先語と同時に作った小規模な回帰セットであり、一般的な日本語 IME 品質を単独では示しません。外部データは独立した文章精度と不足語彙・既知の誤変換の検出に使います。

## アーキテクチャ

1. **辞書引き**: 入力位置ごとに読みの前方一致を検索する。
2. **ラティス構築**: 一致した辞書エントリを辺にし、一致がない位置には一文字の未知語辺を追加する。
3. **K-best Viterbi**: 語コストと POS 遷移コストを加算し、各位置で上位 `beam_width` 経路を保持する。
4. **候補整形**: 表記を一意化し、コスト、表記の順で決定的に並べる。
5. **文脈リランク**: 前文・後方文字列がある場合、Phase 1 の全候補を選択した `Reranker`（`mock` / `llm` / `cross-encoder`）で採点してから `top_k` 件に絞る。
6. **確信度判定**: リランク前の第2候補と第1候補のコスト差を確信度とし、しきい値未満なら将来の LLM リランク対象にする。

## セットアップ

Python 3.11 以上が必要です。`llama-cpp-python`、`sentence-transformers`、CPU 版 PyTorch は通常のランタイム依存です。

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

Cross-Encoder は初回実行時に Hugging Face から取得し、リポジトリ外の `~/.cache/beme/cross-encoder/` に保存します。既定は日本語特化の `hotchpotch/japanese-reranker-base-v2`（MIT、Ruri v3 130M base）、比較対象は `BAAI/bge-reranker-v2-m3`（568M、多言語、Apache 2.0）です。

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

python -m beme convert こうしょう \
  --preceding-context "歴史を研究する" \
  --reranker cross-encoder \
  --cross-encoder-model hotchpotch/japanese-reranker-base-v2 \
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
- `--reranker`: 文脈リランカ。`mock`（既定）、`llm`、`cross-encoder`。
- `--model-path`: LLM の GGUF。既定値は `~/.cache/beme/models/Qwen2.5-0.5B.Q4_K_M.gguf`。
- `--llm-weight` / `--viterbi-weight`: LLM 最終スコアの重み。両方の既定値は `1.0`。`--viterbi-weight` は Cross-Encoder にも適用する。
- `--llm-context-size`: llama.cpp のコンテキスト長。既定値は 256 トークン。
- `--cross-encoder-model`: 上記 2 モデルから選択。既定は日本語特化モデル。
- `--cross-encoder-cache-dir`: モデルキャッシュ。既定値は `~/.cache/beme/cross-encoder/`。
- `--cross-encoder-weight`: Cross-Encoder の関連度重み。既定値は `1.0`。
- `--cross-encoder-batch-size`: CPU 推論バッチサイズ。既定値は `8`。
- `--cross-encoder-local-files-only`: キャッシュにないモデルをダウンロードせず、ロード失敗として扱う。
- `--json`: 候補のトークン内訳を含む JSON を出力する。

入力は空でないひらがなに限り、長音符 `ー` も許可します。不正な入力や辞書は CLI エラーになります。辞書 TSV のヘッダーは次のとおりです。

```text
reading<TAB>surface<TAB>pos<TAB>cost
```

`cost` は有限の非負数です。

## 辞書の再生成

MeCab / IPADIC の検証環境は開発依存として導入します。コンパイル済み辞書ではなく、固定した IPADIC 原典 CSV を `scripts/build_ipadic_dictionary.py` で変換します。

```sh
uv sync --group dev
curl -fLO https://deb.debian.org/debian/pool/main/m/mecab-ipadic/mecab-ipadic_2.7.0-20070801+main.orig.tar.gz
python scripts/build_ipadic_dictionary.py \
  mecab-ipadic_2.7.0-20070801+main.orig.tar.gz \
  --expect-sha256 b62f527d881c504576baed9c6ef6561554658b175ce6ae0096a60307e49e3523
```

スクリプトはカタカナ読みをひらがなへ変換し、階層 POS を BeME の POS へ集約し、IPADIC 単語コストを `1.75..5.25` へ線形写像します。さらに、NFKC 正規化後の surface に全角または半角のカタカナ文字が 1 文字以上含まれる IPADIC 由来エントリだけ、写像後コストへ固定で `+1.0` します。IPADIC は形態素解析用辞書なので、解析入力として必要なカタカナ表層が IME の生成候補では過剰に出るためです。これは IPADIC を IME 辞書に使うことへの対症療法です。AJIMEE 200 件のアブレーションでは `+1.0` が最良（CER `0.3414 → 0.2914`）で、`+1.5..3.0` は正しいカタカナ外来語まで抑えて劣化したため、完全除外やより強い値にはしません。手作り overlay は補正せず、コスト上限 `1.65` より取り込み語の下限を高くして既存の同音異義語順位を維持します。固定ソース、SHA-256、全変換規則、サイズ・メモリ実測、ライセンスは [`data/README.md`](data/README.md) を参照してください。

LLM の最終スコア（小さいほど上位）は次式です。

$$
\text{score}
= w_\text{viterbi}\,\text{cost}
- w_\text{llm}\frac{1}{N}\sum_{i=1}^{N}\log P(t_i\mid \text{preceding},t_{<i})
$$

前文を一度プレフィルして `save_state()` し、候補ごとに `load_state()` して KV cache を復元します。後方文字列がある場合の評価トークン列は `candidate + following` です。コンテキスト上限を超える場合は後方文字列の末尾側、次に前文の先頭側を切り詰めます。

Cross-Encoder は `preceding + "【変換対象】" + following` を query、候補表記を passage として一括採点します。モデルの raw classification logit を関連度 $s_\text{CE}$ とし、次式で Viterbi コストと結合します。

$$
\text{score}
:= w_\text{viterbi}\,\text{cost}
- w_\text{CE}\,s_\text{CE}(\text{context},\text{candidate})
$$

小さいほど上位です。単語 1 件の passage では sigmoid 後の値が極小域に圧縮され、Viterbi コストとの結合で意味差が消えるため、単調変換前の raw logit を使います。LLM / Cross-Encoder とも、import、モデル欠落、初期化の失敗時は警告を標準エラーへ出してモックへフォールバックします。

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
python scripts/evaluate.py --suite all --reranker cross-encoder \
  --cross-encoder-model BAAI/bge-reranker-v2-m3 --json
python scripts/evaluate.py --suite all --reranker cross-encoder \
  --cross-encoder-model hotchpotch/japanese-reranker-base-v2 --json
```

レポートにはクエリ数、文脈付きケースを含む NDCG@k / MRR@k、Top-1 accuracy、カスケード対象率、実際の文脈リランク率、リランカの候補 1 件あたりレイテンシを出力します。レイテンシはモデルロードを除く各 `rerank()` 全体を候補数で割った値です。NDCG はゲイン $2^{relevance}-1$ と `log2(rank + 1)` の割引を使い、MRR は最初の関連候補の逆順位です。`preceding_context` と `following_context` は任意で、省略時は空文字列として扱います。

2026-08-21、辞書拡充前の自作ミニ辞書、Intel N100、CPU、`top_k=5`、`beam_width=64`、LLM コンテキスト長 256、`--suite all`、文脈リランク対象 5,272 候補での各 1 回の実測です。モデルロード時間は除外しています。

| リランカ | Handmade Top-1 | NDCG@5 | MRR@5 | AJIMEE CER | 平均編集距離 | 完全一致 | Mozc pass | 候補レイテンシ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mock | 1.000000 | 0.992972 | 1.000000 | 0.842278 | 17.725 | 0.010000 | 49/498 (0.098394) | 0.002 ms |
| llm | 0.928571 | 0.966609 | 0.964286 | 0.842917 | 17.730 | 0.010000 | 49/498 (0.098394) | 325.754 ms |
| BAAI/bge-reranker-v2-m3 | 1.000000 | 0.992972 | 1.000000 | **0.840914** | **17.700** | 0.010000 | 49/498 (0.098394) | 206.683 ms |
| hotchpotch/japanese-reranker-base-v2 | 1.000000 | 0.992972 | 1.000000 | 0.841172 | 17.715 | 0.010000 | 49/498 (0.098394) | **48.668 ms** |

Mozc には文脈がなく第2ランカーを実行しないため、4 方式の pass rate は同じです。BGE は AJIMEE の平均編集距離を日本語特化モデルより合計 3 文字だけ改善しましたが、候補レイテンシは 4.25 倍でした。日本語特化モデルは手作り精度と Mozc pass が BGE と同じで、LLM より 6.69 倍速いため、BeME の既定 Cross-Encoder に採用します。ただし 48.668 ms/候補は IME の候補群全体にはまだ遅く、実運用採用では候補数制限・量子化・ONNX 等の高速化が必要です。

### Phase 4: d1 文脈リランキング

2026-08-24、d1 dev 1,000 件だけで `viterbi_weight=8`・`rerank_k=10` を選択し、d1 test 2,000 件（contrast pair 128 件）を 1 回評価しました。局所モデルは Phase 3 の A0 seed 9、Cross-Encoder は `hotchpotch/japanese-reranker-base-v2` revision `f21b54987ac864d7a355f3384bf05d0939e31cb1` です。

「リランカ単独」は候補生成を省く条件ではありません。候補生成は局所モデルと同一で、順位付けだけを relevance にした `viterbi_weight=0`・`rerank_k=64` です。

| 条件 | d1 test 全体 Top-1 | contrast 128 Top-1 |
|---|---:|---:|
| 局所モデル単独（L） | .731000 | .328125 |
| リランカ単独（R） | .521000 | .421875 |
| 併用（LR） | **.831500** | **.531250** |

LR は seed 9/17/33 の全 seed で L より改善方向かつ McNemar `p<.05` でした。単一 seed の p 値や seed 間で合算した分割表では判定していません。`surface_recall@10` はリランク前後で全体 `.988000`、contrast `.960938` のままです。

Intel N100・4 core miniPC（host `WIN-I7DHB6T7L79`）の d1 test 先頭50件では、全候補の平均 `3.321963 s/変換` に対し、`rerank_k=10` は `.739371 s/変換`（77.74%削減）でした。`confidence<.25` は救済 recall 1.0 でも test fire rate `.576` だったため、停止規則に従ってcascade gateは配線していません。詳細、loss 54件、artifact hashは [`docs/missions/issue9-phase4/phase4-report.md`](docs/missions/issue9-phase4/phase4-report.md) を参照してください。

### IPADIC 辞書拡充と P0 の比較（mock）

同日、同じ `top_k=5`、`beam_width=64`、`--suite all --reranker mock` で再測定しました。P0 は手作り 14 件の順位品質を変えず、AJIMEE の macro CER を IPADIC 取り込み直後からさらに `0.051353`（15.04%）低下させ、Mozc も 7 ケース改善しました。

| 辞書 | エントリ | Handmade Top-1 | NDCG@5 | MRR@5 | AJIMEE CER | 完全一致 | Mozc pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| 拡充前・自作ミニ | 133 | 1.000000 | 0.992972 | 1.000000 | 0.842278 | 0.010000 | 49/498 (0.098394) |
| IPADIC + 手作り優先語（P0 前） | 349,043 | 1.000000 | 0.992972 | 1.000000 | 0.341433 | 0.060000 | 81/498 (0.162651) |
| IPADIC + 手作り優先語 + カタカナ `+1.0` | 349,043 | 1.000000 | 0.992972 | 1.000000 | **0.290080** | **0.070000** | **88/498 (0.176707)** |

「きしゃ」は `記者 > 汽車 > 帰社`、履歴文脈付き「こうしょう」は `考証 > 交渉 > 高尚`、商談の後方文脈付きでは `交渉 > 高尚 > 校章` を維持しました。全手作りケースの Top-1 `1.0` が優先語保護の回帰判定です。Mozc pass も P0 前の `81/498` から `88/498` へ上がり、AJIMEE で選んだ `+1.0` が少なくとも固定 Mozc スイートを悪化させないことを確認しました。

頻度バイアスを確認する文脈ケースの実順位:

| 手法 | 「歴史資料の内容を調べる」 | 後方「について商談を進める」 |
|---|---|---|
| mock | 考証 > 交渉 > 高尚 | 交渉 > 高尚 > 校章 |
| llm | **交渉** > 考証 > 鉱床（誤り） | 交渉 > 鉱床 > 考証 |
| BGE | 考証 > 公章 > 交渉 | 交渉 > 工章 > 公商 |
| 日本語特化 | 考証 > 校章 > 交渉 | 交渉 > 考証 > 工廠 |

両 Cross-Encoder は「歴史→考証」を Top-1 にして、LLM logits の「交渉」頻度バイアスをこのケースでは回避しました。一方、2 位以下には `公章`・`工章` のような不自然なゼロショット順位が残り、辞書コストとの結合は引き続き必要です。

### 外部評価データと判定方法

詳細な出所、固定 commit、SHA-256、ライセンス、データごとの判定規則は [`data/README.md`](data/README.md) を参照してください。

**AJIMEE-Bench** の `JWTD_v2/v1` は 200 件の全角カタカナ読みと複数の許容出力を持ちます（ローマ字入力ではありません）。全角カタカナをひらがなへ正規化し、句読点・数字・記号はリテラルとして保持します。リテラルで区切られたひらがな区間を BeME で変換し、区間の K-best を合成します。`context_text` は左文脈に使います。項目ごとに許容出力のうち最小の Levenshtein 編集距離と $\mathrm{CER}=\mathrm{distance}/|\mathrm{reference}|$ を採用し、平均 CER、平均編集距離、完全一致率を集計します。

**Mozc `oss.tsv`** は conversion 系 498 件を回帰判定し、BeME に存在しない suggestion 系 2 件を理由付きでスキップします。`Conversion Expected [N]` は正解全文の順位（`N` は 0 始まり）、`Conversion Match` は Top-1 の必須部分文字列、`Conversion Not Match` は Top-1 に含めてはいけない既知の誤変換として判定します。`--top-k` より深い期待順位は観測できないため、必要な深さを指定して評価してください。

ライセンスは upstream の一次情報に従います。

- AJIMEE-Bench `JWTD_v2/v1`: **CC BY-SA 3.0**。依頼時資料の MIT 記載とは異なり、upstream README がデータは元データに準拠して CC BY-SA 3.0 と明記しています。帰属: azooKey/AJIMEE-Bench contributors および日本語 Wikipedia 入力誤りデータセット v2。`data/licenses/AJIMEE-BENCH-CC-BY-SA-3.0.txt` を参照。
- Mozc `oss.tsv`: **BSD 3-Clause**、Copyright 2010-2018 Google Inc. `data/licenses/MOZC-BSD-3-Clause.txt` に著作権表示、条件、免責条項を収録。
- MeCab IPADIC 由来 `dictionary.tsv`: NAIST / ICOT の IPADIC ライセンス。固定ソースと変換規則は `data/README.md`、著作権表示・再配布条件・無保証条項は `data/licenses/IPADIC.txt` を参照。

## テスト

```sh
pytest -q
```

辞書検証、ラティス接続、未知語フォールバック、複数分割、候補の一意性と決定性、確信度の境界、CLI、リランカ契約、前文・後方文脈による順位変更、文脈付き NDCG/MRR に加え、IPADIC の読み・POS・コスト変換、重複排除、記号・不正読み除外、手作り優先語の上書き、全角・半角・混在 surface のカタカナ `+1.0` と overlay 非適用、カタカナ→ひらがな入力アダプタ、句読点保持、Levenshtein/CER、Mozc の正例・禁止誤変換・順位・スキップ規則、導入データ件数（200/500）、手作り14件の維持をテストします。LLM / Cross-Encoder 用テストは実モデルをロードせず、決定的な fake backend で尤度・関連度と Viterbi コストの結合、バッチ入力、`save_state` / `load_state`、ロード失敗時のフォールバックを検証します。

## Phase 2 学習

Phase 2 の d1 学習データはリポジトリ外から供給される必要があります。`data/d1/train.jsonl`（10,000件）、`dev.jsonl`（1,000件）、`test.jsonl`（2,000件）は Wikipedia `20231101.ja` を MeCab でかな化した記事 disjoint split です。d1 は gitignore 対象の外部データであり、存在しない場合に synthetic data で代用しないでください。詳細な実測値は [`docs/missions/issue9-phase2/phase2-report.md`](docs/missions/issue9-phase2/phase2-report.md) を参照してください。

固定条件で学習します。

```bash
uv run python scripts/train_scorer.py \
  --train data/d1/train.jsonl --dev data/d1/dev.jsonl --test data/d1/test.jsonl \
  --dictionary data/dictionary_v2.tsv \
  --initial-weights data/weights/decomposed_v1.json \
  --epochs 5 --learning-rate 0.1 --seed 9 \
  --output-weights artifacts/issue9-phase2/run1/weights.json \
  --output-report artifacts/issue9-phase2/run1/report.json
```

再現性は独立した出力先で実行し、weights と report を比較します。

```bash
# 上記コマンドの出力先を run2/weights.json と run2/report.json に変更
cmp artifacts/issue9-phase2/run1/weights.json artifacts/issue9-phase2/run2/weights.json
cmp artifacts/issue9-phase2/run1/report.json artifacts/issue9-phase2/run2/report.json
```

学習済み weights は別プロセスの既存評価へ明示的に渡します。実モデルは不要です。

```bash
uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv \
  --weights artifacts/issue9-phase2/run1/weights.json --json
```

## 制約と今後

- 既定辞書は IPADIC の活用形・固有名詞を含む 349,043 エントリです。NEologd の新語や接続コスト行列は未導入で、TSV ロード時の最大 RSS はクリーンな CLI 実測で約 156 MiB です。
- キーワードモックは小規模評価データに合わせた手作りルールです。
- N100 CPU の旧ミニ辞書測定では候補レイテンシは日本語 Cross-Encoder 約 49 ms、BGE 約 207 ms、0.5B LLM 約 326 msでした。d1・A0・長い文脈での Phase 4 実測は日本語 Cross-Encoder 71.747–73.454 ms/候補、全候補 3.321963 s/変換、`rerank_k=10` でも .739371 s/変換であり、cascade gateは安全な発火率を得られず未配線です。
- ビーム幅が小さすぎると、有望な経路を途中で枝刈りする可能性があります。
- 確信度しきい値 2.0、LLM/Viterbi/Cross-Encoder 重み 1.0 は初期値です。今後は独立したホールドアウトデータで調整します。

詳細な要件と Phase 1・Phase 2 の設計判断は `docs/spec.md` を参照してください。

## Phase 3 評価とアブレーション

Phase 3 は `decomposed_v1` を基準に、次の 12 条件を比較します。

`A0=decomposed_v1`、`A1=decomposed_v1-lex_only`、`A2=decomposed_v1-trans_only`、`A3=decomposed_v1-no_ipadic_cost`、`A4=decomposed_v1-no_source`、`A5=decomposed_v1-no_katakana`、`A6=decomposed_v1-no_shape`、`M0=decomposed_v1+matrix_v1`、`M1=decomposed_v1+matrix_v1-pos_pairs`、`F0=decomposed_v1+freq_v1`、`F1=decomposed_v1+freq_v1-no_ipadic_cost`、`X0=decomposed_v1+matrix_v1+freq_v1`。

補助測定とハーネスの確認:

```bash
uv run python scripts/build_ablation_weights.py --help
uv run python scripts/build_matrix_table.py --help
uv run python scripts/oracle_recall.py --help
uv run python scripts/diagnose_cases.py --help
uv run python scripts/audit_leakage.py --help
uv run python scripts/measure_cost_scale.py --help
uv run python scripts/run_ablations.py --help
```

IPADIC matrix table は `matrix.def` から一度だけ生成します。`ipadic` pip package に含まれるのは `matrix.bin` であり、`matrix.def` は含まれません。

GPU マシンには d1 の学習 split が repo 外にあるため、学習は `beme-gpu` で行います。clone 後に `data/d1/{train,dev,test}.jsonl` を GPU 側の d1 実体へ配置し、次を実行します。

```bash
uv run python scripts/run_ablations.py --conditions A0 --seeds 9 \
  --output-root artifacts/issue9-phase3
uv run python scripts/run_ablations.py --jobs 6 \
  --output-root artifacts/issue9-phase3
uv run python scripts/run_ablations.py --summarize-only \
  --output-root artifacts/issue9-phase3
```

詳細な実測値、未測定項目、A2/A3 の探索 collapse は [`docs/missions/issue9-phase3/phase3-report.md`](docs/missions/issue9-phase3/phase3-report.md) を参照してください。
