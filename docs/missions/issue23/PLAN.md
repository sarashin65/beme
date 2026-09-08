# issue #23: E0（分割が見える長文評価セット）構築計画

## 0. 目的と事前確認

E0 は、単語中心の d1 では観測できない複数 token 間の分割・接続選択を、十分な検出力を持つ文単位の paired 評価で測るための評価専用セットである。主指標は文完全一致率と CER、主要な診断指標は分割誤り率とする。E0 を学習、重み調整、閾値調整には使わない。

リポジトリを確認した結果、次の実装資産を再利用できる。

- `scripts/build_d1_data.py`: Wikipedia の固定 revision、記事 split、文分割、MeCab 読み抽出、正規化・fingerprint の実装がある。ただし `article_groups()` は局所 span を列挙し、`balanced_select()` は reading ごとの round-robin を行うため、E0 の選択処理には使わない。
- `scripts/audit_leakage.py`: NFKC・空白除去、12-gram による文書重複、d1 train 語彙との重複監査を持つ。ただし現 CLI と manifest 検証は d1 専用なので、共通関数を流用して E0 用入口を追加する必要がある。
- `src/beme/benchmark.py` と `scripts/evaluate.py`: AJIMEE の文変換、複数 reference、完全一致、編集距離、CER を実装済みである。
- `scripts/compare_ajimee_segmentation.py`: `build_lattice()`、`constrained_best_path()`、境界比較、paired 集計を実装済みである。現状は gold path が条件ごとの scorer に依存するので、E0 では固定した gold 境界を読むように変更する。
- AJIMEE は 200 件の全文 reference を持つ。一方、`data/mozc_oss.tsv` は 500 行（評価可能 498 件）だが、全文正解とは限らない `Match` / `Not Match` 等の assertion を含み、文脈や原文もない。

注意として、この checkout の `data/d1/manifest.json` は train/dev/test = 10,000/1,000/2,000 を記録している一方、依頼文は 12,000/2,000/2,000 としており、さらに manifest が hash を持つ `train.jsonl`、`dev.jsonl`、`test.jsonl`、`articles.jsonl` は checkout 内に存在しない。S1 の開始条件として、使用する d1 artifact 一式を復元し、どの版を正とするかを hash 付きで固定する。件数不一致を推測で埋めない。

## 1. E0 の設計

### 1.1 データソースの選択

E0 は一つの不均質な集合に既存 suite を連結するのではなく、次の層を持つ bundle とする。

| 層 | 目標件数 | 用途 | 主集計への扱い |
|---|---:|---|---|
| `e0-wikipedia-core` | 2,000 文 | 十分な規模の文単位評価、matrix 実験の主評価 | 主集計 |
| `e0-wikipedia-diagnostic` | 最大 450 文（各誤り型 150 を上限） | 少数誤り型の定性的・層別診断 | core と混ぜず別集計 |
| `e0-ajimee` | 200 文 | 既存結果との連続性、外部 domain での再現確認 | 副次集計 |
| Mozc 既存 suite | 評価可能 498 件 | quality regression | E0 の完全一致/CER には含めない |

主素材は Wikipedia `20231101.ja` の固定 revision `b04c8d1...` とする。新しい文章を執筆・収集するのではなく、d1 と同じ固定 snapshot から文を再取得する。優先順位は次のとおりである。

1. d1 artifact に本文 snapshot が保存されている場合はそれを使用する。
2. 保存されていなければ、manifest の dataset/config/revision から記事本文を再取得し、d1 の `article_id` / `article_sha256` と context 断片で同一性を検証する。
3. `left_context + positive + right_context` の重なりからの復元は候補生成と検証に使う。ただし context は片側 48 文字で切れているため、文頭・文末まで連続して復元できない文を「完全な文」と推定して採用しない。

d1 train/dev に属する記事は E0 から除外する。d1 test 記事も過去の開発判断で反復参照済みのため、原則として主 core には使わず、復元手順の検証材料にする。主 core は同じ固定 snapshot 内で、d1 の全記事 ID と AJIMEE の出典文書 fingerprint に重ならない記事から、`sha256("20260908:e0:" + article_id)` で決定的に選ぶ。この抽出は既存 corpus の held-out 化であり、新規コーパス収集ではない。

固定 revision の再取得が不可能で、d1 test 文だけで構築せざるを得ない場合は代替版を `e0-wikipedia-recovered` と明記する。その場合、d1 test との重複、記事クラスタ数、過去に評価へ使われた事実を manifest に残し、独立な主 benchmark とは扱わない。train/dev 記事を件数確保のために足すことは禁止する。

AJIMEE はライセンスと schema を保持したまま E0 adapter で読む。Wikipedia core との pooled 値は参考値に限定し、各層の値を必ず別に出す。Mozc は assertion の意味を全文 gold に変換できないため、件数を水増しする目的で統合しない。

### 1.2 文単位の抽出と選択

抽出単位は、固定 snapshot の記事本文を `scripts/build_d1_data.py` と同じ `_SENTENCE_BOUNDARY`（`。！？` の直後または改行）で切った完全な一文である。同一の文分割関数を共通化し、d1 と E0 で規則が乖離しないようにする。見出し断片、空文、表・箇条書きだけの行、reading を作れない文、正規化後の重複文は候補から除く。句読点、数字、記号は `allow_literals=True` で入力と gold の同位置に保持する。

`balanced_select()` は一切呼ばない。具体的には次の二段階で選ぶ。

1. 全候補文に article ID、sentence index、gold 読み、読み長、gold token 数を付与する。
2. 事前定義した長さ層ごとに `sha256(seed + article_id + sentence_index + normalized_surface)` の昇順で選ぶ。reading の種類や converter の候補順位では選ばない。同一記事は最大 5 文、core 全体で最低 400 記事を要求し、文を独立標本とみなす偏りを抑える。

選択前にリーク除外を実行し、選択後にも同じ監査を再実行する。baseline の出力や error type を見て core を入れ替えてはならない。error type を増やすための追加抽出は diagnostic 層だけで行い、core の指標へ混ぜない。

### 1.3 規模、層別キー、分布目標

core は QC 合格後ちょうど 2,000 文とし、最初に 2,400 文を候補化して修正不能・重複・監査不合格分の予備を持つ。AJIMEE を含む全文 gold 層は合計 2,200 文となる。規模だけでなく記事クラスタ数を品質条件にする。

`span_tokens` は「固定した canonical gold segmentation に含まれる変換 token 数」と定義し、句読点等の literal edge は数えない。複数の許容 gold segmentation がある場合は `gold_span_tokens` を配列で保存し、層別用 `span_tokens` は canonical path の値とする。`reading_length` はひらがな正規化後の Unicode code point 数で、literal も含む `input_length` とは分ける。

core の割当目標は次のとおりとする。これは reading ごとの均等化ではなく、長文を確実に含めるための sentence-level quota である。

| canonical `span_tokens` | 目標比率 | 最低件数 |
|---|---:|---:|
| 2--4 | 10% | 200 |
| 5--8 | 25% | 500 |
| 9--16 | 40% | 800 |
| 17 以上 | 25% | 500 |

追加 gate として、`span_tokens >= 5` を 90%以上、reading length 17 以上を 70%以上、33 以上を 25%以上とする。quota が固定 snapshot で満たせなければ閾値を黙って変えず、候補 pool の実分布と不足を報告して計画を改訂する。article 上限を緩めて埋めることもしない。

各 case に最低限、次を記録する。

- `id`, `source`, `license`, `article_id`, `article_sha256`, `sentence_index`, `source_revision`
- `input`（読み）、`references`（許容全文表記）、`canonical_reference`
- `gold_paths`（token ごとの reading/surface と入力上の境界）、`span_tokens`, `reading_length`, `input_length`
- `selection_stratum`, `selection_probability`, `annotation_status`, `annotator_count`
- `annotation_provenance`（provider/model snapshot、prompt/schema hash、raw response hash、checker/human audit/adjudication の履歴）
- `d1_anchor_ids`, `normalized_fingerprint`, `source_text_sha256`

誤り分類は case 固有の正解ラベルではなく、**モデル条件ごとの評価結果**である。このため case JSONL に将来のモデルの `error_type` を埋め込まず、評価 artifact に `condition`, `error_type`, `error_flags` を記録する。排他的な主分類は次の順で決める。

1. reference のいずれかに完全一致: `correct`
2. 許容 gold path が辞書/lattice に一つもない: `dictionary_missing`
3. gold は到達可能だが、予測 path の reading 境界が許容 gold paths のどれとも一致しない: `segmentation`
4. 境界は一致するが表記が異なる: `homophone`

混合誤りを失わないよう `error_flags` には `over_split`, `under_split`, `boundary_shift`, `surface_substitution`, `literal_mismatch` を複数記録できるようにする。主要3分類の「バランス」は core を baseline に合わせて歪めず、core 上の自然発生件数をそのまま報告する。事前固定した C0 baseline で core を一度評価した後、不足する型だけを未採用 pool から最大150件ずつ diagnostic 層へ hash 順に追加する。diagnostic は baseline-conditioned であることと sampling probability を明記し、型ごと最低100件を目標とする。100件に届かない型は low-support として統合・再分類しない。

### 1.4 gold の作成: LLM-as-annotator

Wikipedia の原文一文を canonical surface gold とする。LLM は原文を校閲・言い換えず、文脈中の読みと canonical な token 境界を付与する annotator としてのみ使う。NFKC 前の原文も provenance として保存し、評価用 reference の正規化規則を manifest に固定する。作成順は次のとおりとする。

1. 固定版 fugashi/MeCab + IPADIC で暫定の読み・境界を作る。記号・数字は literal として保持する。
2. pinned snapshot の Opus 級モデルへ、原文、MeCab 案、該当する d1 anchor、境界規約を渡して一次注釈させる。モデルは MeCab 案を追認するのではなく、文脈から読みを判定し、修正箇所と要確認理由も返す。
3. `build_lattice(input, dictionary_v2, allow_literals=True)` 上で原文 surface への到達性を動的計画法で確認する。canonical path は LLM 出力を規約と自動 validator に通したものとし、評価対象の scorer や weights は一切使わない。
4. 独立モデルによる選択的 cross-check と、後述の blind human audit を行う。不一致・規約違反・低信頼フラグは自動採択せず adjudication queue に送る。
5. 各 gold path の reading 連結が `input`、surface 連結が reference に完全一致することを自動検査する。辞書にない gold も除外せず `gold_reachable=false` として残し、辞書欠落を測れるようにする。

「単語境界」は一般的な日本語の語感ではなく、E0 の評価規約で定義する。原則は IPADIC-compatible な形態素単位（助詞・助動詞を分離）とし、辞書内の複合語を一 token とする別分割は、同じ読み・表記を保ち lattice 上で到達可能な場合だけ alternative path として許可する。prompt の few-shot 例に、複合名詞、活用、助数詞、数字・記号、カタカナ語、固有名詞、当て字を含める。規約と例は pilot 前に固定し、core のモデル精度を見て変更しない。

#### Prompt と出力 schema

system prompt は、(a) 原文を一文字も変更しない、(b) 文脈上の自然な読みをひらがなで出す、(c) 上記境界規約で token 化する、(d) 不明な固有名詞等を推測で確定せず `needs_review` にする、(e) JSON 以外を出さない、を明記する。user input は `case_id`, `source_sentence`, `mecab_tokens`, `d1_anchors` と規約 version を含める。出力は「読み -> 正解文 -> 単語境界」を機械的に照合できる次の形に固定する（説明文や chain-of-thought は要求・保存しない）。

```json
{
  "schema_version": "e0-annotation-v1",
  "case_id": "...",
  "input_reading": "ひらがなの全文読み。記号・数字は規約どおり保持",
  "canonical_reference": "入力の source_sentence と byte-for-byte 同一",
  "tokens": [
    {
      "surface": "...",
      "reading": "...",
      "surface_start": 0,
      "surface_end": 1,
      "reading_start": 0,
      "reading_end": 1,
      "is_literal": false
    }
  ],
  "alternative_paths": [],
  "corrections_from_mecab": ["reading", "boundary"],
  "risk_flags": ["proper_noun"],
  "needs_review": false,
  "review_reasons": []
}
```

JSON Schema は `additionalProperties=false`、必須 field、enum、型、非負 offset を強制する。さらに API schema の外側で、surface/token 連結、reading/token 連結、offset の連続性、ひらがな正規化、literal の同位置性を検証する。schema error や連結不整合時は同一 request を1回だけ再送し、それでも失敗すれば人手 queue に送る。

#### 決定性と provenance

- `temperature=0`、API が対応する場合は固定 `seed=20260908`、strict JSON Schema/Structured Outputs を使う。top-p 等の sampling parameter も manifest に固定する。
- model alias ではなく snapshot/model ID、provider、prompt 本文と hash、schema hash、request parameters、SDK/API version、request/response ID、usage tokens、raw response hash を case ごとに保存する。
- temperature 0 と seed は再現性を高めるが、provider 側更新や並列実行差まで含む byte-level 決定性を保証しない。最初に validator を通過した response を immutable raw artifact として凍結し、release 再構築では API を再実行せずその artifact を入力にする。
- pilot 50文を同じ設定で2回実行し、正規化後の完全一致率を記録する。不一致があれば nondeterministic field を特定し、snapshot が固定できないモデルは一次 annotator に採用しない。

#### モデルの役割

- **一次 annotator: Opus 級の pinned model**。50文の prompt 開発用集合と、human gold 100文の qualification pilot を通過した一つのモデル/versionだけを全件に用いる。第一候補を Opus とし、利用不能時は Astra を同じ pilot で再 qualification して置き換える。途中でモデルを混在させない。
- **独立 checker: Astra（一次が Opus の場合）**。異なる provider/model family を使うことに意味がある。`needs_review`、MeCab 修正、固有名詞、専門用語、数字、長さ上位10%、`gold_reachable=false`、d1 conflict の全件と、残りの10% hash sampleを blind に再注釈する。ただし二者一致は正しさの証明ではなく、二者不一致は自動で一次を覆さない。
- **Sol** は prompt/schema 作成、validator、差分分類、コストの低い dry-run に使えるが、最終 gold の一次判定には使わない。一次と checker が不一致の場合も Sol の多数決では決めず、人手 adjudication に送る。

同一モデルへの self-consistency 再質問は、temperature 0 では独立な証拠になりにくく、系統誤りも共有するため通常運用には入れない。全2,000文を Opus と Astra の両方に掛ける案は disagreement recall を上げるが、費用をほぼ倍増させても共通誤りを検出できないため、pilot で選択的 checker の見逃しが多い場合だけ採用する。

d1 の `positive` は局所 span の gold としてのみ流用する。E0 文中に d1 case の `left_context + positive + right_context` が一致したときは、その箇所の surface/reading を hard anchor として自動照合する。ただし d1 にない部分、文全体の読み、文境界、複数 gold segmentation は新規に作り、局所 gold の単純連結を全文 gold とみなさない。

### 1.5 human audit と品質 gate

pilot では、LLM 出力を見ずに人が注釈した100文（無作為50 + リスク層50）を qualification set とする。release audit では core から seed付き層化無作為300文と、リスク層から100文の計400文を抽出し、LLM 出力を隠して人が独立注釈する。pilot 100文は release audit と重複させない。監査者には辞書/規約と原文だけを提示し、MeCab/LLM の案を見せない。不一致は別の確認者または設計者が adjudicate する。

合否は surface、全文 reading、canonical boundary を別々に出し、主 gate を一文内の三者すべてが一致する **case exact agreement** とする。alternative path の差は、どちらも規約上許容なら不一致に数えず `gold_paths` の和集合候補として adjudicate する。

| gate | qualification 100文 | release audit 400文 |
|---|---:|---:|
| canonical surface | 100% | 100% |
| 全文 reading exact | 98%以上 | 98%以上 |
| canonical boundary exact | 98%以上 | 98%以上 |
| case exact agreement | 97%以上 | 97%以上 |
| boundary micro-F1 | 99%以上 | 99%以上 |
| リスク層の case exact | 95%以上 | 95%以上 |

各率には Wilson 95% CI と件数を併記し、専門用語、固有名詞、数字、reading length 上位10%、d1 anchor の層別値も出す。単一の point estimate だけで合格にしない。release gate は上表の point threshold をすべて満たし、case exact agreement の Wilson 95% CI 下限が95%以上、adjudication 後の未解決0件とする。

gate を一つでも下回った場合は、誤り taxonomy を作り、prompt/規約を version up して qualification からやり直す。同じ model/prompt での再質問だけで穴埋めしない。2回目も不合格なら、該当リスク層を全件人手確認する。全体または複数の主要層で不合格なら一次方式を破棄し、全2,000文の従来人手確認（4--7人日）へ戻す。監査結果を見て不都合な400文を差し替えることは禁止する。

d1 anchor と不一致の case は `d1_conflict` として常に人手 adjudication に送る。source revision/offset のずれ、正規化規約の差、d1 の局所 gold 誤り、E0 の誤りを区別し、判断根拠を log に残す。規約差が dataset-wide なら変換規則を先に固定して全件へ適用し、個別上書きで整合させない。解消不能なら当該 case を reject し、同じ stratum の次の hash 順候補で補充する。d1 自体は E0 構築の副作用として書き換えない。

## 2. 構築手順

### S1: 入力 artifact とプロトコルの凍結

- d1 JSONL、記事 manifest/snapshot を復元し、manifest 記載 hash と照合する。
- 10k/1k/2k と 12k/2k/2k のどちらを使うかを commit、SHA-256、件数で確定する。
- Wikipedia revision、AJIMEE、Mozc、辞書 v2、MeCab/IPADIC version、seed `20260908` を `data/e0/manifest.json` の入力欄へ固定する。
- annotation 規約、prompt/schema、一次/確認モデルの snapshot ID、sampling parameters、qualification/audit sample の抽出 seed と合否 gate を固定する。
- E0 を参照せず、schema、選択 quota、評価条件、合否 gate を先に commit する。

成果物（実装時）: `docs/missions/issue23/e0-protocol.md`, `data/e0/schema.json`。

### S2: リーク index の先行生成

- `scripts/audit_leakage.py` の正規化、12-gram、article split 再計算を共通関数化する。
- d1 train/dev/test の全 context/surface、既存評価 suite の原文/context、既採用記事 ID、正規化全文 fingerprint から forbidden index を作る。train/dev との文書重複は学習・選択リーク、test との重複は過去の評価への適合として区別して記録するが、独立 core ではどちらも blocking とする。
- 記事 ID 一致、全文一致、正規化12-gram一致を blocking leakage とする。reading/surface 語彙の一致は通常の語彙共有なので参考値に留める。
- d1 JSONL が欠ける、入力 hash が違う、監査不能項目がある場合は fail closed で候補抽出を止める。現行監査の「unverifiable でも成功」は E0 build gate には使わない。

成果物: `scripts/audit_e0_leakage.py` または後方互換な `scripts/audit_leakage.py --e0-*`、`data/e0/leakage-index.json`。

### S3: 文候補の復元・抽出

- `build_d1_data.py` から `Article`、固定 revision loader、文分割、`node_reading()`、normalization を再利用・共通化する。
- context stitcher は overlap が一意で、文頭・文末記号まで到達した場合だけ complete sentence を出す。再取得本文と照合できる場合は本文を正とし、断片は検算に使う。
- d1 全記事と forbidden index を除外した固定 Wikipedia から候補を作る。
- case ごとの provenance と pool 全体の除外理由件数を保存する。

成果物: `scripts/recover_e0_sentences.py`, `data/e0/candidates.jsonl`, `data/e0/recovery-report.json`。

### S4: 暫定 gold と deterministic selection

- MeCab の暫定読み、literal、暫定境界、辞書到達性を付ける。MeCab 出力は LLM への提案であって最終 gold ではない。
- converter の正解順位・誤り有無は selection feature に使わない。
- 長さ quota、記事最大5文、最低400記事を満たすよう hash 順で2,400候補を annotation queue に選ぶ。
- `balanced_select` が import/call されていないことを unit test と build manifest の `selection_algorithm` で保証する。

成果物: `scripts/build_e0_data.py`, `data/e0/annotation-queue.jsonl`。

### S5: LLM annotation、human audit、adjudication

- 100文の blind human gold で prompt/model qualification を行い、合格した pinned modelで2,400候補を一次注釈する。
- risk rule と hash sample に基づく独立モデル cross-check を行い、schema/validator 不合格、model disagreement、`needs_review`、d1 conflict を adjudication queue に送る。
- core 2,000文を確定後、別の400文を blind human audit し、§1.5 の gate を判定する。
- audit UI/TSV は human gold の提出前には LLM/MeCab 案を隠し、提出後の adjudication 画面で input、原文、両モデルの token 境界、d1 anchor、辞書到達性を並べる。
- rejected case は同じ stratum の次の hash 順候補で補充し、恣意的に選ばない。
- AJIMEE 200件にも固定 gold 境界を注釈し、条件依存の `constrained_best_path` 選択をなくす。ただし upstream の複数 reference は維持する。

成果物: `data/e0/wikipedia.jsonl` 2,000件、`data/e0/ajimee.jsonl` 200件、immutable raw LLM responses、annotation/audit/adjudication log、agreement report。

### S6: validator、最終リーク監査、manifest

`scripts/validate_e0_data.py` で以下を fail-closed に検証する。

- schema、件数、ID/fingerprint 一意性、記事数・記事上限、quota
- input/reference と全 gold path の連結整合性、reading length/span tokens の再計算一致
- source revision/hash、ライセンス/帰属、annotation 完了
- prompt/schema/model snapshot と raw response hash の provenance、qualification/audit gate 合格、未解決 queue 0件
- d1 train/dev/test、AJIMEE、handmade との document leakage が0件
- AJIMEE は自身の E0 層としてのみ既知重複を許可し、Wikipedia core との重複は0件
- 全 artifact の SHA-256 と build command を manifest に保存

候補生成前の監査と完成後の監査の両方が pass しなければ release しない。

### S7: d1 との差の品質 gate

d1 の実ファイルから `span_tokens` と reading length を再計算し、E0 core と比較する。大標本では p 値だけが小さくなるため、有意差と実質差の両方を要求する。

- two-sided permutation KS（10,000回、seed `20260908`）を `span_tokens` と reading length に実施し、Holm 補正後 `p < 0.01`。
- 効果量として Cliff's delta と中央値比を出す。
- material gate: E0 の median reading length 16以上かつ d1 の2倍以上、median span tokens 5以上、single-token 5%以下。
- 既知値 d1 single-token 86.5% は照合用であり、実 artifact から再計算した値を公式値にする。

いずれかを満たさなければ「d1 と構造的に異なる」とは認定せず、quota または抽出規則を評価結果を見る前に改訂して再構築する。モデル精度を見て再選択しない。

### S8: release rehearsal と凍結

- 同一入力・seed から byte-identical な JSONL/manifest ができることを clean environment で2回確認する。
- smoke model で全件を走らせ、空 hypothesis、例外、極端な runtime、分類不能が0件であることを確認する。smoke の精度は採否や選択に使わない。
- `data/e0/` の version、release date、既知制約を記録し、以後の変更は version を上げる。

## 3. 評価方法

### 3.1 指標

各条件で case 単位に hypothesis、最小編集距離、最小 CER、完全一致、gold reachability、予測境界、排他的 error type、詳細 flags を保存する。現行 `SentenceCandidate` は surface/cost しか保持しないため、評価専用 adapter で converter が実際に選んだ token path と境界も保存する。hypothesis の surface から `constrained_best_path()` で経路を再推定すると、同じ surface に複数 path がある場合に scorer 依存で分類が変わるため、予測境界の公式値には使わない。

- **文完全一致率**: Top-1 が許容 reference のいずれかと一致した文の割合。
- **Mean CER**: 各文で許容 reference 中の最小 CER を取り、その macro mean。micro CER（総編集距離/総 reference 文字数）も補助値として出す。
- **分割誤り率**: 全文を分母とした `error_type == segmentation` の割合。誤りだけを分母とする構成比も併記するが、主判定には使わない。
- **同音誤り率、辞書欠落率**: 全文を共通分母として報告する。
- **層別値**: `span_tokens`、reading length、source、gold reachability、固有名詞/数字 flags ごとに完全一致・CER・3誤り率を出す。30件未満は low-support と表示する。

複数 reference の CER 最小 reference と境界判定の reference を別々に選ばない。まず `(CER, edit distance, upstream order)` で reference を固定し、その reference に属する許容 gold paths と境界比較する。gold path は dataset で固定し、評価条件の scorer で選び直さない。

不確実性は 10,000回の paired cluster bootstrap（article ID 単位、seed `20260908`）で、完全一致率差、mean CER差、各誤り率差の95% percentile CIを出す。AJIMEE は article ID がないため case 単位 bootstrap とする。完全一致と分割誤りの paired 2値比較には McNemar 検定も出す。主要検定は core の分割誤り1本とし、完全一致/CER/層別は副次解析として効果量とCIを重視する。

### 3.2 matrix 実験の再評価

E0 release hash を固定してから issue #22 と同じ条件を再実行する。

| 条件 | weights | 固定差分 |
|---|---|---|
| C0 | `decomposed_v1-no_script-surface_len+corpus_v1.json` | 対照 |
| C0+matrix | `decomposed_v1-no_script-surface_len+corpus_v1+matrix_v1.json` | `trans.matrix_cost = 0.5` の追加 |

辞書 `dictionary_v2.tsv`、`top_k=5`、`beam_width=64`、`cascade_threshold=2.0`、reranker `mock` を固定し、weight sweep や E0 上の再調整は行わない。

1. E0 core 2,000文を両条件で走らせ、case ID、reference、実効設定、入力 hash が完全一致することを assert する。
2. AJIMEE 200件も同じ2条件で再実行し、issue #22 の結果が再現することを確認する。
3. `resolved`（C0のみ分割誤り）、`induced`（matrixのみ分割誤り）、both、neither の2×2表を作る。
4. `resolved/induced` に two-sided McNemar、article-cluster paired bootstrap に分割誤り率差のCIを適用する。
5. 完全一致率差、mean CER差、辞書欠落率差を guardrail とし、error type 遷移と `span_tokens` 層別で機構を確認する。

事前判定は、(a) core の分割誤り率が低下、(b) `resolved > induced` かつ McNemar `p < 0.05`、(c) cluster bootstrap 95% CI 上限が0未満、(d) 完全一致と mean CER の点推定が悪化せず辞書欠落が増えない、の全条件を満たせば Pass とする。方向は改善だが有意でない場合は Inconclusive、分割は改善しても guardrail を破る場合は Trade-off とする。AJIMEE は外部再現性の証拠であり、core の失敗を相殺して pooled Pass にしない。

S1 で、既知の AJIMEE 差（91→86）と利用可能ならその discordant 表を用い、2,000文・400記事、複数の級内相関を仮定した McNemar/cluster bootstrap の感度曲線を事前に作る。E0 の結果を見て sample size や選択を変えない。2,000件でも記事内相関や discordant 数が少なければ検出力は保証されないため、有意でない結果を「効果なし」と断定しない。

## 4. 見積もり工数

| タスク | 工数 | 備考 |
|---|---:|---|
| S1 入力復元・protocol/schema 凍結 | 0.5--1人日 | d1 版の不一致解消を含む |
| S2 リーク監査の一般化・先行 index | 1--1.5人日 | fail-closed 化とテストを含む |
| S3 文復元・候補抽出 | 1--2人日 | snapshot 再取得可否で変動 |
| S4 暫定 gold・選択器 | 1.5--2人日 | 暫定 path と quota test が中心 |
| S5 LLM prompt/schema・runner 設計 | 1--1.5人日 | qualification、raw response/provenance を含む |
| S5 blind human audit | 0.75--1.5人日 | pilot 100文 + release 400文、重複なし |
| S5 disagreement adjudication | 0.5--1.5人日 | 10--20% disagreement を想定。超過時は別途 |
| S6 validator・最終監査・manifest | 1--1.5人日 | schema/property test を含む |
| S7 分布検定・品質 gate | 0.5--1人日 | artifact とレポート生成 |
| S8 release rehearsal | 0.5人日 | clean 再現実行 |
| 評価器拡張・matrix 再評価 | 1--1.5人日 | 実験計算時間を除く作業時間 |
| **合計** | **9.25--15.5人日** | **約2--3人週** |

従来の「全2,000文の人手一次確認」4--7人日はなくなり、blind audit 500文と disagreement adjudication の **1.25--3人日**へ縮小する。したがって人手注釈部分だけで概ね2.75--5.75人日を削減する。ただし品質 gate 不合格時にはリスク層の全件確認、さらに必要なら従来方式の4--7人日へ戻るため、これは合格時の見積もりである。

API 費用は実行直前の価格表と実測 token 数で再計算し、USD と価格取得日を manifest に残す。計画時の比較用仮定を **1 call/case = input 1,200 tokens、output 600 tokens、2,000文** とすると、1 full pass は input 2.4M / output 1.2M tokens である。2026-09-08 時点の [Anthropic Opus 価格](https://www.anthropic.com/claude/opus) と [OpenAI model comparison](https://developers.openai.com/api/docs/models/compare) にある通常価格の例では次の規模になる（prompt caching、Batch 割引、税、retry は除外）。

| full pass | input / output 単価（$/1M tokens） | 2,000文の概算 |
|---|---:|---:|
| Opus | 5 / 25 | 約 $42 |
| Sol | 4 / 20 | 約 $34 |
| Astra | 10 / 50 | 約 $84 |

推奨構成は、2,000文だけなら Opus 一次 full pass 約$42に、Astra のリスク層 + 10% sample（全体の30%と仮定）約$25を加え、retry 用20%余裕を含め **約$80** である。実際の構築では予備を含む2,400候補と AJIMEE 200文も対象になるため、同じ仮定なら上限枠を **約$110** とする。2,000文全件を Opus + Astra の2 passにすると約$126、さらに Sol も全件に掛ける3 passは約$160、20%余裕込みで約$190である。実際には日本語 tokenization、few-shot 長、alternative path 数で変動するため、pilot 50文の usage から `N * (Tin*Pin + Tout*Pout) / 1,000,000` で再見積もりし、上限予算を超えたら件数ではなく checker 対象率を事前規則の範囲で調整する。

機械処理の wall-clock は Wikipedia streaming、MeCab 解析、LLM batch、2条件×2,200文の lattice 探索に依存するため別枠とする。LLM は MeCab 単独より強いが gold oracle ではないため、raw response の凍結、自動整合検査、独立モデルの選択的 cross-check、blind human audit の四層をすべて release 条件とする。

## 5. LLM annotation 固有のリスク

| リスク | 検出 | 扱い |
|---|---|---|
| 長文で token を脱落・重複、offset をずらす | 連結/offset validator、長さ上位10% flag | 自動再送1回、その後は人手。長文層の agreement を別掲 |
| 専門用語・古語・当て字の読みをもっともらしく捏造 | MeCab 差分、Wikipedia 文脈、`needs_review`、独立 checker | 専門語 flag 全件を checker、対立時は人手。根拠のない別読みを追加しない |
| 人名・地名・作品名等の固有名詞の読み違い | 固有名詞 flag、d1 anchor、model disagreement | 人手 adjudication。原文だけで確定不能なら case reject。Web 検索結果を ad hoc gold にしない |
| 数字、日付、単位、英字、記号の読み/ literal 方針が揺れる | 規約例、literal validator、数字 flag | dataset-wide 規則で統一し、個別最適化しない |
| MeCab 案への anchoring bias | blind human audit、一次モデルに「誤り得る案」と明記 | pilot で MeCab 修正 recall を確認。必要なら checker には MeCab 案を見せない |
| 二モデルが同じ系統誤りをする | blind human audit | model agreement を正解率と呼ばない。human gate は省略しない |
| model/API 更新で再現しない | snapshot ID、raw artifact/hash、再実行一致 pilot | release は raw response から再構築。モデル変更は E0 version を上げ再 qualification |
| d1 の局所 gold と不整合 | exact offset anchor check | `d1_conflict` を人手 adjudicate。規約差は全件規則化し、無解決 case は同層から補充 |

## 6. 完了条件

- Wikipedia core が2,000文、400記事以上、同一記事5文以下で、全文 gold と固定境界を持つ。
- AJIMEE 200文は別層、Mozc は別 suite として provenance を保つ。
- `balanced_select` を通らず、sentence-level quota と hash だけで再現可能に選ばれている。
- pinned Opus級モデルの全件 annotation、選択的独立 checker、blind human audit 500文が完了し、§1.5 の全 agreement gate を満たし、未解決0件である。
- temperature/seed/schema だけで再現性を主張せず、prompt/model/API provenance と immutable raw response hash から注釈を追跡できる。
- 事前・事後リーク監査、schema validator、d1との差の統計・material gate がすべて pass する。
- 完全一致、CER、分割/同音/辞書欠落を case 単位に再計算でき、article-cluster paired 比較ができる。
- 入力・出力 hash、実行コマンド、version、ライセンス、既知制約を manifest から追跡できる。
