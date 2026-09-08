# E0 case schema v1（凍結）

この文書は E0 の case JSONL の protocol/schema を S1 で凍結する。case は一文と、その文を読みから生成するための固定 gold path を表す。モデル条件ごとの hypothesis や error type は case に保存せず、評価 artifact に分離する。

## 完全な case 例

```json
{
  "schema_version": "e0-case-v1",
  "id": "e0-wikipedia-core-000001",
  "source": "wikipedia",
  "license": "CC BY-SA 3.0 and GFDL",
  "article_id": "12345",
  "article_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "source_revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
  "sentence_index": 7,
  "source_text_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "input": "けんきゅうをおこなった。",
  "references": ["研究を行った。"],
  "canonical_reference": "研究を行った。",
  "gold_paths": [
    {
      "path_id": "canonical",
      "is_canonical": true,
      "tokens": [
        {"reading": "けんきゅう", "surface": "研究", "reading_start": 0, "reading_end": 5, "surface_start": 0, "surface_end": 2, "is_literal": false},
        {"reading": "を", "surface": "を", "reading_start": 5, "reading_end": 6, "surface_start": 2, "surface_end": 3, "is_literal": false},
        {"reading": "おこなっ", "surface": "行っ", "reading_start": 6, "reading_end": 10, "surface_start": 3, "surface_end": 5, "is_literal": false},
        {"reading": "た", "surface": "た", "reading_start": 10, "reading_end": 11, "surface_start": 5, "surface_end": 6, "is_literal": false},
        {"reading": "。", "surface": "。", "reading_start": 11, "reading_end": 12, "surface_start": 6, "surface_end": 7, "is_literal": true}
      ]
    }
  ],
  "span_tokens": 4,
  "reading_length": 12,
  "input_length": 12,
  "selection_stratum": "span_2_4",
  "selection_probability": 0.001,
  "annotation_status": "adjudicated",
  "annotator_count": 2,
  "d1_anchor_ids": [],
  "normalized_fingerprint": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "annotation_provenance": {
    "provider": "pinned-provider",
    "model_snapshot": "pinned-model-snapshot-id",
    "prompt_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    "schema_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    "request_parameters": {"temperature": 0, "seed": 20260908, "top_p": 1},
    "raw_response_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    "response_id": "response-opaque-id",
    "checker": {"provider": "independent-provider", "model_snapshot": "checker-snapshot", "raw_response_sha256": "1111111111111111111111111111111111111111111111111111111111111111", "status": "agree"},
    "human_audit": [{"auditor_id": "auditor-opaque-id", "audit_id": "audit-0001", "status": "agree", "submitted_at": "2026-09-08"}],
    "adjudication": {"required": false, "status": "not_required", "history": []}
  }
}
```

## フィールド規則

| field | 型 | 必須 | 規則 |
|---|---|---:|---|
| `schema_version` | string | yes | 固定値 `e0-case-v1`。変更時は v2。 |
| `id` | string | yes | bundle 内で一意。変更不能な case ID。 |
| `source`, `license` | string | yes | `source` は `wikipedia` または `ajimee`。source に対応するライセンスを保存。 |
| `article_id`, `article_sha256`, `source_revision` | string | yes | Wikipedia は必須。AJIMEE は `article_id` を case/document ID として保持し、欠損を空文字で隠さない。hash は SHA-256 hex。revision は入力 snapshot。 |
| `sentence_index` | non-negative integer | yes | 固定 snapshot 内の文番号。 |
| `source_text_sha256` | string | yes | 正規化前の原文一文の UTF-8 SHA-256。 |
| `input` | string | yes | 評価入力の全文読み。ひらがなを基本とし、literal は規約どおり保持。空文字不可。 |
| `references` | array<string> | yes | 許容される全文表記。1件以上、重複不可。 |
| `canonical_reference` | string | yes | `references` の要素。原文 surface を変更しない canonical。 |
| `gold_paths` | array<object> | yes | 1件以上。canonical path はちょうど1件。代替 segmentation を許す場合のみ複数。 |
| `span_tokens` | non-negative integer | yes | canonical path の非-literal token 数。 |
| `reading_length`, `input_length` | non-negative integer | yes | Unicode code point 数。`reading_length` は gold reading 連結、`input_length` は `input`。 |
| `selection_stratum` | string | yes | S1 で固定した quota 層（例: `span_2_4`, `span_5_8`, `span_9_16`, `span_17_plus`）。 |
| `selection_probability` | number | yes | 0 より大きく 1 以下。hash 順 deterministic selection の事前値。 |
| `annotation_status` | enum string | yes | `annotated`, `checked`, `audited`, `adjudicated`, `rejected`。release case は `adjudicated` または監査不要の `checked`。 |
| `annotator_count` | non-negative integer | yes | 一次 annotator、checker、人手 adjudicator の実際の人数。 |
| `d1_anchor_ids` | array<string> | yes | 対応する d1 ID。なければ空配列。 |
| `normalized_fingerprint` | string | yes | 文書リーク監査用の正規化 fingerprint、SHA-256 hex。 |
| `annotation_provenance` | object | yes | 下記構造。raw response hash と履歴を欠落させない。 |

追加 field は v1 では禁止する（JSON Schema `additionalProperties: false`）。任意値は optional にせず、該当しない場合は `[]`、`false`、`null` など schema が指定する値で明示する。

## `gold_paths` の構造

`gold_paths` は canonical/alternative ごとの path object の配列である。各 path は `path_id`、`is_canonical`、`tokens` を持つ。`tokens` の各 token は次を持つ。

- `reading`, `surface`: それぞれ input/reference 上の連続 substring。
- `reading_start`, `reading_end`: `input` の code-point offset、半開区間 `[start,end)`。
- `surface_start`, `surface_end`: 該当 reference の code-point offset、半開区間 `[start,end)`。
- `is_literal`: 辞書変換 token でない句読点・数字・記号等は `true`。

token は reading/surface の順序で連続し、offset は非重複でなければならない。literal edge は `span_tokens` に数えない。canonical path の surface 連結は `canonical_reference`、alternative path の surface 連結は `references` のいずれかに一致させる。

## `annotation_provenance` の凍結構造

必須の provenance は `provider`、`model_snapshot`、`prompt_sha256`、`schema_sha256`、`request_parameters`、`raw_response_sha256`、`response_id`、`checker`、`human_audit`、`adjudication` である。

- `provider` / `model_snapshot`: alias ではなく provider 名と固定 model ID/snapshot。
- `prompt_sha256` / `schema_sha256`: 実際に送った prompt 本文と strict output schema の SHA-256。
- `request_parameters`: temperature、seed、top_p 等と SDK/API version を含む。temperature は 0、seed は `20260908` を標準値とする。
- `raw_response_sha256`: JSON の再整形後ではなく immutable raw response bytes の hash。raw response artifact 自体は case ID と hash で外部保存する。
- `checker`: 独立 checker の provider/model snapshot、raw response hash、`agree`/`disagree`/`not_run` の status。
- `human_audit`: auditor ID を匿名化した audit ID、status、日時の履歴配列。LLM 出力を隠した blind audit かどうかは audit log 側でも記録する。
- `adjudication`: `required`、`status`、`history`。history の各 event は actor、reason、before/after hash、timestamp を持ち、未解決は `open` のまま release しない。

## バリデーション規則

schema validation に加え、次を fail closed で検証する。

1. ID、fingerprint、path ID は bundle 内で一意。hash は lowercase SHA-256 hex。
2. `input_length == len(input)`、`reading_length == len(input)`。Unicode code point 単位で再計算し、byte offset と混同しない。
3. 各 path の token の reading 連結が `input` と完全一致する。
4. canonical path の surface 連結が `canonical_reference` と完全一致する。各 reference は少なくとも1つの gold path の surface 連結に一致する。
5. token offset は `0 <= start <= end`、path 内で連続し、最後の end は対応する input/reference 長と一致する。
6. `canonical_reference in references`、canonical path は1件、`is_canonical=true` の path は1件。
7. literal token は規約上の literal 位置でのみ許可し、literal でない token の reading は正規化済みひらがな規則に適合させる。
8. `span_tokens` は canonical path から `is_literal=false` の token 数を再計算した値と一致する。
9. `reading_length` と `input_length`、`source_text_sha256`、`normalized_fingerprint` を再計算して一致させる。
10. `gold_paths` の各 path は lattice 上で検証可能かを `gold_reachable` 相当の評価 artifact に記録する。辞書未到達でも case 自体は捨てず、辞書欠落診断に残す。
11. `annotation_status` が release 状態なら、raw response hash、prompt/schema hash、checker/human audit/adjudication の未解決状態を確認する。
12. case JSON に model 条件別の `hypothesis`、`error_type`、scorer、weight は入れない。これらは評価 run の出力であり、gold を汚染するため禁止する。
