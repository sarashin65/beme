# S1: 既存コード調査メモ

調査対象は `scripts/build_d1_data.py`、`scripts/audit_leakage.py`、`scripts/compare_ajimee_segmentation.py`。E0 実装時に再利用できる責務と、E0 の protocol のために置き換えるべき責務を分離して記録する。

## `scripts/build_d1_data.py`

### 構造

- 定数: Wikipedia dataset/config/revision、d1 の split target、`_SENTENCE_BOUNDARY`、読み正規表現、正規化規則。
- 小さな共通処理: `normalize_text()`、`sha256_text()`、`article_split()`、`node_reading()`、`sentence_nodes()`、`iter_jsonl_articles()`、`iter_hf_articles()`。
- 候補生成: `article_groups()`。
- d1 特有の選択: `group_fingerprint()`、`balanced_select()`、`add_contrast_pairs()`、`add_frequency_buckets()`。
- 出力と orchestration: `write_jsonl()`、`build()`、CLI parser/main。

### article_groups と文分割

`article_groups()` は article ごとに `_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？])|[\r\n]+")` で分割し、各文を `strip()` して空文を除外する。`sentence_nodes()` が fugashi/MeCab node と surface の start/end offset を作る。各 node の MeCab feature 第8欄を `node_reading()` でカタカナからひらがなへ変換し、ひらがな・長音符だけの reading を採用する。

その後、start node から最大 `max_span_tokens`（d1 default 3）を走査する。ambiguous surface、単独 CJK surface、辞書に存在しない reading、reading 長超過を除外し、連結 reading を converter に渡す。converter の候補に positive surface があり、候補が2件以上ある場合だけ d1 group を作る。group には left/right context、reading、positive、候補 surface/cost、gold index、span token 数、article hash 等が入る。

### balanced_select

`balanced_select()` は reading ごとに group を束ね、random seed で並べた reading-balanced round robin を行う。先に同じ reading で異なる positive の contrast pair を予約し、`per_reading` 上限まで選ぶ。split 間の重複排除は `test -> dev -> train` の順で `group_fingerprint()` を使う。

これは E0 の文単位・長さ層・article 上限選択とは目的が異なるため、E0 から import/call しない。E0 は full sentence を候補単位にし、converter の候補順位・正解有無を selection feature にしない hash 順選択にする。

### E0 で再利用できるもの／変更が必要なもの

再利用候補は `Article`、`iter_jsonl_articles()`、固定 revision の `iter_hf_articles()`、`article_split()` の考え方、`sentence_nodes()`、`node_reading()`、`normalize_text()`、`sha256_text()` である。実装は S2 で共通 helper 化するのが安全で、既存 production code は変更しない。

変更が必要なのは、(a) 文全体を候補にする抽出、(b) d1 の局所 span/context ではなく source sentence と全文 gold を保存する schema、(c) `balanced_select()` の reading balancing をやめた length-stratum/hash selection、(d) article max 5・最低400記事・2,400候補から2,000採用の gate、(e) d1 candidate generation に依存しない gold path と annotation provenance、(f) source/d1/AJIMEE のリーク fingerprint である。

## `scripts/audit_leakage.py`

### 構造と共通化候補

- `_sha256(path)`: artifact hash 検証にそのまま使える。
- `_normalized(value)`: 現行は NFKC と whitespace 除去。E0 では NFKC、空白規則、casefold、12-gram 規則を protocol として明示し、d1 builder の `normalize_text()` との差を解消する必要がある。
- `split_for_article()`: manifest の seed/ranges から article split を再計算する関数。
- `audit_manifest()`: d1 manifest/data_gate の整合性検査。ただし d1 固有。
- `_d1_examples()`: d1 JSONL の存在確認と loader。E0 の fail-closed input check の参考になる。
- `audit_split_recomputation()`: article ID と split の照合。
- `_evaluation_material()`、`audit_evaluation_overlap()`: AJIMEE/Mozc/handmade の reading/surface/document material と 12-gram overlap を作る。
- `audit_mozc_claim()`: Mozc に source document がないことの検証。
- `_atomic_json()`: 監査 report の安全な出力。

### E0 のための変更点

現行 CLI は `--d1-dir --ajimee-cases --mozc-cases --cases --output` の d1 専用で、未配置 d1 を `unverifiable` として成功扱いにする。E0 build gate ではこれを許さず、d1 train/dev/test/articles の全 hash と件数が検証できなければ fail closed にする。

共通化は hash、NFKC/空白正規化、12-gram index、article ID/split 再計算、document fingerprint の helper として行う。E0 では d1 の train/dev/test context、全 d1 article ID、AJIMEE 原文/context、handmade 原文を forbidden index に入れ、候補抽出前と完成後の両方で検査する。reading/surface の共通は語彙共有として別集計し、document leakage と混同しない。

## `scripts/compare_ajimee_segmentation.py`

### 構造

`parse_args()` が dictionary、AJIMEE cases、baseline/treatment weights、top-k、beam width、cascade threshold、output を受け取る。`main()` は `Dictionary.from_tsv()`、`Weights.load()`、`extractor_for()`、`LinearScorer`、`Converter` を条件ごとに構築し、各 case を `analyze_case()` に渡す。

`analyze_case()` の流れは次のとおり。

1. `convert_sentence()` で入力読みを変換し、Top-1 hypothesis を得る。
2. `character_error_rate()` と `edit_distance()` で複数 reference から CER 最小（距離、reference order を tie-break）の reference を選ぶ。
3. `build_lattice(reading, converter.dictionary, allow_literals=True)` で入力 lattice を構築する。
4. `constrained_best_path(lattice, reference, converter.scorer)` と hypothesis に対する同関数で path を求める。
5. `boundaries(path)` が token reading の累積位置を内部境界として返し、gold/hypothesis の境界を比較する。
6. exact、gold 未到達、境界差、surface 差を `correct`/`coverage`/`segmentation`/`homophone` に分類する。

最後に case ID 単位で baseline/treatment の error transition、segmentation の resolved/induced/both/neither、summary を JSON に保存する。

### E0 での使い方と注意

`build_lattice()` は input reading から辞書 edge と literal edge を持つ lattice を作る。`allow_literals=True` は句読点・数字・未知 literal を同位置に通すため E0 に適する。`constrained_best_path()` は surface に制約された path のうち scorer による最良 path を返すため、既存 AJIMEE script の gold boundary は scorer 依存である。

E0 の gold boundary はこの関数で再選択しない。凍結 case の `gold_paths` を正とし、`build_lattice()` で reachability を検査する。評価条件の scorer によって同じ surface の tokenization を決めると、条件間で gold が変わり、分割評価が汚染される。hypothesis の公式 boundary も、可能なら converter が実際に選んだ token path を保存して比較する。surface から `constrained_best_path()` で後付け推定するだけの値は診断用に限定する。

AJIMEE adapter では既存の複数 references、CER、編集距離、paired 集計を再利用できる。ただし E0 protocol では AJIMEE case にも固定 `gold_paths` を付与し、Wikipedia core と pooled せず、層別 summary として別に出す。

## 再利用 API と変更一覧

| 目的 | 再利用候補 | E0 で必要な対応 |
|---|---|---|
| 記事入力 | `Article`、`iter_*_articles` | snapshot/hash/source provenance を保持する adapter |
| 文分割 | `_SENTENCE_BOUNDARY` | 完全な一文のみ、見出し/空文/表断片を除外し、規則を version 固定 |
| 読み抽出 | `sentence_nodes`、`node_reading` | MeCab は暫定 annotation とし、canonical gold は別 provenance |
| fingerprint | `sha256_text`、`_normalized`、`group_fingerprint` の考え方 | input/reference/source を区別し、E0 の NFKC・空白・casefold 規則を一つに固定 |
| リーク監査 | `audit_split_recomputation`、12-gram 処理 | unavailable を成功扱いせず fail closed、d1/AJIMEE/handmade を事前 index 化 |
| lattice | `build_lattice(..., allow_literals=True)` | reachability のみ利用。gold path は case artifact から読む |
| path/境界 | `boundaries`、`constrained_best_path` | scorer-dependent gold selection は禁止。実際の hypothesis path を評価出力に保存 |
| paired 集計 | compare script の per-case records、transition table | E0 の fixed case ID、article-cluster bootstrap、error flags を追加 |

変更対象は S2 以降の E0 用 script/adapter に限定する。`src/beme/` 以下の production code は変更しない。
