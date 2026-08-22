# Issue #9 Phase 1（学習可能なスコアラー基盤）実装プラン

作成日: 2026-08-22
対象: `feat/issue9-learned-scorer` @ `4ba97ad`（main と同内容）
状態: 実装未着手。本書はプランのみ。
スコープ: [Issue #9](https://github.com/sarashinanikki/beme/issues/9) の **Phase 1 のみ**。Phase 2（正解経路・structured perceptron）は非スコープだが、接続点は本書で確定させる。

---

## 0. 結論

1. **Phase 1 の本質は「精度を上げること」ではなく「現行の目的関数を、値を 1 ビットも変えずに `特徴量 × 重み` へ書き換えること」**である。精度が動いたら Phase 1 は失敗とみなす。判定は bit-identical な golden 比較で行う（§4）。
2. 現行のコスト計算は `lattice.py` の **4 箇所**に散っている（`k_best_viterbi` / `backward_costs_by_pos` / `backward_viterbi` / `_paths_by_node`）。Scorer を導入しても 1 箇所でも直結のままだと再現テストが通っても探索が壊れるため、4 箇所すべてを同一 Scorer 経由にすることを完了条件にする。
3. 現行の探索状態は `previous_pos`（POS 文字列）である。Phase 2 で `ψ` に left/right ID や活用形を入れると状態が変わるため、Phase 1 の時点で **状態を `scorer.state_key(edge) -> Hashable` に一般化**し、後方 DP の状態領域を「その位置へ入る edge の状態集合」に限定する。これをやらないと Phase 2 で後方 DP が爆発する。
4. **left_id / right_id / 活用形 / 詳細POS / 辞書出所 / IPADIC 原コストは、現在ランタイムに一切存在しない**。`data/dictionary.tsv` は 4 列（reading/surface/pos/cost）で、IPADIC 原コストは `map_ipadic_cost()` とカタカナ `+1.0` を適用済みの単一 float に潰されている。Phase 1 のタスク1件目は辞書スキーマの拡張であり、IPADIC 原典アーカイブ（12.2 MB・SHA-256 固定・リポジトリ非同梱）の再取得が前提になる。
5. 重みファイルは **JSON・バージョン付き・`feature_spec` id 付き**とし、`data/weights/legacy_v1.json` をリポジトリへ同梱する。gitignore された生成物（`data/dictionary_learned_v2.tsv`）が Phase 0 baseline の再現性を壊した実例があるため（§2-C）、Phase 1 の成果物は同梱側に倒す。

---

## 1. コードから確認した現状（実測・引用付き）

### 1.1 現行の目的関数の実体

経路コストは次の 1 本の式しかない（`src/beme/lattice.py:357`, `:409`）。

```text
path.cost = Σ token.cost + Σ pos_transition_cost(previous_pos, current_pos)
```

- `token.cost` = 辞書 TSV の `cost` 列そのまま（`Token.from_entry`・`lattice.py:31`）。
- `pos_transition_cost`（`lattice.py:225-239`）は次の 4 分岐。
  1. `previous_pos is None`（BOS） → `0.0`
  2. `_TRANSITION_COSTS` に (prev, cur) がある → 表の値（**実測 13 ペア**）
  3. `previous_pos == current_pos` → `0.08`
  4. それ以外 → `0.12`
- 分岐 2 が分岐 3 より先に評価される。`("unknown","unknown"): 0.20` は同POSだが表の値が勝つ。**再現時にこの優先順位を落とすと値が変わる。**
- ラティス生成側の固定値: `UNKNOWN_COST = 8.0`（`lattice.py:17`）、非ひらがな literal edge は `Token(char, char, "literal", 0.0)`（`lattice.py:171`）。`"literal"` は `_TRANSITION_COSTS` に一切現れないので常に 0.08 / 0.12 側へ落ちる。

### 1.2 辞書側に焼き込まれている補正

`src/beme/ipadic.py` のビルド時（= TSV 生成時）に、次が `cost` 列へ不可逆に合成される。

| 補正 | 実装 | 備考 |
|---|---|---|
| IPADIC 原コストの線形写像 | `map_ipadic_cost()`・`ipadic.py:101-122` | `1.75 + 3.50 × clamp((c−3000)/7000, 0, 1)`、`round(…, 3)` |
| カタカナ表層ペナルティ | `KATAKANA_SURFACE_COST_PENALTY = 1.0`・`ipadic.py:203-204` | NFKC 後にカタカナを含む IPADIC 由来 entry のみ |
| 手作り overlay | `build_dictionary(overlays=…)`・`ipadic.py:240-244` | `data/dictionary_manual.tsv`（133 語）で上書き。floor 1.75 より低い値を置くことで優先度を保証 |
| 同一キーの重複統合 | `ipadic.py:205-211` | `(reading, surface, pos)` 単位で最小コストを採用 |

**IPADIC CSV の 13 フィールドのうち、TSV へ残るのは 表層(0) / 品詞(4-6→16ラベルへ集約) / 読み(11) / コスト(3の写像後) のみ**。左文脈ID(1)・右文脈ID(2)・活用型(8)・活用形(9)・原形(10)・詳細POS 3 階層の原文字列は捨てられている。`_archive_rows`（`ipadic.py:131-145`）は `.csv` で終わるメンバーしか読まないため、`matrix.def` も読んでいない。

実測（`data/dictionary.tsv`）: 349,043 entries / 14,583,715 bytes / POS ラベル 16 種 / cost 範囲 0.350–6.250。

### 1.3 探索の実体

| 関数 | 役割 | コスト直結箇所 |
|---|---|---|
| `k_best_viterbi`（`lattice.py:365`） | 本番の exact K-best（best-first 探索） | `:409` |
| `backward_costs_by_pos`（`lattice.py:293`） | 上のヒューリスティック `h`。**厳密な残余コスト** | `:324-326` |
| `backward_viterbi`（`lattice.py:242`） | 位置単位の残余コスト（`Lattice.backward_costs`・テストが直接参照） | `:272`, `:278` |
| `_paths_by_node`（`lattice.py:345`） | `Lattice.paths_at()` 用のビーム探索。別実装で同じ式を持つ | `:357` |

補足 2 点（Phase 2 の設計に効く）:

- `k_best_viterbi` のコメント（`:370-373`）は「全コストが非負だから admissible」と書いているが、実際に使っている `h` は **厳密な残余コスト**である。厳密 `h` は consistent なので `f(child) ≥ f(parent)` が常に成り立ち、**学習で負の重み・負の遷移コストが出ても best-first の順序保証は壊れない**（負コストで壊れるのは「非負だから admissible」という論拠の方だけ）。ただし `Dictionary.from_tsv` は `cost < 0` を弾く（`dictionary.py:115-118`）ので、負の値を扱うなら別の入り口が要る。
- 決定性は `_edge_key`（`:132`）と `_path_key`（`:337`）の tie-break に依存し、両方 `token.cost` を含む。Token へフィールドを増やすときは、この 2 つのキー定義を**変えない**（変えると同コスト時の順位が動き、golden が落ちる）。

### 1.4 コストのスケールに依存している下流

学習で cost のスケールが変わると、次が黙って挙動を変える。Phase 1（固定重み）では動かないが、Phase 2/3 で必ず再校正が要る。

- `Converter.convert` の `confidence = c[1].cost - c[0].cost` と `cascade_threshold = 2.0`（`converter.py:167-169`）
- `MockContextReranker` / `LLMLogitsReranker` / `CrossEncoderReranker` の `viterbi_weight * candidate.cost`（`reranker.py:293`, `:433`）
- `benchmark.py` の文合成（`SentenceCandidate(candidate.surface, candidate.cost)`・`:113`）でスパン間のコストを単純加算

### 1.5 現行 main の実測ベースライン（本作業で再取得・2026-08-22）

`data/dictionary.tsv`（リポジトリ同梱・SHA-256 `efcdf5a1…`）で `uv run python scripts/evaluate.py --suite all --reranker mock` を実行:

```text
Handmade: 14 cases, NDCG@5=0.992972, MRR@5=1.000000, Top-1=1.000000
AJIMEE:   200 cases, mean CER=0.290080, exact=0.070000
Mozc:     88/498 passed (0.176707), 2 skipped
実行時間: 6.5 s（71 tests は 26 s）
```

**この 4 値が Phase 1 の不変条件**である（§4.3）。

---

## 2. Issue 本文の「見立て」とコードのズレ

いずれも実際にコード／実行で確認した。方針そのものは妥当で、修正が必要なのは前提の粒度だけ。

### A. 「IPADIC 由来の単語生起コスト」は、ランタイムには存在しない

Issue は現行目的関数の構成要素として「IPADIC 由来の単語生起コスト」を挙げ、特徴量候補にも `IPADIC word cost` と `現行BeME cost` を**別々に**並べている。しかし §1.2 の通り、IPADIC 原コストは TSV 生成時に写像・クリップ・丸め・カタカナ加算・overlay 上書き・重複統合を経た**単一 float**へ潰れている。したがって、

- 「IPADIC word cost あり/なし」の Phase 3 アブレーションは、**辞書を作り直さない限り実行できない**。
- カタカナ `+1.0` を「独立ルールとして増やし続ける設計を置き換える」には、`cost` 列ではなく **`raw_cost` 列を特徴量に使う**必要がある。`cost` を使うとカタカナ寄与が二重計上になる。

→ Phase 1 のタスク 2・3（辞書スキーマ拡張と再生成）を先頭に近い位置へ置く根拠。ただし IPADIC 原典（12.2 MB・`b62f527d…`）はリポジトリ非同梱で、`data/README.md` の手順で都度ダウンロードする必要がある。**ネットワークが使えない環境ではタスク 3 が実行できない**ため、タスク順はタスク 3 に依存しない構成にしてある。

### B. 「16 POS の手書き接続コスト」は正確には「16 POS ラベル上の 13 ペア + 2 個の既定値」

`data/dictionary.tsv` の POS ラベルは実測 16 種で、ここは合っている。一方 `_TRANSITION_COSTS` は **13 ペア**しか持たず、残り（16×16 + literal/unknown を含めた全組合せ）は `0.08`（同POS）か `0.12`（既定）に落ちる。つまり手書きされているのは表ではなく**「表 + 2 つのフォールバック定数」**であり、再現用の特徴量は 13 個ではなく **16 個**（BOS 1 + ペア 13 + same 1 + default 1）必要になる。

### C. Phase 0 baseline の辞書がリポジトリに存在しない

Issue の Phase 0 追記は baseline を `data/dictionary_learned_v2.tsv`（AJIMEE CER 0.274751 / Mozc 92/498 / handmade Top-1 0.857143）と宣言しているが、このファイルは `.gitignore` されており作業ツリーにない。再生成には `data/corpus.txt`（100 MB・これも gitignore）の再取得が要る。実際に同梱辞書で測ると §1.5 の通り **CER 0.290080 / Mozc 88/498 / handmade Top-1 1.000000** で、handmade は 0.857143 ではなく 1.0 になる。

→ Phase 1 の回帰テストは **同梱の `data/dictionary.tsv` を基準**にする（誰でも `git clone` 直後に再現できる）。`dictionary_learned_v2.tsv` は Phase 3 の精度比較で使う baseline であって、Phase 1 の不変条件の基準にはしない。この区別を Phase 1 レポートに明記する。

### D. 「既存の Viterbi 探索をそのまま利用できる」— 概ね正しいが、状態定義が制約になる

`ψ(previous_edge, current_edge)` を一般に許すと、後方 DP の状態は `previous_edge` 全体になり、`h` の計算量が edge 数に比例して増える。現行 `backward_costs_by_pos` は「ラティス全体の POS 集合 × 全位置」を総当たりしており（`:306-313`）、状態を left/right ID（IPADIC は 1,316 種）へ広げるとそのまま破綻する。**`ψ` は「前 edge から取り出した有限の状態キー」にのみ依存する、という制約を設計に明記する**（§3.2）。これは Issue の特徴量候補（前POS / 前right_id / 前活用形 / 単語長）すべてを満たすので実害はない。

### E. 「非負コスト」の記述

§1.3 の通り、実際の `h` は厳密値なので負の重みでも探索順序は保たれる。ただし `Dictionary.from_tsv` の非負バリデーションは残るため、**負になりうるのは「重み × 特徴量」であって辞書 cost 列ではない**、と設計上分離しておく。

---

## 3. Phase 1 の設計

### 3.1 モジュール境界

```
src/beme/
  features.py   FeatureSpec / FeatureExtractor / 経路特徴 Φ(P)
  scorer.py     Scorer プロトコル / LinearScorer / legacy 既定インスタンス
  weights.py    Weights（読み書き・検証・バージョニング）
  lattice.py    探索のみ。コスト式を持たない
  dictionary.py 拡張スキーマ
```

`Scorer` プロトコル（`scorer.py`）:

```python
class Scorer(Protocol):
    spec_id: str
    def state_key(self, edge: LatticeEdge) -> Hashable: ...
    def edge_cost(self, edge: LatticeEdge) -> float: ...
    def transition_cost(self, previous: Hashable | None, edge: LatticeEdge) -> float: ...
```

- `lattice.py` の 4 関数はすべて `scorer: Scorer | None = None` を受け、`None` なら `default_scorer()`（= legacy 固定重み）を使う。既存呼び出し側（`converter.py`・テスト）は無改修で通る。
- `LinearScorer(extractor, weights)` は `(previous_state, edge)` をキーにした遷移コストのメモ化と、`edge` をキーにした edge コストのメモ化を持つ。`LatticeEdge` は frozen slots dataclass なので hashable。

### 3.2 遷移状態の一般化

- `scorer.state_key(edge)` が探索状態を決める。legacy は `edge.token.pos` を返すので**現行と完全に同一**。
- 後方 DP を `backward_costs_by_state(lattice, scorer, *, states=None)` へ一般化する。状態領域は既定で「その位置へ入る edge の `state_key` 集合」（位置 0 は `{None}`、終端は入る edge の集合）に限定する。`k_best_viterbi` は `remaining[edge.end][state_key(edge)]` しか引かないので、限定しても**引かれるキーは必ず存在する**。
- 互換のため `backward_costs_by_pos(lattice)` は残し、`states=` に「全 POS ∪ {None}」を渡す薄いラッパにする（既存の公開 API・`__all__` を壊さない）。
- ドキュメントコメントに「`ψ` は `state_key` にのみ依存してよい。edge 全体に依存する遷移特徴を入れると `h` が厳密でなくなり、K-best の順序保証が壊れる」と明記する。

### 3.3 辞書スキーマ拡張（後方互換）

`data/dictionary.tsv` を壊さないため、**必須 4 列 + 任意列**方式にする。

- 必須: `reading  surface  pos  cost`（先頭 4 列・順序固定。既存ファイルはそのまま読める）
- 任意（5 列目以降・順不同・各 1 回まで・未知列名はエラー）:

| 列 | 型 | 既定 | 用途 |
|---|---|---|---|
| `pos_detail` | str（IPADIC 細分類 1–3 を `/` 連結、`*` は空へ） | `""` | φ |
| `conjugation` | str（`活用型/活用形`） | `""` | φ, ψ |
| `left_id` | int ≥ 0 | `-1`（未知） | ψ |
| `right_id` | int ≥ 0 | `-1`（未知） | ψ |
| `source` | str enum `ipadic` / `overlay` / `learned` / `manual` | `""` | φ |
| `raw_cost` | int（IPADIC 原コスト。overlay は空） | `None` | φ |

- `DictionaryEntry` は上記をデフォルト値付きフィールドとして追加する（`DictionaryEntry(r, s, p, c)` の既存呼び出しを壊さない）。
- 生成物は `data/dictionary_v2.tsv`。**`data/dictionary.tsv` は Phase 1 の間そのまま残し、既定も変えない**（Phase 1 の不変条件の基準だから）。既定切り替えは Phase 3 の判断。
- サイズ / メモリは実測ゲートを置く（§5 タスク3）。既存実測は TSV 14.6 MB・N100 で読み込み 1.851 s・`beme convert` 最大 RSS 159,524 KiB。

### 3.4 特徴量仕様 `legacy_v1`（現行完全再現用）

`FeatureSpec` は id 付きで、**アクティブ特徴が lexical 1 個・transition 1 個だけ**になるよう定義する。これが bit-exact 再現の鍵（§4.2）。

**lexical（`φ`）— 常に 1 個**

| 特徴名 | 値 | 重み |
|---|---|---|
| `lex.base_cost` | `entry.cost`（= Token.cost） | `1.0` |

**transition（`ψ`）— 常にちょうど 1 個。判定順は現行 `pos_transition_cost` と同一**

| 順 | 条件 | 特徴名 | 重み |
|---|---|---|---|
| 1 | `previous is None` | `trans.bos` | `0.0` |
| 2 | `(prev, cur) ∈ _TRANSITION_COSTS` | `trans.pair.<prev>><cur>`（13 個） | 表の値 |
| 3 | `prev == cur` | `trans.same_pos` | `0.08` |
| 4 | それ以外 | `trans.default` | `0.12` |

合計 **16 特徴**。`_TRANSITION_COSTS` の 13 値は `legacy_v1.json` へ移し、`lattice.py` からは辞書リテラルを削除する（Issue の「置き換えるもの」に対応）。ただし**削除ではなく初期重みとして温存**する、という Issue の方針どおり、値そのものは 1 つも変えない。

**`decomposed_v1`（Phase 2/3 の出発点。Phase 1 では既定にしない）**

`lex.base_cost` を次へ分解し、`raw_cost` 列を使う。1 経路あたり複数特徴が立つので bit-exact にはならない（§4.2 で別扱い）。

```
lex.ipadic_cost_mapped   値 = map_ipadic_cost(raw_cost)      重み 1.0
lex.katakana_surface     値 = 1 if カタカナ含む else 0        重み 1.0
lex.source.overlay       値 = 1 if source == overlay          重み 0.0（初期）
lex.unknown              値 = 1 if pos == "unknown"           重み 8.0（UNKNOWN_COST）
lex.literal              値 = 1 if pos == "literal"           重み 0.0
lex.reading_len / lex.surface_len / lex.script.* / lex.proper_noun …  重み 0.0（初期）
trans.right_left_id.<r>><l> / trans.conj.<form>><pos> …             重み 0.0（初期）
```

初期重み 0.0 の特徴は目的関数を変えないので、`decomposed_v1` を legacy と同値に構成すること自体は可能。ただし加算順序が変わるので一致判定は許容誤差付き（§4.2）。

### 3.5 重みファイル形式

`data/weights/legacy_v1.json`:

```json
{
  "format": "beme.weights/1",
  "name": "legacy_v1",
  "feature_spec": "legacy_v1",
  "created": "2026-08-22",
  "dictionary_sha256": "efcdf5a1baeaeb74f8ca6c85552d6fac335d40ad274f1e77181c21c8f37f617b",
  "averaged": false,
  "updates": 0,
  "default_weight": 0.0,
  "weights": {
    "lex.base_cost": 1.0,
    "trans.bos": 0.0,
    "trans.pair.noun>particle": 0.05,
    "trans.same_pos": 0.08,
    "trans.default": 0.12
  }
}
```

規約:

- `json.dump(..., ensure_ascii=False, sort_keys=True, indent=2)` + 末尾改行。float は `repr` 相当（Python の `json` 既定で round-trip する）。**バイト単位で決定的**にし、diff で重みの変化が読めるようにする。
- `feature_spec` が現在の抽出器 id と違えば**読み込み時に即エラー**。Phase 2 で spec を変えた重みを黙って混ぜない。
- `weights` に無い特徴は `default_weight`。`weights` にあるが spec に無い特徴名は既定でエラー、`strict=False` で警告のみ（Phase 2 でアブレーションするときに使う）。
- `dictionary_sha256` は検証用の記録。不一致は警告に留める（辞書差し替え実験を止めない）。
- **サイズ設計**: Phase 1 は 16 特徴なので JSON で十分。Phase 2 で lexical 特徴が語彙規模（数十万）になった時のため、`"weights_path": "legacy_v1.weights.tsv.gz"` という**外部化フィールドを format v1 の時点で予約**しておき、JSON 側は metadata のみになりうる形にする。Phase 1 では実装しないが、フィールド名と読み分岐だけ用意する。

### 3.6 経路特徴ベクトル `Φ(P)` の保存

Phase 2 の更新式 `θ ← θ + η(Φ(P_gold) − Φ(P_pred))` が必要とするのは経路単位の特徴和である。

```python
def path_features(path: LatticePath, extractor: FeatureExtractor) -> dict[str, float]
```

- `LatticePath` は token の連結リストで、`state_key` は Token から取れるので、経路から `Φ` を再構成できる。**ただし現行 `LatticePath` は `Token` しか持たず `LatticeEdge`（start/end）を持たない**。`φ` に位置依存特徴を入れる予定はないので Phase 1 では問題にならないが、Phase 2 のオラクル DP は位置が要る。→ `path_features` は `(lattice, path)` ではなく token 列だけで閉じる設計にし、位置依存特徴を使わない制約を spec 側に書く。
- 保存形式は JSON Lines（1 行 1 経路）:

```json
{"reading":"きしゃ","surface":"記者","cost":0.8,"rank":0,"features":{"lex.base_cost":0.8,"trans.bos":1.0}}
```

- 検算則: **`Σ weights[f] * features[f]` が `path.cost` と一致すること**をテストで常に確認する（legacy は完全一致、他 spec は `abs<1e-9`）。これが Phase 2 の学習が壊れていないことの最小限のチェックになる。

### 3.7 Phase 2 への接続点（Phase 1 で満たしておく条件）

| Phase 2 が要求するもの | Phase 1 側の担保 |
|---|---|
| 制約付きオラクル DP（読み位置 × 正解表記位置 × 直前接続状態） | 「直前接続状態」= `scorer.state_key`。§3.2 で状態を抽象化済み |
| `Φ(P_gold) − Φ(P_pred)` | §3.6 の `path_features` |
| 更新後の重みで即再探索 | `LinearScorer` の重みは差し替え可能。メモ化は重み更新時に破棄する `invalidate()` を持たせる |
| 学習済み重みを別プロセスで再現 | §3.5 の JSON + `--weights` CLI オプション（タスク6） |
| coverage error と model error の分離 | Phase 2。Phase 1 では「gold 表記がラティスに存在するか」を判定できる `Lattice.edges` API が既にあるので追加不要 |
| 負の重み | §1.3 / §2-E。`Dictionary` の非負制約は cost 列のみ。スコア側は非負制約を課さない |

---

## 4. 現行 1-best 再現の回帰テスト設計

受け入れ条件「固定重みで現行 1-best を再現できる」を、**3 層**で検証する。

### 4.1 3 層の構成

| 層 | 何を守るか | 実装 | 実行時間 |
|---|---|---|---|
| L1 単体等価 | `LinearScorer(legacy)` の遷移コストが `pos_transition_cost` と全組合せで一致 | `tests/test_scorer.py`・POS ラベル 18 種（辞書 16 + `unknown` + `literal`）+ `None` の **19 × 18 = 342 組**を parametrize | < 1 s |
| L2 golden K-best | 実辞書での K-best が **経路・コストともに bit-identical** | `tests/test_scorer_golden.py` が `tests/data/kbest_golden_legacy_v1.jsonl` と照合 | 数 s（辞書ロード込み） |
| L3 E2E 指標 | 4 指標が §1.5 と完全一致 | `scripts/evaluate.py --suite all --reranker mock` を手動 or `-m slow` テストで | 6.5 s |

### 4.2 bit-exact を成立させるための条件（設計制約）

浮動小数の加算順序が変われば値が変わる（実測: `きしゃのきしゃがきしゃできしゃした` の 2 位コストは `5.659999999999999`）。したがって次を**設計制約として固定**する。

1. 累積式の順序を変えない: `total = path.cost + scorer.edge_cost(edge) + scorer.transition_cost(prev, edge)`（現行 `lattice.py:357`, `:409` と同順）。
2. `legacy_v1` はアクティブ特徴が lexical 1 個 / transition 1 個。内積は `0.0` から始めて 1 項だけ足すので、
   - `0.0 + 1.0 * cost == cost`
   - `0.0 + w * 1.0 == w`
   がいずれも IEEE754 で厳密に成立する。→ **許容誤差ではなく `==` で assert できる。**
3. `_edge_key` / `_path_key` の tie-break 定義を変えない。
4. 複数特徴が立つ spec（`decomposed_v1`）は 2. が崩れるので、**別テストで `math.isclose(abs_tol=1e-9)` + 「1-best の surface 列が完全一致」** を検証する。値の一致ではなく順位の一致を守る。

### 4.3 golden の作り方（順序が重要）

**golden は必ずリファクタ前に、現行コードで生成する。** リファクタ後に生成すると何も検証していないことになる。

1. `scripts/dump_kbest.py` を追加（現行コードのみに依存）:
   ```
   uv run python scripts/dump_kbest.py \
     --dictionary data/dictionary.tsv --k 10 \
     --cases tests/data/golden_readings.txt \
     --output tests/data/kbest_golden_legacy_v1.jsonl
   ```
   出力 1 行 = 1 入力読み。`{"reading":..., "paths":[{"cost": <repr float>, "surface":..., "tokens":[[reading,surface,pos,cost],…]}, …]}`。**float は `repr()` で round-trip する文字列として書く**（`json` 既定でよい）。
2. `tests/data/golden_readings.txt` のケース選定（決定的・再生成可能）:
   - `data/evaluation_cases.jsonl` の handmade 14 読み（同音異義の sentinel）
   - `data/mozc_oss.tsv` の先頭から、ひらがな読みとして有効な 50 件
   - `data/ajimee_bench_jwtd_v2_v1.json` の先頭 20 件から抽出したひらがな連続スパン（`benchmark.py` の分割規則と同じもの）
   - 合成ストレスケース 6 件: 全ひらがな長文、`unknown` 経由（辞書に無い読み）、literal 混在（数字・記号）、1 文字語連続、`("unknown","unknown")` が発火する読み、同コスト tie が起きる読み
   - 目標 90 前後。K=10 で 1 ファイル 1–2 MB 程度に収める。
3. 生成時の環境を golden の先頭行に metadata として記録: `git rev`, `dictionary sha256`, `k`, `python version`。テストは `dictionary sha256` の不一致を検出したら**明示的に skip ではなく fail** させる（黙って通さない）。
4. L2 テストは以下を assert:
   - 経路数が一致
   - 各順位の `cost` が **文字列表現まで一致**（`repr(float)` 比較で bit-exact を保証）
   - 各順位の `surface` と token 列が一致

### 4.4 「1-best だけ」ではなく K-best 全体を守る理由

Issue の受け入れ条件は 1-best だが、Phase 4 で K-best をリランカへ渡す以上、2 位以下の順序が動けば #2 / #5 の結果も動く。`h` の状態領域を §3.2 で変えるため、**順位保証が壊れる典型的な回帰は 1-best ではなく 2 位以下に出る**。K=10 で守る。

---

## 5. タスク分割

各タスクは worker 1 回の指示で完結する粒度。**タスク 1 は他のすべてに先行**する。タスク 3 のみ外部ネットワーク（IPADIC 原典 12.2 MB）を要する。

共通制約（各指示文にも書く）: git commit / push しない・対象外ファイルを触らない・`uv run pytest -q` が全通することを確認して報告する。

---

### タスク 1: ベースライン凍結（golden K-best ダンプ + 現行指標の記録）

**見積り: S（半日未満）／依存: なし／リファクタ一切なし**

- `scripts/dump_kbest.py` を新規追加（§4.3-1 の I/F）。
- `tests/data/golden_readings.txt` を §4.3-2 の規則で生成するロジックを同スクリプトに `--build-cases` として持たせ、生成物もコミット（再生成可能性 + 決定性の両立）。
- `tests/data/kbest_golden_legacy_v1.jsonl` を**現行コードで**生成。
- `tests/test_scorer_golden.py` を追加し、**現行コードに対して**golden が一致することを確認（この時点では自明に通る＝ golden とテストの整合性の検証）。
- `docs/missions/issue9/baseline.md` に §1.5 の 4 指標・実行コマンド・`git rev 4ba97ad`・辞書 SHA-256・所要時間を記録。Issue の Phase 0 baseline（`dictionary_learned_v2.tsv`）とは**別物である旨を明記**（§2-C）。

**完了条件**
- [ ] golden JSONL が生成され、ケース数 ≥ 80・全ケースで 1 経路以上
- [ ] `tests/test_scorer_golden.py` が現行コードで pass
- [ ] `baseline.md` に 4 指標が記録されている
- [ ] `uv run pytest -q` が全通（既存 71 + 新規）

**検証**
```bash
uv run python scripts/dump_kbest.py --build-cases --output tests/data/golden_readings.txt
uv run python scripts/dump_kbest.py --dictionary data/dictionary.tsv --k 10 \
  --cases tests/data/golden_readings.txt --output tests/data/kbest_golden_legacy_v1.jsonl
uv run pytest -q
uv run python scripts/evaluate.py --suite all --reranker mock
# 期待: CER=0.290080 / exact=0.070000 / Mozc 88/498 / handmade Top-1=1.000000
```

---

### タスク 2: 辞書スキーマの後方互換拡張

**見積り: S／依存: なし（タスク 1 と並行可）**

- `DictionaryEntry` へ §3.3 の 6 フィールドをデフォルト値付きで追加。
- `Dictionary.from_tsv` を「先頭 4 列必須 + 既知の任意列（順不同）+ 未知列はエラー」へ変更。型・範囲バリデーションと既存同様の `path: line N:` 形式のエラーメッセージ。
- `Dictionary.__init__` の並び順キー（`dictionary.py:41`）は**変更しない**。
- ビルダ側（`ipadic.py`）は本タスクでは触らない。

**完了条件**
- [ ] 既存の `data/dictionary.tsv`（4 列）が従来どおり読める
- [ ] 任意列付き TSV が読め、欠損列は既定値になる
- [ ] 未知列名・不正型（`left_id` に負値、`source` に enum 外）が行番号付きでエラー
- [ ] 既存テスト（`tests/test_dictionary_lattice.py` ほか）が無改修で全通
- [ ] golden テスト（タスク1）が引き続き pass

**検証**
```bash
uv run pytest -q tests/test_dictionary_lattice.py tests/test_scorer_golden.py
uv run pytest -q
uv run python scripts/evaluate.py --suite all --reranker mock  # 4 指標が §1.5 と一致
```

---

### タスク 3: IPADIC ビルダ拡張と `dictionary_v2.tsv` 生成

**見積り: M／依存: タスク 2／要ネットワーク（12.2 MB・SHA-256 固定）**

- `ipadic.py` の `_load_imported_entries` で CSV の 左文脈ID(1) / 右文脈ID(2) / 原コスト(3) / 詳細POS(5,6,7) / 活用型(8) / 活用形(9) を保持。
- 重複統合（`(reading, surface, pos)` の最小コスト採用）で**どの行の属性を残したか**が曖昧になるので、「採用した行の属性を採用する」と実装・ドキュメント両方に明記。
- `source` 列: IPADIC 由来 = `ipadic`、overlay 上書き = `overlay`。
- 出力を `--schema v2` で切り替え（既定は現行 4 列のまま。既存の再生成手順を壊さない）。
- `data/dictionary_v2.tsv` を生成してコミット。`data/README.md` に生成手順・列定義・SHA-256・エントリ数・バイト数を追記。
- **実測ゲート**を報告に含める:
  - TSV バイト数（現行 14,583,715）
  - `Dictionary.from_tsv` の読み込み時間（現行比）
  - `beme convert きしゃ` の最大 RSS（`/usr/bin/time -v`。現行 N100 実測 159,524 KiB）
  - ゲート: 読み込み時間 ≤ 2.0×、最大 RSS ≤ 1.6×。超えたら列を int 化 / 分離テーブル化を検討し、判断を報告に書く（この場でリファクタしない）

**完了条件**
- [ ] `--schema v2` で 6 列追加の TSV が生成でき、`Dictionary.from_tsv` で読める
- [ ] 既定（v1）出力が従来と**バイト単位で同一**（`data/dictionary.tsv` を再生成して `sha256sum` が `efcdf5a1…` と一致）
- [ ] `data/dictionary_v2.tsv` のエントリ数が 349,043 と一致
- [ ] 実測ゲート 2 項目の数値が報告されている
- [ ] `uv run pytest -q` 全通

**検証**
```bash
curl -fLO https://deb.debian.org/debian/pool/main/m/mecab-ipadic/mecab-ipadic_2.7.0-20070801+main.orig.tar.gz
sha256sum mecab-ipadic_2.7.0-20070801+main.orig.tar.gz  # b62f527d...
uv run python scripts/build_ipadic_dictionary.py mecab-ipadic_*.tar.gz --output /tmp/dict_v1_check.tsv \
  --expect-sha256 b62f527d881c504576baed9c6ef6561554658b175ce6ae0096a60307e49e3523
sha256sum /tmp/dict_v1_check.tsv data/dictionary.tsv   # 一致すること
uv run python scripts/build_ipadic_dictionary.py mecab-ipadic_*.tar.gz --schema v2 --output data/dictionary_v2.tsv
uv run python -c "from beme.dictionary import Dictionary; d=Dictionary.from_tsv('data/dictionary_v2.tsv'); print(len(d), d.entries[0])"
/usr/bin/time -v uv run beme convert きしゃ 2>&1 | grep -i maximum
```

**ネットワーク不可のときの代替**: 本タスクをスキップし、`tests/data/ipadic_sample.csv`（数百行の手作りサンプル）でビルダのユニットテストのみ実装する。`dictionary_v2.tsv` の生成は後追いにし、タスク 4–8 は 4 列辞書のまま進める（`legacy_v1` は追加列を使わないので影響なし）。

---

### タスク 4: `features.py` / `scorer.py` の実装（探索非依存）

**見積り: M／依存: タスク 2（辞書フィールド）**

- `FeatureSpec` / `FeatureExtractor`（§3.4 の `legacy_v1`）を実装。
- `Weights`（この時点ではインメモリのみ。永続化はタスク 6）と `LinearScorer` を実装。
- `default_scorer()` = `legacy_v1` 抽出器 + `legacy_v1` 重み（この時点ではコード内定数から生成してよい）。
- **`lattice.py` はまだ変更しない。**
- `tests/test_scorer.py`（L1・§4.1）を追加: 19×18 の全組合せで `LinearScorer.transition_cost(prev, edge) == pos_transition_cost(prev, cur)` を `==` で assert。`edge_cost(edge) == edge.token.cost` も同様。
- `_TRANSITION_COSTS` の 13 値が `legacy_v1` の重みと 1 対 1 で一致することを assert（値を写し間違えたら落ちる）。

**完了条件**
- [ ] 342 組合せすべてで `==` 一致（許容誤差なし）
- [ ] 特徴抽出が「lexical ちょうど 1 個 / transition ちょうど 1 個」を返すことを assert
- [ ] `lattice.py` の diff が 0 行
- [ ] `uv run pytest -q` 全通

**検証**
```bash
uv run pytest -q tests/test_scorer.py -v
git diff --stat src/beme/lattice.py   # 空であること
uv run pytest -q
```

---

### タスク 5: `lattice.py` の Scorer 配線と状態一般化（**Phase 1 の山場**）

**見積り: M–L／依存: タスク 1, 4**

- `k_best_viterbi` / `backward_costs_by_pos` / `backward_viterbi` / `_paths_by_node` の 4 箇所すべてを `scorer` 経由へ変更。累積式の順序は §4.2-1 のとおり据え置き。
- `backward_costs_by_state(lattice, scorer, *, states=None)` を実装し、既定で「その位置へ入る edge の状態集合」に限定（§3.2）。`backward_costs_by_pos(lattice)` は互換ラッパとして残す。
- `_TRANSITION_COSTS` と `pos_transition_cost` は**削除せず**、`legacy_v1` の重みから導出される薄い互換関数として残す（`__init__.py` の公開 API・既存テストを壊さない）。実体の重複定義は禁止。値の出所は 1 箇所（`legacy_v1` の重み定義）にする。
- `Converter.__init__` に `scorer: Scorer | None = None` を追加（既定 `None` = legacy）。
- `Lattice.k_best` / `paths_at` / `backward_costs` にも `scorer` を透過。

**完了条件**
- [ ] `tests/test_scorer_golden.py`（タスク1 の golden）が **bit-identical で pass**（cost の `repr` 文字列まで一致）
- [ ] `grep -n "token.cost" src/beme/lattice.py` の結果が tie-break キー（`_edge_key` / `_path_key`）のみ = 探索式に直結の cost 参照が残っていない
- [ ] `scripts/evaluate.py --suite all` の 4 指標が §1.5 と完全一致
- [ ] `pos_transition_cost` の既存テストが無改修で pass
- [ ] `uv run pytest -q` 全通

**検証**
```bash
uv run pytest -q
uv run python scripts/evaluate.py --suite all --reranker mock
# 期待: NDCG@5=0.992972 / MRR=1.000000 / Top-1=1.000000 / CER=0.290080 / exact=0.070000 / Mozc 88/498
grep -n "token\.cost" src/beme/lattice.py
uv run python -c "
from beme.dictionary import Dictionary
from beme.lattice import build_lattice, k_best_viterbi
d=Dictionary.from_tsv('data/dictionary.tsv')
p=k_best_viterbi(build_lattice('きしゃのきしゃがきしゃできしゃした',d),beam_width=3)
print([(repr(x.cost),x.surface) for x in p])"
# 期待: ('5.56','記者の記者が記者で記者した') / ('5.659999999999999', …) の 2 件が先頭
```

---

### タスク 6: 重みの保存・読込と別プロセス再現

**見積り: M／依存: タスク 5**

- `src/beme/weights.py`: §3.5 の JSON 形式の `load` / `dump` / 検証（`format` / `feature_spec` / 未知特徴 / `strict` フラグ / `weights_path` 予約フィールドの読み分岐スタブ）。
- `data/weights/legacy_v1.json` を生成してコミット。`legacy_v1` の重み定義の**唯一の出所**をこのファイルへ移す（コード内定数は読み込み失敗時のフォールバックにせず、ファイル欠損はエラーにする — 黙って別の目的関数で走らせない）。
- `scripts/evaluate.py` と `beme convert` に `--weights PATH`（既定 `data/weights/legacy_v1.json`）を追加。
- ラウンドトリップテスト: dump → load → 同一 dict、JSON バイト列が決定的（2 回 dump して同一）。
- **別プロセス再現テスト**: `subprocess.run([sys.executable, "-m", "beme", "convert", …, "--weights", tmp_path, "--json"])` を 2 通り（既定重み / 一時ファイルへコピーした同内容）で実行し、stdout が完全一致することを assert。受け入れ条件「学習済み重みを別プロセスで読み込み、同じ結果を再現できる」に直接対応。
- 重み差し替えの健全性テスト: `trans.default` を 0.12 → 5.0 にした一時重みで 1-best が**変わる**ことを assert（重みが本当に効いていることの検証。値が効いていなければ Phase 2 は無意味）。

**完了条件**
- [ ] `data/weights/legacy_v1.json` があり、それを使った `evaluate.py --suite all` が §1.5 の 4 指標と一致
- [ ] `feature_spec` 不一致・`format` 不一致・未知特徴名がそれぞれエラー
- [ ] 別プロセス実行の stdout が完全一致
- [ ] 重み改変で 1-best が変わる
- [ ] `uv run pytest -q` 全通

**検証**
```bash
uv run pytest -q tests/test_weights.py -v
uv run python scripts/evaluate.py --suite all --reranker mock --weights data/weights/legacy_v1.json
uv run beme convert きしゃ --json --weights data/weights/legacy_v1.json
python - <<'PY'
import json,pathlib
w=json.loads(pathlib.Path("data/weights/legacy_v1.json").read_text())
w["weights"]["trans.default"]=5.0
pathlib.Path("/tmp/w_broken.json").write_text(json.dumps(w,ensure_ascii=False,sort_keys=True,indent=2))
PY
uv run beme convert きしゃのきしゃ --json --weights /tmp/w_broken.json  # 1-best が変わること
```

---

### タスク 7: 経路特徴ベクトル `Φ(P)` の抽出・保存（Phase 2 接続）

**見積り: S–M／依存: タスク 5, 6**

- `features.path_features(path, extractor)` を実装（§3.6）。
- `scripts/dump_features.py`: 読みリストを受けて K-best 各経路の `Φ` を JSON Lines へ出力。
- **一致検算テスト**: 任意のラティス・任意の K に対し `Σ w[f]·Φ[f] == path.cost`（legacy は `==`、その他 spec は `abs_tol=1e-9`）。
- 差分ユーティリティ `feature_delta(gold, pred) -> dict[str, float]`（Phase 2 の更新式が直接使う）。0 になる項を落とすこと・キー順が決定的であることをテスト。
- JSON Lines のラウンドトリップ（保存 → 読込 → 同一 dict）。

**完了条件**
- [ ] golden の全ケース・全順位で `Σ w·Φ == cost`
- [ ] `feature_delta` が対称性（`delta(a,b) == -delta(b,a)`）と決定的キー順を満たす
- [ ] `dump_features.py` の出力が 2 回実行でバイト単位一致
- [ ] `uv run pytest -q` 全通

**検証**
```bash
uv run pytest -q tests/test_features.py -v
uv run python scripts/dump_features.py --cases tests/data/golden_readings.txt --k 5 --output /tmp/phi1.jsonl
uv run python scripts/dump_features.py --cases tests/data/golden_readings.txt --k 5 --output /tmp/phi2.jsonl
cmp /tmp/phi1.jsonl /tmp/phi2.jsonl
```

---

### タスク 8: `decomposed_v1` spec と Phase 1 レポート

**見積り: M／依存: タスク 3（`dictionary_v2.tsv`）, 7**

- `decomposed_v1`（§3.4）を実装。初期重みは「legacy と同じ目的関数になる値」に設定し、それ以外は 0.0。
- 等価テスト（許容誤差版・§4.2-4）: golden 全ケースで
  - コスト差が `abs_tol=1e-9` 以内
  - **1-best surface が完全一致**、K=10 の surface 列が完全一致
- `dictionary_v2.tsv` を使ったときの `evaluate.py --suite all` の 4 指標が §1.5 と一致することを確認（追加列があっても legacy 経路は変わらない）。
- `docs/missions/issue9/phase1-report.md` を作成:
  - Phase 1 チェックリスト 4 項目・受け入れ条件のうち Phase 1 該当分の充足状況
  - §1.5 と再測定値の対比表
  - 辞書 v2 のサイズ / ロード時間 / RSS 実測
  - `dictionary_learned_v2.tsv` baseline との関係（§2-C）
  - Phase 2 への引き継ぎ事項（`state_key` 設計・`Φ` 形式・`LatticePath` が位置を持たない件）

**完了条件**
- [ ] `decomposed_v1` で 1-best と K=10 の surface 列が legacy と一致
- [ ] 両 spec で `Σ w·Φ == cost` 検算が通る
- [ ] レポートに Phase 1 チェックリスト 4 項目の証跡（コマンドと出力）が載っている
- [ ] `uv run pytest -q` 全通

**検証**
```bash
uv run pytest -q
uv run python scripts/evaluate.py --suite all --reranker mock --weights data/weights/decomposed_v1.json
uv run python scripts/evaluate.py --suite all --reranker mock --dictionary data/dictionary_v2.tsv
```

---

### タスク 9（任意・Phase 3 の素地）: `matrix.def` の抽出と遷移特徴源

**見積り: M／依存: タスク 3／Phase 1 の受け入れ条件には不要**

- `ipadic.py` の `_archive_rows` は `.csv` しか読まないので、`matrix.def`（EUC-JP・先頭行が `<左文脈数> <右文脈数>`）を読む経路を追加。
- `data/ipadic_matrix.npz` 相当（または gzip TSV）へ出力。1,316×1,316 規模のサイズ・ロード時間を実測し、同梱可否を判断して報告。
- `trans.matrix_cost`（値 = `matrix[right_id][left_id]` を正規化したもの、初期重み 0.0）を `decomposed_v1` へ追加できる形にする。**既定では 0.0 のままにし、目的関数を変えない。**

**完了条件**
- [ ] `matrix.def` が読め、次元と数値が原典と一致するサンプル検証がある
- [ ] 追加後も `evaluate.py --suite all` の 4 指標が §1.5 と一致（重み 0.0 なので変わらないこと）
- [ ] サイズ・ロード時間の実測が報告されている

---

## 6. 受け入れ条件との対応

| Issue の受け入れ条件 | Phase 1 での担保 | タスク |
|---|---|---|
| 現行のラティス探索が新しい scorer interface 経由で動く | 4 箇所すべてを `Scorer` 経由に。`grep token.cost` で検証 | 5 |
| 固定重みで現行 1-best を再現できる | L1/L2/L3 の 3 層。L2 は bit-identical | 1, 4, 5 |
| 学習済み重みを別プロセスで読み込み、同じ結果を再現できる | JSON 重み + `--weights` + subprocess テスト | 6 |
| 正解経路と予測経路の特徴量差で学習できる | `path_features` / `feature_delta`（学習自体は Phase 2） | 7 |
| `(reading, target_surface)` から正解経路を取得できる | **Phase 2**（`state_key` 抽象化で前提を用意） | — |
| coverage error と model error を分けて報告できる | **Phase 2** | — |
| held-out test で baseline と比較 | **Phase 3** | — |
| handmade の回帰をケース単位で説明する | **Phase 3**（Phase 1 は回帰ゼロが条件） | — |

Phase 1 チェックリスト 4 項目 → タスク 3 / 4+5 / 1+4+5 / 6+7。

---

## 7. リスク

| リスク | 検出方法 | 対策 |
|---|---|---|
| golden をリファクタ後に生成してしまい何も検証していない | タスク 1 の完了条件に「現行コードで pass」を含める | タスク 1 を必ず単独で先に完了させ、その diff に `lattice.py` を含めない |
| 浮動小数の加算順序変更で bit 一致が崩れる | L2 が `repr` 文字列比較で落ちる | §4.2 の 4 制約を設計として固定。複数特徴 spec は別テストへ隔離 |
| 状態領域の限定で `h` が引けないキーが出る | `KeyError` / `inf` で K-best が空を返す | `remaining[e.end][state_key(e)]` しか引かないことをコードコメントと assert で担保。golden で検出 |
| 辞書 v2 でメモリ / ロード時間が悪化 | タスク 3 の実測ゲート | 超過時は列の int 化・分離テーブル化を検討し判断を報告（勝手に実装しない） |
| IPADIC 原典が取得できずタスク 3 が止まる | ダウンロード失敗 | タスク 3 の代替（サンプル CSV でユニットテストのみ）で他タスクを止めない。`legacy_v1` は追加列不使用 |
| `_TRANSITION_COSTS` 削除で `__init__.py` の公開 API が壊れる | 既存テスト `tests/test_dictionary_lattice.py` | 互換関数として残し、値の出所は重みファイル 1 箇所に統一 |
| コストスケール変更でリランカ / cascade 閾値が黙って変わる | Phase 1 では固定重みなので発火しない | §1.4 をレポートに明記し、Phase 3 の再校正項目として引き継ぐ |
| Issue の Phase 0 baseline（`dictionary_learned_v2.tsv`）と Phase 1 の基準辞書を混同する | 数値が 0.274751 / 0.290080 で食い違う | §2-C の区別をレポート冒頭に固定文言で記載 |

## 8. 非目標（Phase 1 でやらないこと）

- 精度を上げること。Phase 1 で 4 指標が動いたら失敗。
- 実際の学習（perceptron の更新ループ）。Phase 2。
- `matrix.def` を目的関数へ**有効な重みで**投入すること。Phase 3 のアブレーション。
- 既定辞書を `dictionary_v2.tsv` へ切り替えること。Phase 3。
- 頻度特徴（コーパス頻度・読み内表記頻度）の実装。Phase 1 のチェックリスト外。列追加の余地だけ残す。

## 9. 参照

- [Issue #9](https://github.com/sarashinanikki/beme/issues/9) / 設計ノート `docs/beme-learned-path-scorer.html`
- `docs/spec.md` / `data/README.md`（辞書生成規則・SHA-256・N100 実測）
- `docs/deepdive-2026-08-21.md`（正解経路 175/200）
- `docs/split-error-analysis-2026-08-21.md`（分割誤り 100/200 の内訳）
- `docs/cost-learning-reading-field-fix-2026-08-21.md`（`dictionary_learned_v2.tsv` の経緯）
- 過去プラン: `docs/missions/issue4/PLAN.md` / `docs/missions/p2/PLAN.md` / `docs/missions/p5/PLAN-v2.md`
