# Issue #9 Phase 3（評価とアブレーション）実行プラン

作成日: 2026-08-22
対象: `feat/issue9-phase3` @ `9004851`（`origin/main`・Phase 2 マージ済み）
状態: **プランのみ。実装未着手。**
正本: GitHub issue #9 本文の Phase 3 チェックリスト、[Phase 3 着手時の注意点コメント](https://github.com/sarashin65/beme/issues/9#issuecomment-5380697683)、`docs/missions/issue9-phase2/phase2-report.md` §Limitations and Phase 3 Handoff

---

## 0. 最初の worker への指示文の案

> Issue #9 Phase 3 のタスク 1 を実施してください。
>
> リポジトリ: `/home/sarashin/work/beme-9p3`（ブランチ `feat/issue9-phase3`・クリーン）
>
> まず次を順に読んでください。
> 1. `docs/missions/issue9-phase3/PLAN.md` の §1〜§3.1（前提と、アブレーション機構をなぜ feature spec 切替で作るかの設計判断）
> 2. `docs/missions/issue9-phase3/task1-mission.md`（あなたの指示書。ここに目的・対象ファイル・変更内容・完了条件・検証コマンド・制約がすべて書いてあります）
> 3. `src/beme/features.py`・`src/beme/weights.py`・`src/beme/scorer.py`・`scripts/train_scorer.py`・`scripts/evaluate.py`
>
> タスク 1 は「特徴仕様レジストリと特徴フィルタ」の実装です。**アブレーション用の重みゼロ化はしません**。`FeatureSpec` に宣言されていない特徴名は `FeatureExtractor` が出力しない、という一点でアブレーションを表現します。理由は PLAN §3.1 にあります。
>
> 最重要の落とし穴が 2 つあります。指示書に詳細がありますが先に伝えます。
> - `scripts/evaluate.py:263` の `decomposed_v1_extractor() if weights.feature_spec == "decomposed_v1" else legacy_v1_extractor()` は、未知の spec id を**黙って legacy_v1 にフォールバック**します。アブレーション重みを渡すと例外もなく別モデルの数値が出ます。レジストリ経由にして未知 id は例外にしてください。
> - `Weights.load()` は未知 spec id に対し `FeatureSpec(id)`（lexical=`("lex.base_cost",)`・transition=空）を作るため、アブレーション重みファイルが「unknown feature name」で全部弾かれます。ここもレジストリ経由にしてください。
>
> 既存の `legacy_v1` / `decomposed_v1` の数値は 1 bit も変えないこと。`uv run pytest -q` は現在 521 passed です。実装完了後、指示書の検証コマンドを実行し、実際の出力を報告してください。`git commit` / `git push` はしないでください。

---

## 1. 結論

Phase 3 は次の **9 タスク**に分ける。うち 6 タスクは新規コード、2 タスクは実行と測定、1 タスクはレポート。

| # | タスク | 種別 | 依存 |
|---:|---|---|---|
| 1 | 特徴仕様レジストリと特徴フィルタ（アブレーション機構） | 実装 | なし |
| 2 | IPADIC matrix cost 特徴と遷移状態の拡張 | 実装 | 1 |
| 3 | 読み条件付き頻度特徴 | 実装 | 1 |
| 4 | oracle recall@k 測定 CLI | 実装 | なし |
| 5 | 層別評価とケース単位 before/after 診断 CLI | 実装 | なし |
| 6 | 評価データに対するリーク監査 CLI | 実装 | なし |
| 7 | コスト尺度の再較正測定（confidence / cascade / Viterbi ブレンド） | 実装 | なし |
| 8 | アブレーション実行ハーネスと GPU マシンでの全実行 | 実行 | 1–7 |
| 9 | Phase 3 レポート | 文書 | 8 |

チェックリスト 7 項目のうち **5 項目は既存特徴の除去で表現できる**が、**2 項目（matrix cost・読み条件付き頻度）は特徴そのものが未実装**であり、タスク 2・3 が新規実装になる。これがコードを読んで確定した Phase 3 最大の事実である（§2.1）。

アブレーションは**重みゼロ化ではなく feature spec 切替**で実装する。理由は §3.1。

---

## 2. 調査で確定した前提

### 2.1 チェックリスト 7 項目の実装状況（コード実測）

`src/beme/features.py:64-75` の `decomposed_v1` が宣言する特徴は次の 29 個である。

- lexical(13): `lex.ipadic_cost_mapped` / `lex.katakana_surface` / `lex.source.overlay` / `lex.unknown` / `lex.literal` / `lex.base_cost` / `lex.overlay_cost` / `lex.reading_len` / `lex.surface_len` / `lex.script.{kanji,hiragana,katakana,other}` / `lex.proper_noun`
- transition(16): `trans.bos` / `trans.pair.*`(13) / `trans.same_pos` / `trans.default`

| issue のチェックリスト項目 | 現状 | Phase 3 での扱い |
|---|---|---|
| lexical 特徴のみ | 除去で表現可 | タスク 1 |
| transition 特徴のみ | 除去で表現可 | タスク 1 |
| IPADIC word cost あり/なし | `lex.ipadic_cost_mapped` が存在 | タスク 1 |
| **matrix cost あり/なし** | **特徴が存在しない** | **タスク 2（新規実装）** |
| overlay/source 特徴あり/なし | `lex.source.overlay` + `lex.overlay_cost` が存在 | タスク 1 |
| **読み条件付き頻度あり/なし** | **特徴が存在しない** | **タスク 3（新規実装）** |
| 学習後の oracle recall@k を測定 | 測定手段が存在しない | タスク 4（新規実装） |

`docs/missions/issue9/PLAN.md` の「Task 9（任意・`matrix.def` 抽出）は Phase 3 の素地として保留」と整合する。

### 2.2 実測した構造的事実

いずれも本ワークツリーで実際に測って確認した。**推測ではない。**

#### (a) `lex.reading_len` は構造的に学習不能

任意の完全経路について `Σ len(token.reading) == len(reading)` が厳密に成り立つ。gold と pred は同じ読みを覆うので `feature_delta` が常に 0 になり、perceptron の更新が一度も立たない。

実証: Phase 2 の学習済み重み `~/beme-p2/artifacts/issue9-phase2/run1/weights.json` に `lex.reading_len` の項が**存在しない**（初期重みファイルにも無く、一度も更新されなかったため `Weights` に現れない）。

#### (b) `lex.script.*` と `trans.*` は厳密に共線

任意の完全経路について次が厳密に成り立つ。

```text
Σ lex.script.{kanji,hiragana,katakana,other} == |P| == Σ trans.*
```

各トークンが script one-hot をちょうど 1 個、遷移特徴（`trans.bos` を含む）をちょうど 1 個立てるためである。つまり特徴空間に**厳密な線形従属が 1 本**あり、「トークン数ペナルティ」方向の重みは個別には同定できない。

検証: Mozc の 200 読み × K-best 5 = 1,000 経路で `Σscript == |P| == Σtrans` かつ `lex.reading_len == len(reading)` かつ `lex.surface_len == len(surface)` を照合し、**不一致 0**。

この共線性は Phase 2 の学習結果に実際に現れている。初期値 0.0 の `lex.script.*` が 1.02〜1.39 へ、`trans.default` が 0.12 → 1.03、`trans.same_pos` が 0.08 → 1.05 へ同時に増えた。学習が獲得した支配的な効果は「トークン数を減らせ」であり、それが 2 つの特徴族へ恣意的に配分されている。

**帰結**: アブレーションレポートで**生の重み値を特徴の重要度として読んではならない**。§3.8 の判定基準は必ずエラー率差で行い、重みは正規化して（例: `lex.ipadic_cost_mapped` を 1 とする比で）補助的に載せる。

#### (c) 遷移状態を `(pos, right_id)` へ広げたときの実測コスト

matrix cost は「前 edge の `right_id` × 現 edge の `left_id`」に依存するため、遷移状態を `pos` から広げる必要がある。`LinearScorer.state_key` を差し替えて K-best を実測した。

| 読み集合 | 平均長 | `pos` のみ（現行） | `(pos, right_id)` | 倍率 | 遷移キャッシュ |
|---|---:|---:|---:|---:|---|
| Mozc 150 件 | 10.7 字 | 2.17 s | 2.75 s | **×1.27** | 53,093 → 119,523 |
| Mozc 連結 40 件 | 31.0 字 | 1.76 s | 2.48 s | **×1.41** | 51,987 → 121,098 |

`backward_costs_by_state`（`src/beme/lattice.py:341-392`）が各位置に**実際に入ってくる状態だけ**を domain にしているため、状態語彙 801 に対して爆発しない。matrix cost 導入は性能上実行可能である。ゲートは §3.2 で ×2.0 に置く。

#### (d) transition のみのアブレーションは縮退する（想定内）

`lex.*` の重みを全て 0 にして K-best を実行すると、`はしのうえ → 橋之上`（1 トークン・cost 0.000）になる。`trans.bos = 0.0` のため 1 トークン経路のコストが厳密に 0 になり、大量の同点を `_path_key` のタイブレークが決める。

これは**バグではなくアブレーションの正しい下限**である。タスク 8 はこの縮退を予期された結果として扱い、同点数を必ず記録する（§4 の A2 行）。

#### (e) cascade 閾値は現状 1 つの指標にも効いていない

`src/beme/converter.py:174` で `needs_rerank = confidence < threshold` を計算しているが、実際のリランク発火は `:176` の `reranked = not context.is_empty and self.reranker is not None` であり、**`needs_rerank` はどこからも参照されていない**（`src/beme/`・`scripts/` 全体を grep 済み）。`evaluation.py:255` が `rerank_rate` として集計するだけである。

**帰結**: `--cascade-threshold` を再較正しても AJIMEE CER / Mozc / handmade の精度指標は 1 つも動かない。Phase 3 での再較正は「Phase 4 でゲートとして使えるように統計量を測り直す」作業であり、精度改善タスクではない。この事実をレポートに書く（§3.6）。

#### (f) リランカの `viterbi_weight` は mock では使われない

`viterbi_weight * candidate.cost` を使うのは `CrossEncoderReranker`（`reranker.py:293`）と `LLMLogitsReranker`（`reranker.py:433`）だけで、既定の `MockContextReranker` は候補コストを混ぜない。したがって Viterbi ブレンド比の再較正には実モデルが要る。実モデルの所在は §2.4。

### 2.3 d1 データの実測（GPU マシン `beme-gpu:~/beme-d1/repo/data/d1/`）

`manifest.json` と `data_gate.json` を直接読んだ結果。

| 項目 | 値 |
|---|---|
| 件数 | train 10,000 / dev 1,000 / test 2,000 |
| ユニーク記事 | train 696 / dev 65 / test 129 |
| ユニーク読み | train 8,964 / dev 976 / test 1,925 |
| 出所 | `wikimedia/wikipedia` config `20231101.ja` revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` |
| 分割 | `sha256(seed:article_id) % 10000`・seed 20260822・train<7700≤dev<8500≤test |
| 候補生成辞書 | `data/dictionary.tsv` sha256 `efcdf5a1baeaeb74f8ca6c85552d6fac335d40ad274f1e77181c21c8f37f617b`（**v2 ではない**） |
| 生成条件 | `max_reading_characters=16` / `max_span_tokens=3` / `per_reading_cap=8` / `candidate_k=64` / `beam_width=64` |
| ラベル監査 | 200 件抽出・正解 199・`label_accuracy=0.995` |
| 記事 disjoint | `train_dev` / `train_test` / `dev_test` すべて重複 0 |
| 評価データ重複除外 | AJIMEE original/context と handmade context の正規化部分文字列で 238 term・記事 3 本を除外済み。Mozc は「読みと assertion のみで原文を持たないため語彙共有はリークとみなさない」と明記 |
| SHA-256 | train `b9609547…` / dev `8a6aea13…` / test `7b0ad8ff…`（manifest の `artifacts_sha256`） |

**重要**: 候補生成は `dictionary.tsv`（v1）で行われたが、学習と評価は `dictionary_v2.tsv` で行う。d1 の `candidates` 列は Phase 3 では使わない（オラクル DP が v2 ラティス上で gold を引き直す）。Phase 2 test の coverage error 0 はこの条件下での結果である。

**リーク監査（タスク 6）は白紙からやるのではなく、この manifest の除外がリポジトリ内の評価データに対して本当に効いているかの独立検証である。** 特に「Mozc は語彙共有をリークとみなさない」という判断は d1 生成側の主張であり、Phase 3 で追検証する対象。

### 2.4 実行環境の実測

| 資産 | miniPC（`/home/sarashin/work/beme-9p3`） | GPU マシン（`beme-gpu`） |
|---|---|---|
| CPU / RAM | 4 core / 7 GB | 24 core / 31 GB |
| d1 の `*.jsonl` | **無い**（`.gitignore:8` で除外・`manifest.json` 等だけ同梱） | `~/beme-d1/repo/data/d1/`（train 18 MB / dev 1.8 MB / test 3.6 MB） |
| Phase 2 学習済み重み | 無い | `~/beme-p2/artifacts/issue9-phase2/{run1,run2}/` |
| `dictionary_v2.tsv` | あり（33 MB・repo 同梱） | `~/beme-p2/data/dictionary_v2.tsv` |
| `matrix.def`（IPADIC 原文） | `~/.cache/beme-diag/ipadic/matrix.def`（23,008,378 B・1316×1316・1,731,857 行） | **無い** |
| `ipadic` pip パッケージ | dev group | dev group。ただし同梱は `matrix.bin` のみで **`matrix.def` は入っていない** |
| LLM gguf `Qwen2.5-0.5B.Q4_K_M` | `~/.cache/beme/models/` にあり | **無い** |
| cross-encoder（bge-reranker-v2-m3 / japanese-reranker-base-v2） | `~/.cache/beme/cross-encoder/` に両方あり | **無い** |
| `data/corpus.txt` | 無い | 無い |
| HF datasets キャッシュ | — | 空（wikipedia 未取得） |

実測した所要時間:

- `uv run python scripts/evaluate.py --suite all --reranker mock --dictionary data/dictionary_v2.tsv --weights data/weights/decomposed_v1.json` → miniPC で **15.92 秒**。出力は Phase 2 の baseline と完全一致（handmade NDCG@5 0.992972 / Top-1 1.000000、AJIMEE CER 0.290080 / exact 0.070000、Mozc 88/498）。**Phase 2 の baseline は本ワークツリーで再現済み。**
- Phase 2 の学習 1 回（5 epoch・train 10,000）→ GPU マシンの `run1`/`run2` のファイル mtime（21:03→21:05、21:05→21:07）から **約 2 分**。`train_scorer.py` は wall clock を artifact に残さない設計なので mtime が唯一の根拠であり、推定値として扱う。

---

## 3. 設計

### 3.1 アブレーション機構: 重みゼロ化ではなく feature spec 切替

3 案を検討した。

| 案 | 内容 | 判定 |
|---|---|---|
| A: 重みゼロ化 | 初期重みで対象を 0 にする | **却下**。perceptron が 1 更新目で 0 から動かしてしまう。凍結を入れるなら `training.py` の更新式に mask を通す必要があり、Phase 2 が単体テストで固定した符号規約の中核を触ることになる |
| B: feature spec 切替 | `FeatureSpec` に宣言されない特徴を `FeatureExtractor` が出力しない | **採用** |
| C: CLI フラグ | `train_scorer.py` に `--disable-feature` を足す | **却下**。学習時にしか効かず、`evaluate.py` 側で同じ除外を再指定しないと不整合になる。重みファイル自体が条件を自己記述しない |

案 B が優れる理由:

1. **出力されない特徴は更新もされない。** `path_features` は `extractor.lexical_features()` / `transition_features()` の返り値だけを合算するので、除外は `feature_delta` を経由して学習にもスコアリングにも同時に効く。凍結ロジックが不要。
2. **重みファイルが条件を自己記述する。** `Weights.load` は `feature_spec` に宣言されていない特徴名を strict モードで拒否する（`weights.py:112-115`）。条件と成果物の取り違えが型エラーになる。
3. **`train_scorer.py` に CLI 追加が要らない。** extractor は `--initial-weights` の `feature_spec` から決まる（`train_scorer.py:147-148`）。アブレーションは初期重みファイルを差し替えるだけで走る。
4. Phase 2 の学習コアを 1 行も触らない。

必要な変更は 3 点。

- **レジストリ**: `resolve_spec(id) -> FeatureSpec` と `extractor_for(id) -> FeatureExtractor` を `features.py` に置き、`weights.py:106-111`・`train_scorer.py:134-139`・`evaluate.py:263-265` の 3 箇所の分岐をこれに置き換える。
- **フィルタ**: `lexical_features` / `transition_features` の返り値を `spec.lexical_features` / `spec.transition_features` の宣言集合で最後に絞る。`legacy_v1` と `decomposed_v1` は全特徴を宣言しているので出力が 1 bit も変わらない。
- **未知 id は例外**: `evaluate.py` の legacy フォールバック（`else legacy_v1_extractor()`）は**黙って別モデルを走らせる silent-wrong** なので、レジストリの `KeyError` を伝播させる。

アブレーション spec の id 規約: `decomposed_v1` を基底に、除去は `-`、追加は `+` を使う。例 `decomposed_v1-no_ipadic_cost` / `decomposed_v1+matrix_v1`。`Weights.load` の strict 検査に載るので、id と特徴集合の対応は 1 箇所（レジストリ）にしか書けない。

### 3.2 matrix cost 特徴（タスク 2）

**特徴定義**: `trans.matrix_cost`（実数値）。値は IPADIC `matrix.def` の `cost(previous.right_id, current.left_id)` を、既存の IPADIC word cost と同じ方式（`beme.ipadic` の raw→BeME コスト写像に準ずるスケーリング）で写した値。文頭は id 0 を使う。どちらかの id が欠損（overlay 由来など `left_id/right_id` が空の 133 行）なら特徴を立てない。

**遷移状態の拡張**: `FeatureSpec` に `transition_state: tuple[str, ...] = ("pos",)` を足す。`("pos",)` のとき `state_key` は**現行どおり `edge.token.pos` そのもの**を返す（`str` のまま。既存 DP の分割と `LatticePath.state` の型を変えないため）。`("pos", "right_id")` のときだけ `NamedTuple` を返す。

worker が確実に踏む落とし穴を先に潰しておく:

- `transition_features` は `str(previous)` でペア名を作り（`features.py:153`）、`previous == current` で `trans.same_pos` を判定する（`:158`）。状態が `NamedTuple` になったら**両方壊れる**。`previous.pos` を見るように直すこと。
- `features.extractor_state_key`（`features.py:241-249`）は `LinearScorer.state_key` と**同じ値**を返さねばならない。片方だけ直すと `path_features` の遷移特徴が学習中の実際の遷移とずれる。
- `LinearScorer._transition_cache` のキーは `(previous_state, token)`。状態が広がるとキャッシュも広がる（実測 53k → 120k）。位置非依存という前提は matrix cost でも保たれる（id は token の属性であり `edge.start/end` を見ない）ので、Phase 1 コメントの警告には抵触しない。
- `backward_costs_by_pos`（`lattice.py:394`）は既定 scorer（legacy）を使うので `str` キーのまま。触らない。

**データ形式**: `matrix.def` 全体（23 MB テキスト・1,731,857 行）は repo に入れない。`dictionary_v2.tsv` に実在する id は left/right とも **801 個**（最大 1315）で、必要なのは 801×801 = **641,601 ペア**。抽出スクリプトで `int16` 配列 + id 索引のコンパクトなバイナリ（約 1.3 MB）を `data/ipadic_matrix_v1.bin` として生成し、これを repo 同梱する。`matrix.def` は miniPC の `/home/sarashin/.cache/beme-diag/ipadic/matrix.def` にしか無いので、**抽出は miniPC で 1 回だけ行い、生成物を GPU マシンへ配る**。

**性能ゲート**: `--suite all` の実行時間が `decomposed_v1` 比 **×2.0 以内**（実測倍率は ×1.27〜×1.41 なので余裕がある）。超えたら `trans.matrix_cost` を採用しない判断材料としてレポートに残す。

### 3.3 読み条件付き頻度特徴（タスク 3）

`data/corpus.txt` は miniPC にも GPU マシンにも無く、HF の wikipedia キャッシュも空である（§2.4）。100 MB コーパスの再取得は Phase 3 の律速になり、しかも「Wikipedia 頻度が評価データへリークしていないか」という監査対象（注意点 5）を自分で増やす。よって次の二段構えにする。

**F1（本命・コーパス不要）: 辞書内の読み条件付き事前分布**

読み `r` を共有する辞書エントリ集合 `S(r)` 上で、IPADIC raw cost から条件付き分布を作る。

```text
p(s | r) = exp(-raw_cost(s, r) / T) / Σ_{s' ∈ S(r)} exp(-raw_cost(s', r) / T)
lex.reading_cond_logprob    = -log p(s | r)
lex.reading_candidate_count = log(1 + |S(r)|)
```

`T` は辞書構築時の raw cost スケールに合わせた定数（`beme.ipadic` の既存定数から決め、マジックナンバーを新設しない）。`raw_cost` を持たない行（overlay 133 行）は `S(r)` の分母には数えるが特徴を立てない。

これが `lex.ipadic_cost_mapped` と別物である理由は**正規化**にある。`ipadic_cost_mapped` は単語の絶対コスト、`reading_cond_logprob` は「その読みの候補集合の中での相対的な強さ」で、読みの曖昧度が高いほど値が伸びる。設計ノート（`docs/beme-learned-path-scorer.html` §4）の「読み内での表記頻度」に対応するのはこちらである。テーブルは辞書ロード時に構築でき、外部データも新規 repo データも要らない。

**F2（任意・条件付き）: Wikipedia train 記事限定のコーパス頻度**

F1 のアブレーションが決着しない場合にのみ実施する。実施条件は次を**すべて**満たすこと。

1. タスク 6 のリーク監査が完了し、除外すべき記事 ID 集合が確定していること
2. カウント対象を d1 の train 記事 696 本に限定し、dev 65 本 / test 129 本と評価重複で除外済みの 3 本を除くこと
3. `manifest.json` の revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` を固定して取得すること

**Phase 3 の既定スコープは F1 のみ。F2 は明示的に任意扱いとし、やらない場合はレポートに「やらなかった」と理由を書く。**

### 3.4 oracle recall@k（タスク 4）

注意点 3 の「oracle recall@k は条件付きランキング精度と分けて測る」を、**3 つの別々の量**として定義する。混同が起きやすいので定義を固定する。

| 指標 | 定義 | 依存 | 意味 |
|---|---|---|---|
| **lattice coverage** | `constrained_best_path(lattice, gold, scorer) is not None` | k に依存しない | 正解表記を作る経路がラティスに存在するか。Phase 2 の `coverage_error` と同一 |
| **path recall@k** | `k_best_viterbi(beam_width=k)` の経路のいずれかの surface が gold と一致 | k と重みに依存 | 探索が正解を上位 k 経路に入れられるか |
| **surface recall@k** | `Converter.convert(k=k)` の候補 surface に gold が含まれる | k・重み・`beam_width` に依存 | **リランカが実際に受け取る候補集合**での recall |

**`path recall@k` と `surface recall@k` を混同してはならない。** `Converter.convert`（`converter.py:164-170`）は `beam_width` 本の経路を取ってから surface で重複排除し、その後 `[:k]` する。同じ surface を作る経路が複数あるので、`beam_width=64` でもユニーク surface は 64 未満になる。Phase 4 のリランカが見るのは 3 つ目である。したがって出力には **k と `beam_width` を必ず両方**記録する。

測定対象は d1 test 2,000 件（gold が定義できる唯一のデータ）と、gold 表記が取れる範囲で Mozc 498 件・handmade 14 件。AJIMEE は文単位で `convert_sentence` を通す構造なので単一読みの recall と定義が合わない。**AJIMEE は本指標の対象外**とし、理由をレポートに書く。

k は `1, 5, 10, 20, 64` を測り、baseline（`decomposed_v1` 初期重み）と学習済みの両方で出す。学習が「1-best を上げただけ」なのか「上位集合ごと良くしたのか」がこれで分かる。**これが Phase 4 の判断材料そのものである。**

### 3.5 層別評価（タスク 5）

注意点 4 の 4 軸を、d1 test 2,000 件に対して次の層で切る。層はレコードから機械的に決まるものだけを使い、目視分類を入れない。

| 軸 | 層の定義 | 根拠 |
|---|---|---|
| 読み長 | `len(reading)` を 1–3 / 4–6 / 7–10 / 11–16 | manifest の `max_reading_characters=16` |
| 固有名詞 | gold path のトークンに `pos == "proper_noun"` を含むか | 辞書 349,043 行中 138,990 行が proper_noun で最大クラス |
| 数値読み | 読みが数詞（`いち/に/さん/…/じゅう/ひゃく/せん/まん`）を含むか、または gold 表記が数字を含むか | MeCab かな化が誤りを入れやすい |
| MeCab 変換品質 | `span_tokens`（1–3）別、および `contrast_pair_id` を持つ 64 件 | manifest 由来のレコード属性。品質そのものは測れないので**代理指標であると明記する** |

「MeCab 変換品質で層別」は本来ラベル正解率が要るが、監査は 200 件抽出の `label_accuracy=0.995` しかない。**`span_tokens` と contrast pair を代理指標として使い、真の変換品質の層別ではないとレポートに書く**（Phase 2 の反省: 仮説を測定として書かない）。

### 3.6 コスト尺度の再較正（タスク 7）

注意点 1 に対して、§2.2(e)(f) の実測を踏まえ次のように分解する。

**(1) confidence の分布（必須・Phase 3 で完結）**

`confidence = candidates[1].cost - candidates[0].cost`（`converter.py:173`）は生のコスト差なので、学習で重みスケールが変われば分布ごと動く。baseline と学習済みで、handmade 14 / AJIMEE 200 / Mozc 498 / d1 test 2,000 の confidence 分布（分位点 p10/p25/p50/p75/p90）を出す。

再較正の判定基準は **「baseline の `cascade_threshold=2.0` と同じ発火率を与える閾値」**（= baseline の confidence 分布での 2.0 の分位点に対応する、学習済み分布の分位点）とする。精度で選ばない。理由は下記(3)。

**(2) データ収集の穴**

handmade の per-query には `confidence` があるが（`evaluation.py:266`）、AJIMEE の `per_case`（`benchmark.py:215-227`）と Mozc の `per_case`（`:327-339`）には**無い**。タスク 7 は `benchmark.py` を改造せず、独立した測定スクリプトで `Converter.convert` を直接呼んで収集する（既存評価の出力を変えない）。

**(3) `viterbi_weight` は Phase 3 のスコープ外にする**

`viterbi_weight` を使うのは実リランカだけ（§2.2(f)）で、実モデルは miniPC にしか無く、しかも「局所モデルの K-best をリランカへ渡す」こと自体が **Phase 4 のチェックリスト項目**である。Phase 3 は「Phase 4 が較正に使える統計量」を出すところまでとする。具体的には、`viterbi_weight * cost` の項が baseline と学習済みで何倍スケールが変わったかを、**候補間コスト差の分布比**として 1 つの数値で出す。Phase 4 はこれを初期倍率として使える。

**(4) `needs_rerank` が未使用である事実の扱い**

§2.2(e) の通り、cascade フラグは計算されるだけで何もゲートしていない。**Phase 3 ではこれを直さない**（Phase 3 は評価フェーズであり、変換の挙動を変える修正はアブレーション結果の解釈を汚す）。レポートの「Phase 4 引き継ぎ」に、設計上の欠落として明記する。

### 3.7 ケース単位 before/after 診断（タスク 5）

注意点 7 への直接の対応。Phase 2 は handmade の全ケースの順位変動を記録しなかったため、`hashi-no` の回帰説明が仮説どまりになった。Phase 3 では次を**機械生成**する。

1. **全ケースの順位表**: handmade 14 / Mozc 498 の全件について、baseline と学習済みの (a) Top-1 surface、(b) gold の順位、(c) Top-5 の surface とコスト。Top-1 が変わらなかったケースの下位変動も残す。
2. **特徴レベルの差分**: Top-1 が変わったケースについて `path_features(baseline_path) - path_features(learned_path)` と、各特徴の**重み込み寄与** `w[f] * Δf` を降順に出す。「どの特徴が何点分ひっくり返したか」が数値で出るので、`hashi-no` 型の回帰が仮説でなく分解になる。
3. **d1 test の代表誤り**: 学習で新たに誤りになった例と新たに正解になった例を、決定的な順序（レコード id 昇順）で各 20 件。無作為抽出しない。

`path_features` に**必ずスコアリングに使った `Weights` を渡す**こと（Phase 1 コメント項目 2・`features.py:180-194`）。baseline と学習済みで別の重みを使うので、渡し間違えると `ValueError` で落ちる（黙って壊れはしない）。

### 3.8 統計的判定基準（タスク 8・9）

Phase 2 の dev エラーは epoch 1〜5 で 235/234/234/233/235 だった。**epoch 間の差は 1〜2 例で、ノイズと区別できない。** アブレーションの差を同じ精度で読むと誤った結論になる。

したがって:

1. **各アブレーション条件を seed 3 個（9, 17, 33）で学習する。** report する値は 3 seed の中央値と最小–最大幅。
2. **条件間の比較は d1 test 2,000 件の対応ありデータなので McNemar 検定**（正誤の 2×2 分割表）を使う。単純な件数差で優劣を書かない。
3. **判定ラベル**は次の 3 値に固定する。
   - **有意な悪化**: McNemar p < 0.05 かつ中央値が悪化 → その特徴族は必要
   - **有意な改善**: p < 0.05 かつ中央値が改善 → その特徴族は有害または冗長
   - **判別不能**: p ≥ 0.05 → 「差が無い」とは書かず「2,000 件では判別できない」と書く
4. 既存 suite（AJIMEE 200 / Mozc 498 / handmade 14）は**件数が小さく検出力が無い**。回帰検出の補助として値は載せるが、アブレーションの採否判断には使わない。handmade 14 件は特に、1 件の変動が Top-1 を 7.1 pt 動かす。

---

## 4. アブレーション実験マトリクス

基底は `decomposed_v1`（29 特徴）。各条件は「学習 1 回 ×3 seed」+「d1 test 評価」+「既存 suite 評価」+「oracle recall@k」。

`lex.base_cost` / `lex.unknown` / `lex.literal` は**どの条件でも除去しない**。`lex.unknown = 8.0` は未知語経路の暴走を抑える唯一の防波堤であり、`lex.base_cost` は raw cost を持たない overlay/v1 エントリの唯一のコスト源である。これらを抜くのは「アブレーション」ではなくモデルの破壊になる。

| id | spec id | チェックリスト項目 | 基底からの差 | 比較対象 | 判定基準 |
|---|---|---|---|---|---|
| **A0** | `decomposed_v1` | （基準） | なし | Phase 2 run1 | test error 26.9% を ±1 例で再現。**再現しなければ以降を実行しない** |
| **A1** | `decomposed_v1-lex_only` | lexical 特徴のみ | `trans.*` 16 個を除去 | A0 | McNemar |
| **A2** | `decomposed_v1-trans_only` | transition 特徴のみ | `lex.*` 13 個を除去 | A0 | McNemar。**同点数を必ず記録**（§2.2(d) の縮退。1 トークン literal 経路が cost 0 になる） |
| **A3** | `decomposed_v1-no_ipadic_cost` | IPADIC word cost なし | `lex.ipadic_cost_mapped` を除去 | A0 | McNemar |
| **A4** | `decomposed_v1-no_source` | overlay/source 特徴なし | `lex.source.overlay`・`lex.overlay_cost` を除去 | A0 | McNemar。overlay は 349,043 行中 133 行なので効果は小さいはず。**「差が出ない」ことも結論** |
| **A5** | `decomposed_v1-no_katakana` | （注意点 2 の katakana 軸） | `lex.katakana_surface` を除去 | A0 | McNemar |
| **A6** | `decomposed_v1-no_shape` | （注意点 2 の script 軸） | `lex.script.*`4 個・`lex.surface_len`・`lex.proper_noun` を除去 | A0 | McNemar。§2.2(b) の共線性により A6 は「トークン数ペナルティを `trans.*` 側だけに担わせる」条件になる。**この解釈をレポートに書く** |
| **M0** | `decomposed_v1+matrix_v1` | matrix cost あり | `trans.matrix_cost` を追加・状態 `(pos, right_id)` | A0 | McNemar + 実行時間 ×2.0 ゲート |
| **M1** | `decomposed_v1+matrix_v1-pos_pairs` | matrix cost が手書き表を置換できるか | M0 から `trans.pair.*` 13 個を除去 | M0・A0 | McNemar。設計ノートの「matrix.def を使うか捨てるかの二択ではなくなる」の直接検証 |
| **F0** | `decomposed_v1+freq_v1` | 読み条件付き頻度あり | `lex.reading_cond_logprob`・`lex.reading_candidate_count` を追加 | A0 | McNemar |
| **F1** | `decomposed_v1+freq_v1-no_ipadic_cost` | 頻度が word cost を代替できるか | F0 から `lex.ipadic_cost_mapped` を除去 | F0・A3 | McNemar |
| **X0** | `decomposed_v1+matrix_v1+freq_v1` | 上限性能 | matrix + freq の両方 | A0・M0・F0 | McNemar。Phase 4 へ渡す候補モデル |

12 条件 × 3 seed = **36 回の学習**。

各条件について記録するもの（タスク 8 のハーネスが JSON で吐く）:

- `train_scorer.py` の `report.json` 全体（baseline dev/test・全 epoch・選択 epoch・test）
- `weights.json`（`lex.ipadic_cost_mapped` を 1 とした正規化重みも併記。§2.2(b) の理由で生値だけを載せない）
- `--suite all --reranker mock --dictionary data/dictionary_v2.tsv` の JSON
- oracle recall@k（k = 1, 5, 10, 20, 64 × baseline/学習済み）
- 層別 error rate（§3.5 の 4 軸）
- confidence 分位点（§3.6）
- A0 との McNemar 分割表と p 値

---

## 5. タスク分割

各タスクは worker 1 回の指示で完結するサイズにしてある。指示書は `docs/missions/issue9-phase3/taskN-mission.md`。

| 依存順 | Task | 主対象 | 難易度 | 見積り | 依存 |
|---:|---|---|---|---:|---|
| 1 | 1 | 特徴仕様レジストリと特徴フィルタ: `features.py` / `weights.py` / `scorer.py` / `scripts/*.py` | 中 | 4–6h | なし |
| 1 | 4 | oracle recall@k CLI: `scripts/oracle_recall.py` | 中 | 3–5h | なし |
| 1 | 5 | 層別評価とケース診断 CLI: `scripts/diagnose_cases.py` | 中 | 4–6h | なし |
| 1 | 6 | リーク監査 CLI: `scripts/audit_leakage.py` | 中 | 3–4h | なし |
| 1 | 7 | コスト尺度測定 CLI: `scripts/measure_cost_scale.py` | 低 | 2–3h | なし |
| 2 | 2 | matrix cost 特徴と状態拡張: `features.py` / `scorer.py` / `scripts/build_matrix_table.py` | **高** | 6–8h | 1 |
| 2 | 3 | 読み条件付き頻度特徴: `features.py` / `dictionary.py` | 中 | 4–5h | 1 |
| 3 | 8 | アブレーション実行ハーネスと GPU 実行: `scripts/run_ablations.py` | 中 | 3–4h + 実行 2h | 1–7 |
| 4 | 9 | Phase 3 レポート: `docs/missions/issue9-phase3/phase3-report.md` | 中 | 3–5h | 8 |

タスク 1・4・5・6・7 は互いに独立で並行可能。タスク 2 が唯一の高難度（探索の遷移状態に触る）で、Phase 2 の PLAN §2.5 が「位置依存特徴を追加するなら Phase 3 以降に `LatticePath` と scorer cache key を同時再設計」と留保していた箇所に相当する。ただし matrix cost は**位置非依存**なので、`LatticePath` の再設計は不要（§3.2）。

---

## 6. 学習実行場所の判断

**結論: 学習と d1 評価は GPU マシン（`beme-gpu`）で行う。miniPC は実装・テスト・`matrix.def` 抽出・（任意の）実リランカ測定に使う。**

| 検討点 | 判断 |
|---|---|
| d1 の `*.jsonl` の所在 | GPU マシンのみ。`.gitignore:8` で repo 外。**これが決定要因** |
| miniPC へ d1 を持ってくる案 | 却下。23 MB のコピー自体は容易だが、gitignore 対象の学習データの複製を増やすと出所の追跡が壊れる。Phase 2 の manifest SHA-256 は GPU マシン上の実体を指している |
| 計算資源 | GPU マシン 24 core / 31 GB、miniPC 4 core / 7 GB。学習は単一プロセスだが 36 回を回すので並列度が効く |
| GPU の要否 | **不要**。averaged structured perceptron は純 CPU。「GPU マシン」は実質「速い CPU 機」として使う |
| 学習 1 回の所要 | 約 2 分（§2.4・run1/run2 の mtime 実測）。36 回で逐次 72 分、6 並列なら約 15 分 |
| 既存 suite 評価 | 16 秒（miniPC 実測）。どちらでも可。**GPU マシンで学習と同時に回して条件を揃える** |
| `matrix.def` | miniPC にしか無い（GPU マシンの `ipadic` パッケージは `matrix.bin` のみ）。**抽出は miniPC で 1 回。生成物 `data/ipadic_matrix_v1.bin`（約 1.3 MB）を repo に入れて GPU マシンへ配る** |
| 実リランカ | LLM gguf と cross-encoder は miniPC にのみキャッシュ済み。§3.6(3) の通り Phase 3 スコープ外なので、GPU マシンにモデルを落とす必要はない |
| GPU マシンの repo 状態 | `~/beme-p2` は git 管理下にない素の展開物（`git rev-parse` が失敗する）。Phase 3 では**新しい作業ディレクトリ `~/beme-p3` に本ブランチをクローンし、`data/d1` へ `~/beme-d1/repo/data/d1/*.jsonl` を symlink かコピーで配置する**。`train_scorer.py` の `git_commit` が `None` になるのを避けるため、必ず git クローンにする（Phase 2 の run1 report は `git_commit: null` だった） |

並列実行時の注意: 36 回を並列で回すとき、`train_scorer.py` は `--output-weights` / `--output-report` の親ディレクトリに `.beme-tmp-*` を作って atomic replace する（`train_scorer.py:110-127`）。**条件ごとに別ディレクトリを与えれば衝突しない。** 同一ディレクトリへ複数プロセスを向けないこと。

---

## 7. Phase 3 全体の見積もり

| 項目 | 数量 | 根拠 |
|---|---:|---|
| タスク数 | **9** | §5 |
| 新規実装タスク | 6 | タスク 1–7 のうち 8 を除く |
| 新規スクリプト | 5 | `build_matrix_table.py` / `oracle_recall.py` / `diagnose_cases.py` / `audit_leakage.py` / `measure_cost_scale.py` / `run_ablations.py`（6 本。うち 1 本は実行ハーネス） |
| アブレーション条件 | **12** | §4 |
| **学習実行回数** | **36**（12 条件 × 3 seed） | §3.8 の統計要件 |
| 学習の総 wall time | 逐次 約 72 分 / 6 並列 約 15 分 | 1 回 約 2 分（§2.4 実測） |
| 既存 suite 評価回数 | 12（条件ごと 1 回・代表 seed） | 1 回 16 秒 → 約 3 分 |
| oracle recall@k 測定 | 12 条件 × 5 k × 2（baseline/学習済み） | d1 test 2,000 件の K-best を k=64 まで取るので**ここが最も重い**。miniPC 実測から 2,000 件 × 64-best で 1 条件あたり数分と見込む |
| 実装工数 | 約 30–40 時間 | §5 の見積り合計 |
| 実行・レポート | 約 8–12 時間 | タスク 8・9 |
| **Phase 3 合計** | **約 40–55 時間 + 実行 2–3 時間** | |

リスクは実装側に偏っており、計算資源は律速でない。最大の不確実性はタスク 2（matrix cost・遷移状態拡張）で、ここだけ見積りが外れる可能性が高い。

---

## 8. レポート構成（タスク 9・`phase3-report.md`）

Phase 2 レポートの反省（集計値だけで回帰を説明した）を構造で防ぐため、次の順で書く。

1. **Scope and Reproducibility** — 実行機・commit・全コマンドの逐語・入力 SHA-256
2. **Ablation Matrix Results** — §4 の 12 条件 × (d1 test error / 既存 suite / McNemar p / 判定ラベル)。**判定不能は判定不能と書く**
3. **Learned Balance Table** — 設計ノート §8 が予測した表（IPADIC 生起コスト弱め / matrix 0.2 倍程度 / カタカナ少し不利 / 読み条件付き頻度強く / overlay は出所特徴）に対する**実測値の答え合わせ**。正規化重みで示し、§2.2(b) の共線性の注記を必ず添える
4. **Oracle Recall@k** — 3 定義（lattice coverage / path recall@k / surface recall@k）を分けた表。k と `beam_width` を明記。AJIMEE を対象外とした理由
5. **Stratified Results** — §3.5 の 4 軸。MeCab 品質軸は代理指標である旨を明記
6. **Case-Level Before/After** — handmade 全 14 件と Mozc 全 498 件の順位表、Top-1 変動ケースの特徴寄与分解、d1 test の代表誤り各 20 件。**`hashi-no` は特徴寄与分解で決着させる**
7. **Cost-Scale Recalibration** — confidence 分位点表、再較正した `cascade_threshold`、`needs_rerank` が未使用である事実、`viterbi_weight` の初期倍率
8. **Leakage Audit** — d1 manifest の除外の独立検証結果。Mozc の語彙共有をリークとみなさない判断の追検証
9. **Negative and Null Results** — 効果が出なかった条件、判別不能だった条件、実施しなかったこと（F2 コーパス頻度など）とその理由
10. **Limitations and Phase 4 Handoff** — §11

---

## 9. スコープ判断（明示）

| 項目 | 判断 | 理由 |
|---|---|---|
| **重みの配布形態（`pyproject.toml` の package-data）** | **Phase 3 のスコープ外。判断だけ下してレポートに書く** | Phase 1・2 コメントから 2 フェーズ持ち越されている。Phase 3 は評価フェーズであり、パッケージング変更は評価対象の挙動と無関係。**推奨する結論は「重みは外部パス必須の設計と明示する」**。Phase 3 で 12 条件 × 3 seed の重みが増えるので、成果物を wheel に同梱する設計はスケールしない。`data/weights/legacy_v1.json` と `decomposed_v1.json` の 2 つだけを package-data に入れて `default_scorer()` を動かし、学習済み重みは常に外部パスで渡す、という分離を Phase 4 の作業として起票する |
| `needs_rerank` が未使用（§2.2(e)） | **Phase 3 では直さない。レポートに記載** | 変換挙動を変えるとアブレーション結果の解釈が汚れる |
| `viterbi_weight` の再較正 | **Phase 4 へ。統計量だけ出す** | 実リランカが要り、それ自体が Phase 4 のチェックリスト項目 |
| F2（Wikipedia コーパス頻度） | **任意。既定では実施しない** | §3.3 |
| `lex.reading_len` の削除 | **Phase 3 では削除しない。事実だけ記録** | 学習不能（§2.2(a)）だが、削除は spec 変更であり `decomposed_v1` の bit 互換を壊す。Phase 4 の feature spec v2 で外す |
| AJIMEE の oracle recall@k | **対象外。理由を記載** | 文単位変換で単一読みの recall と定義が合わない（§3.4） |

---

## 10. リスクと緩和

| リスク | 影響 | 緩和 |
|---|---|---|
| A0 が Phase 2 の 26.9% を再現しない | 以降の全条件が比較不能 | **A0 を gate にする。再現しなければ止めて原因を調べる**（`~/beme-p2` は git 管理外なので commit 差が疑わしい第一候補） |
| `evaluate.py` の legacy フォールバックを直し忘れる | アブレーション評価が黙って legacy_v1 の数値を出す | タスク 1 の必須テストに「未知 spec id で例外」を入れる（§3.1） |
| `extractor_state_key` と `state_key` の乖離 | 学習中の勾配が実際の遷移とずれる。**例外は出ない** | タスク 2 の必須テストで両者の一致を全 edge について照合 |
| matrix cost で K-best が遅くなりすぎる | 実用性を損なう | ×2.0 ゲート（実測 ×1.27〜1.41 で余裕あり）。超過時は不採用の判断材料として記録 |
| 共線性を無視して重みを重要度と読む | 誤った結論 | §2.2(b) を判定基準に組み込み、判定はエラー率差でのみ行う |
| 1〜2 例差をノイズと区別できない | 誤った採否 | 3 seed + McNemar（§3.8）。「判別不能」を正式なラベルにする |
| handmade 14 件で回帰を判断する | 1 件で 7.1 pt 動く | アブレーション採否に使わない。回帰検出の補助のみ（§3.8-4） |
| `matrix.def` が GPU マシンに無い | タスク 2 の実行が GPU 側で止まる | miniPC で抽出 → 約 1.3 MB のバイナリを repo 同梱（§3.2・§6） |
| 36 並列実行で temp ファイルが衝突 | artifact が壊れる | 条件ごとに別出力ディレクトリ（§6） |
| Phase 2 の run1 重みを再利用 | metadata の `updates`/`averaging_steps` が末尾 epoch の値のまま（Phase 2 コメントの引き継ぎ注意） | **再利用しない。A0 として学習し直す** |
| リーク監査が d1 manifest の主張を追認するだけになる | 監査の意味が無い | 独立実装で照合する。特に「Mozc の語彙共有はリークでない」を追検証（§2.3） |

---

## 11. Phase 4（文脈モデルへの接続）への引き継ぎ事項

Phase 4 のチェックリストは「局所モデルの K-best を #2 / #5 のリランカへ渡す」「oracle recall と Top-1 改善を分けて評価」「局所モデル単独・リランカ単独・併用を比較」である。Phase 3 から次を渡す。

1. **リランカへ渡す局所モデルの確定** — X0（matrix + freq）を第一候補、A0 を対照とする。Phase 3 の判定ラベルが「判別不能」なら**単純な A0 を選ぶ**（複雑な方を選ぶ理由がない）。
2. **surface recall@k の実測曲線** — Phase 4 の `rerank_k` はこの曲線から決める。`beam_width=64` でユニーク surface が何本になるか、gold がどの k で入るかが Phase 3 の成果物にある（§3.4）。**recall@k が頭打ちなら、リランカを強くしても Top-1 は上がらない。** Phase 4 の最初の判断はこれ。
3. **`viterbi_weight` の初期倍率** — 学習済みコストの候補間差分スケールが baseline の何倍かを 1 数値で渡す（§3.6(3)）。`viterbi_weight = 1.0 / 倍率` が探索の出発点。
4. **`cascade_threshold` の再較正値と、それが今は何もゲートしていない事実** — `converter.py:174` の `needs_rerank` を実際のゲートとして配線するかどうかが Phase 4 の設計判断（§3.6(4)）。カスケードで実リランカの呼び出しを間引くなら必須。
5. **層別の弱点** — どの層（読み長・固有名詞・数値読み）が局所モデルで残るか。リランカが担うべき領域がここで決まる（§3.5）。
6. **リーク監査の結果** — Phase 4 は AJIMEE/Mozc での改善を主張することになるので、d1 で学習した重みが評価データへリークしていない根拠が要る（§3.5・タスク 6）。
7. **重みの配布形態の結論** — 「学習済み重みは外部パス必須、`legacy_v1`/`decomposed_v1` のみ package-data」（§9）。Phase 4 でリランカ設定と合わせて起票する。
8. **feature spec v2 の素地** — `lex.reading_len` の削除（学習不能）、`lex.script.*` と `trans.*` の共線性の解消（トークン数特徴を 1 本に明示する）、語彙特徴（reading / surface / POS 詳細 / left-right id / 活用形）の高 cardinality 化と `weights_path` による疎重みの外部化（`weights.py:99-100` で予約済み・未実装）。
9. **位置依存特徴** — matrix cost は位置非依存なので Phase 3 では `LatticePath` を再設計しない。文頭・文末距離のような位置依存特徴を Phase 4 で入れるなら、`LinearScorer._edge_cache` が `edge.token` キーである前提（`scorer.py:40-52`）が壊れる。**キャッシュキーと `LatticePath` の位置保持を同時に再設計すること。** 例外も警告も出ない。

---

## 12. Worker 指示文

- `docs/missions/issue9-phase3/task1-mission.md` — 特徴仕様レジストリと特徴フィルタ
- `docs/missions/issue9-phase3/task2-mission.md` — IPADIC matrix cost 特徴と遷移状態の拡張
- `docs/missions/issue9-phase3/task3-mission.md` — 読み条件付き頻度特徴
- `docs/missions/issue9-phase3/task4-mission.md` — oracle recall@k 測定 CLI
- `docs/missions/issue9-phase3/task5-mission.md` — 層別評価とケース単位 before/after 診断 CLI
- `docs/missions/issue9-phase3/task6-mission.md` — 評価データに対するリーク監査 CLI
- `docs/missions/issue9-phase3/task7-mission.md` — コスト尺度の再較正測定 CLI
- `docs/missions/issue9-phase3/task8-mission.md` — アブレーション実行ハーネスと GPU マシンでの全実行
- `docs/missions/issue9-phase3/task9-mission.md` — Phase 3 レポート
