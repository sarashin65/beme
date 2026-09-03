# Issue #19（局所モデルの過少適合 → S 実測 + F2 語彙化）実行プラン

作成日: 2026-08-27
対象: worktree `/home/sarashin/work/beme-19`・branch `feat/issue19-f2`（`origin/main` @ `a4b690b` 起点・クリーン）
状態: **プランのみ。実装未着手。**
正本: `gh issue view 19`（本文 + 2026-08-26 コメント「検証結果 + Opus アドバイザリー」の着手順序表）

---

## 0. 最初の worker への指示文の案（タスク 1・S 実測）

> Issue #19 のタスク 1（S 実測）を実施してください。
>
> リポジトリ: `/home/sarashin/work/beme-19`（ブランチ `feat/issue19-f2`・クリーン）。**この worktree の中だけで作業してください**（`cd` で他のチェックアウトに移動しない）。
>
> まず次を順に読んでください。
> 1. `gh issue view 19`（本文 + コメント）— 正本。特に「情報天井」の節と着手順序表
> 2. `docs/missions/issue19/PLAN.md` の §1〜§3（前提と、なぜ「同点率」を 4 粒度 × 2 候補集合で測るのか）
> 3. `docs/missions/issue19/task1-mission.md`（あなたの指示書。目的・出力仕様・完了条件・検証コマンド・制約が全部あります）
> 4. `src/beme/features.py`・`src/beme/scorer.py`・`src/beme/oracle.py`・`scripts/diagnose_cases.py`
>
> タスク 1 は **測定スクリプトを 1 本追加するだけ**です。`src/beme/` は 1 行も変更しません。特徴も重みも追加しません。目的は F2（コーパス頻度特徴）の設計判断に必要な 2 つの数字を、d1 の実データ上で確定させることです。
>
> - **S1**: d1 の候補集合上で「lexical 特徴ベクトルが完全一致する候補対」がどれだけあるか。Opus は**辞書レベル**で 24.9% と算出しましたが、それは d1 の実際の候補集合上の値ではありません。あなたが出すのは後者です
> - **S2**: d1 train/test の読み被覆率・(読み,表記) 被覆率・gold 到達不能率。F2 の外部頻度表が「そもそも何件に触れるか」の上限を与えます
>
> 落とし穴を 3 つ先に伝えます。指示書に詳細があります。
> - d1 の候補列（JSONL の `candidates`）は **`data/dictionary.tsv`（v1・sha `efcdf5a1…`）**で生成されています（`data/d1/manifest.json` の `candidate_generation.dictionary`）。一方 学習/評価は **`data/dictionary_v2.tsv`** です。**主測定は v2 の lattice k-best（実際に局所モデルが順位付けする候補集合）**、JSONL の候補列は副測定です。両方出してください
> - 「同点」は粒度で答えが変わります。lexical だけ一致していても transition が違えば順位付けは可能です。指示書の 4 粒度（a〜d）を全部出してください。**意思決定に効く唯一の指標は「gold と非 gold が同点」の率**です
> - 現行 base は A0（`decomposed_v1`）ではなく **A8（`decomposed_v1-no_script-surface_len`）**です（issue #17 / commit `a4b690b` で採用済み）。A8 と A0 の両方で測ってください
>
> `uv run pytest -q` のこの worktree でのベースラインは **638 passed / 2 failed** です。失敗 2 件は `tests/test_audit_leakage.py` で、gitignore された `data/d1/*.jsonl` がこの worktree に置いてあるための環境固有の失敗です。**直さないでください**（合否判定から除外）。
>
> 完了したら、指示書の検証コマンドを実行し、**実際の出力の数値**を報告してください。`git commit` / `git push` はしないでください。

---

## 1. 結論

Issue #19 の S 項目・A 項目を **7 タスク**に分ける。うち 3 タスクは新規コード、3 タスクは実行と測定、1 タスクはレポート。

| # | タスク | 種別 | 実行機 | 依存 |
|---:|---|---|---|---|
| 1 | **S 実測**: d1 候補集合上の同一特徴ベクトル率 + 読み/(読み,表記) 被覆率 | 実装+測定 | ローカル | なし |
| 2 | **S 実測**: 最適化要因の排除（lr / epoch スイープで「28% は容量由来」を確定） | 実行 | GPU (`ws`) | なし（1 と並行可） |
| 3 | **F2-a**: コーパス頻度表ビルダー（train 域記事のみ・リーク対策込み） | 実装 | ローカル | 1（設計反映） |
| 4 | **F2-b**: 3 特徴の実装と feature spec 登録 | 実装 | ローカル | 3（表形式の確定） |
| 5 | **F2-c**: 学習実行とアブレーション判定（C0/C1 × 3 seed・McNemar） | 実行 | GPU (`ws`) | 4 |
| 6 | **F2-d**: Phase 4 contrast セット再測（overall / 非 contrast gain で） | 実行 | GPU (`ws`) | 5 |
| 7 | レポート | 文書 | ローカル | 5, 6（2 があれば含める） |

依存グラフ:

```
task1 ──┐
        ├─→ task3 ─→ task4 ─→ task5 ─→ task6 ─→ task7
task2 ──┴────────────────────────────────────────┘
```

タスク 1 と 2 は独立に走らせてよい。タスク 2 は `ws` の復旧待ちなので、復旧しない場合は task7 のレポートに「未実施」と明記して先に進む（**task2 は task3-6 をブロックしない**）。

---

## 2. 調査で確定した前提（この worktree でコードを読んで実測した事実）

### 2.1 split 判定は再現できる（実測済み）

`scripts/build_d1_data.py:56-62` の `article_split(article_id, seed)` を **13,000 行すべてに適用して、記録済みの `split` フィールドと 0 件不一致**であることを確認した。

```
checked 13000 mismatch 0
```

つまり F2 のコーパス集計で `article_split(article_id, 20260822) == "train"` を条件にすれば、dev/test 記事は構造的に除外できる。**ハッシュ式を写経してはいけない**（写経が唯一のリーク経路になる）。`build_d1_data.py` から `import` すること（`scripts/evaluate_d1.py:31` が `from train_d1_qlora import prompt_text` としている先例がある）。

### 2.2 d1 の候補列と学習時の候補集合は別物

`data/d1/manifest.json`:

- `candidate_generation.dictionary`: `data/dictionary.tsv`（v1・sha `efcdf5a1…`）・`beam_width` 64・`candidate_k` 64
- 一方 `scripts/train_scorer.py:33` の既定辞書は `data/dictionary_v2.tsv`

学習は JSONL の `candidates` を**使わない**。`src/beme/training.py:178-183` は読みから lattice を張り直し、`_top1_path` と `oracle.constrained_best_path(lattice, target_surface, scorer)` を比べている。したがって**局所モデルが実際に順位付けしている候補集合は v2 lattice の k-best** であり、S1 の主測定はそちらで取る。

### 2.3 d1 の規模（実測）

| split | 件数 | 記事数 | 平均候補数 | span_tokens 1/2/3 |
|---|---:|---:|---:|---|
| train | 10,000 | 696 | 49.4 | 8,156 / 980 / 864 |
| dev | 1,000 | 65 | 48.6 | 877 / 74 / 49 |
| test | 2,000 | 129 | 49.2 | 1,730 / 143 / 127 |

候補パスは合計約 64 万本。ローカル 1 パスで測定できる規模。

### 2.4 F2 は `_edge_cache` も k-best も壊さない（コードで確認）

- `src/beme/scorer.py:59` の `self._edge_cache: dict[Token, float]` は **`edge.token` をキー**にしている。コメント（scorer.py:44-58）が明記するとおり lexical 特徴は位置非依存。F2 の 3 特徴は `token.reading` と `token.surface` のみの関数なので、この不変条件を保つ。**キャッシュ変更不要**
- `FeatureSpec.transition_state` は A8/A0 とも `("pos",)` のまま。`state_key_function`（features.py:389-400）も k-best（`lattice.k_best_viterbi`）も無変更
- `path_features`（features.py:326）は lexical をトークンごとに加算するだけなので、多トークンパスでも 3 特徴は自然に和になる
- 唯一必要な I/O 側の変更は `Weights.load` が通す `resolve_spec`（weights.py:104）に新 spec id を登録すること

→ **`src/beme/lattice.py` / `src/beme/converter.py` は 1 行も触らない。**

### 2.5 現行 base は A8 であって A0 ではない

`scripts/run_ablations.py:37-38`:

```
"A7": "decomposed_v1-no_script",
"A8": "decomposed_v1-no_script-surface_len",
```

A8 = decomposed_v1 から `lex.script.*`（4個）と `lex.surface_len`（1個）を除いたもの = lexical 9 + transition 16 = **25 パラメータ**。issue #19 コメントの「既存 30 特徴に追加」は A8 採用前の記述である。**F2 は A8 の上に積むのを本命（C0）とし、issue 記載どおりの A0 + 3 も並行条件（C1）として測る**（§3.3）。

### 2.6 特徴表の遅延ロードには先例がある

`trans.matrix_cost` は呼び出し側に何も渡させず、`features.py:222-226` の `_matrix_table()`（`lru_cache(maxsize=1)`）が `matrix_cost.DEFAULT_MATRIX_PATH` から自力で読む。

一方 `reading_conditional` は `extractor_for(spec_id, dictionary.reading_conditional)` として**呼び出し 7 箇所すべてに引数を配っている**（train_scorer / oracle_recall / measure_cost_scale / diagnose_cases / evaluate / rerank_eval / tests）。F2 で同じことをすると 7 箇所の改修になる。

→ **F2 は matrix 方式を採る**（§3.2）。呼び出し側は無変更。

### 2.7 テストのベースライン

`uv run pytest -q` = **638 passed, 2 failed**（86 秒）。失敗するのは

- `tests/test_audit_leakage.py::test_repo_partial_audit_marks_d1_jsonl_unverifiable`
- `tests/test_audit_leakage.py::test_unverifiable_checks_do_not_fail_the_audit`

どちらも「`data/d1/*.jsonl` がリポジトリに存在しない」ことを前提に書かれており、この worktree には d1 を置いてあるために落ちる。**環境固有であり、issue #19 の変更の合否とは無関係。直さない。**

---

## 3. 設計判断

### 3.1 S1 は 4 粒度 × 2 候補集合で測る

「同一特徴ベクトル」は粒度を決めないと数字が意味を持たない。lexical だけ一致していても transition が違えば線形モデルは順位付けできる。逆に、最終的に効くのは**学習後のコスト**が同点かどうかである。よって次の 4 つを全部出す。

| 粒度 | 定義 | なぜ要るか |
|---|---|---|
| (a) | `decomposed_v1` の lexical 14 次元のみ完全一致 | Opus の辞書レベル 24.9% と直接比較できる唯一の粒度 |
| (b) | A8 の lexical 9 次元のみ完全一致 | 現行 base の実際の情報量 |
| (c) | A8 の全 25 次元（lexical + transition）完全一致 | **どんな重みでも順位付け不能**な真の同点 |
| (d) | A8 初期重み（`data/weights/ablations/decomposed_v1-no_script-surface_len.json`）でのパスコスト同点 | 実運用でタイブレークに落ちる率 |

各粒度について 3 つの率を出す:

1. 候補集合中に同点対を 1 組でも含むケースの割合
2. 最安 2 候補が同点であるケースの割合
3. **gold と非 gold が同点であるケースの割合** ← 意思決定に効く唯一の指標

候補集合は主測定 = **v2 lattice の k-best（k=64・beam 64）**、副測定 = **JSONL の `candidates` 列**（v1 辞書由来）。副測定を残すのは、Opus の見積りと d1 生成時の設定に橋を架けるためである。

### 3.2 コーパス頻度表は遅延シングルトンで読む（matrix 方式）

新規 `src/beme/corpus_freq.py` に `CorpusFrequencyTable` と `DEFAULT_CORPUS_FREQ_PATH` を置き、`features.py` に `_corpus_table()`（`lru_cache(maxsize=1)`）を足す。`FeatureExtractor.__init__` は `corpus_frequency: CorpusFrequencyTable | None = None` を受け、`None` かつ spec が corpus 特徴を宣言しているときだけ既定パスから読む。テストは必ず注入する（既定パスを読むテストを書かない）。

**呼び出し 7 箇所は無変更**。これが `reading_conditional` 方式との違いであり、この選択の理由である。

表が無いのに corpus spec が要求された場合は**明示的に例外**を投げる（黙って全件 OOV にしない）。

### 3.3 特徴の定義（事前登録）

トークン `t = (reading r, surface s)` について、train 域コーパスの生カウント `c(r,s)`・`c(s)`・`c(r) = Σ_s c(r,s)`・総トークン数 `N` を使う。

| 特徴 | 値 |
|---|---|
| `lex.corpus_reading_cond_logprob` | `c(r,s) > 0` のとき `-log(c(r,s)/c(r)) / T`、それ以外 `0.0` |
| `lex.corpus_unigram_logprob` | `c(s) > 0` のとき `-log(c(s)/N) / T`、それ以外 `0.0` |
| `lex.corpus_unseen` | `c(r,s) == 0` のとき `1.0`、それ以外 `0.0` |

設計判断:

- **符号**: 局所モデルは**コスト最小化**である（`training.py:118-119`・`lattice.k_best_viterbi`）。他の lexical 特徴（`lex.ipadic_cost_mapped` 等）はすべて「大きいほど悪い」向きに揃っている。`-log P`（負の対数尤度）にして向きを揃える。名前は issue 記載どおり `..._logprob` のままにするが、docstring と report に**「値は負の対数確率（コスト向き）である」**と明記する
- **平滑化はしない**。未観測時の定数オフセットは `lex.corpus_unseen` の学習重みが担う。α を人手で選ばずに済み、学習可能量になる
- **`c(s) > 0` かつ `c(r,s) == 0` のケースは unigram だけ効く**。unseen フラグはペア基準（`c(r,s)==0`）に統一する。表記自体が未観測の場合は unigram も `0.0` になり、そこは unseen フラグが吸収する。この非対称性は既知の簡略化であり、task1 の被覆率次第で 4 つ目の特徴 `lex.corpus_surface_unseen` を足す（§3.5 の分岐 C）
- **スケール `T`**: 対数確率は `-log P ∈ [0, 25]` 程度で、`lex.ipadic_cost_mapped`（0〜10 程度）より大きい。平均化パーセプトロンは正則化なし・lr 固定なので、スケールは学習軌道に影響する。**既定 `CORPUS_LOGPROB_TEMPERATURE = 10.0`** とし、task1 が報告する `lex.ipadic_cost_mapped` の d1 上の標準偏差と比べて 3 倍以内に収まることを task4 で確認し、収まらなければ task4 で定数を測定値から決め直して**学習前に**固定する（事後調整は禁止）
- **多トークンパス**: 3 特徴ともトークンごとに加算。span_tokens 2〜3 のケース（train の 18.4%）でも定義が壊れない

### 3.4 リーク対策（3 重）

1. **記事 split フィルタ**: `build_d1_data.article_split(article_id, 20260822) == "train"` の記事だけ数える（§2.1 で再現性を実測済み）
2. **評価スイート除外の踏襲**: d1 生成は `load_exclusion_terms()` / `overlaps_evaluation()` で AJIMEE・handmade と重なる記事を除外している（`build_d1_data.py:412-414`）。学習後のモデルは `scripts/evaluate.py --suite all` でこれらのスイートも測るので、**コーパス集計にも同じ除外を必ず適用する**。適用しないと d1 は汚さずに評価スイートを汚す
3. **監査**: ビルダーは manifest に `train_articles_counted` / `dev_test_articles_skipped` / `evaluation_overlap_articles_skipped` を記録し、**カウントした article_id 集合が d1 の dev/test 行の article_id 集合と交わらない**ことを assert する。テストにも同じ検査を置く

残存リスク（レポートに明記する）: 別記事に同一文が重複している場合（Wikipedia のコピペ節）は article_id ベースでは防げない。d1 生成時の正規化グループ重複除去でケース側は対処済みだが、コーパス側の重複までは追わない。ブロッカーではないが既知の限界として記録する。

### 3.5 実測結果が F2 設計に与える影響（事前登録の分岐）

**分岐 A — S1 の「gold と非 gold が同点」率（粒度 c、v2 lattice、test split）**

| 実測値 | 判断 |
|---|---|
| ≥ 10% | 想定どおり。F2 をそのまま進める。F2 の期待効果の上限 = この率 |
| 3〜10% | 進めるが、レポートに「F2 の理論上限は N pt」と**学習前に**事前登録する |
| < 3% | 同点は主因ではない。F2 は続行するが本命扱いを外し、task1 の副産物（gold の順位分布・非同点なのに順序が誤っているケース数）を task7 で主分析にする。issue #19 に所見を追記して着手順序の見直しを提案する |

**分岐 B — S2 の gold 到達不能率（`constrained_best_path` が `None`）**

読みが辞書に無い / gold 表記が lattice で作れないケースは、どんな特徴を足しても解けない。この率が test で 5% を超えるなら、F2 より辞書側（B 項目の固有名詞層）が先である旨を task7 に書く。ただし **F2 の実装自体は止めない**（分岐 A・B は判断材料であって、A 項目の中止条件ではない）。

**分岐 C — task3 が出す (読み,表記) のコーパス被覆率（d1 test の gold ペアのうち `c(r,s) > 0` の割合）**

| 実測値 | 対応 |
|---|---|
| ≥ 70% | そのまま task4 へ |
| 50〜70% | コーパス走査量を 4 倍にして再ビルドし、再測。改善しなければ `lex.corpus_surface_unseen` を 4 つ目の特徴として足す（表記ユニグラムの被覆は必ずペアより高い） |
| < 50% | 走査量 4 倍 + 4 特徴目を両方入れる。それでも < 50% なら task7 に「F2 の効く範囲は d1 の半分未満」と記録し、B 項目（疎な語彙重み + バックオフ）に優先度を移す提案を issue に書く |

いずれの分岐でも **task4/task5 は実行する**。「効かなかった」も測定結果であり、F0 の教訓（新情報ゼロだったから効かなかった）と区別できるのは被覆率の実測があるときだけである。

---

## 4. タスク詳細

各タスクの指示書は `docs/missions/issue19/taskN-mission.md` にある。ここには目的・成果物・依存だけを書く。

### task1 — S 実測（同一特徴ベクトル率 + 被覆率）

- 成果物: `scripts/measure_local_ceiling.py`・`tests/test_measure_local_ceiling.py`・`artifacts/issue19/s-measurement/{ties,coverage}.json`・`docs/missions/issue19/s-measurement-report.md`
- `src/beme/` は変更しない。特徴も重みも追加しない
- 完了条件・検証コマンド: `task1-mission.md`

### task2 — S 実測（最適化要因の排除）

- 成果物: `artifacts/issue19/lr-sweep/` の report.json 群 + `docs/missions/issue19/lr-sweep-report.md`
- 既存 `scripts/train_scorer.py` を**そのまま**使う（`--learning-rate` / `--epochs` は既に CLI 引数）。コード変更なし
- `ws` 未復旧なら未実施として task7 に記録。task3 以降をブロックしない

### task3 — F2-a コーパス頻度表ビルダー

- 成果物: `scripts/build_corpus_freq.py`・`src/beme/corpus_freq.py`（表の読み書き）・`tests/test_corpus_freq.py`・`data/corpus_freq_v1.bin`（gitignore）・`data/corpus_freq_v1.manifest.json`（コミットする）
- 表本体は gitignore（`data/dictionary_v2.tsv` はコミット済みだが、この表は 100MB+ のストリームから再生成可能なため d1 と同じ扱いにする）。manifest に SHA-256・走査バイト数・記事数・dataset revision・除外件数を記録し、レポートはそこを参照する
- GPU 機へは d1 と同じく scp で運ぶ

### task4 — F2-b 特徴実装

- 成果物: `src/beme/features.py` の 3 特徴 + spec 登録・`scripts/build_ablation_weights.py` の 0 初期化・`tests/test_corpus_features.py`
- 登録する spec:
  - `decomposed_v1-no_script-surface_len+corpus_v1`（**C0・本命**・25 + 3 = 28 パラメータ）
  - `decomposed_v1+corpus_v1`（**C1**・30 + 3 = 33 パラメータ・issue 記載どおりの base）
- `lattice.py` / `converter.py` 変更禁止。`extractor_for` の呼び出し 7 箇所も無変更であること（§3.2）

### task5 — F2-c 学習とアブレーション判定

- 成果物: `artifacts/issue19/f2/{C0,C1,A8,A0}/seed{9,17,33}/` + `summary.json`
- `scripts/run_ablations.py` の `CONDITIONS` に C0/C1 を足す。A8/A0 も**同じ日・同じ機で再実行**して比較の足場を揃える（issue #17 の task と同じ流儀）
- 判定は C0 vs A8・C1 vs A0 の McNemar（3 seed ペアリング）

### task6 — F2-d Phase 4 contrast 再測

- 成果物: `artifacts/issue19/f2-contrast/` + 数値表
- **`both_correct` では判定しない**（issue コメント §5 の指示）。overall Top-1 と非 contrast 部分の gain で報告する。F2 は文脈を見ないので `both_correct` は 0 のままであることが予測であり、それを確認するのが目的

### task7 — レポート

- 成果物: `docs/missions/issue19/issue19-report.md`
- issue #19 の「提案するアクション」3 項目への回答・§3.5 の事前登録分岐への判定・artifact SHA-256・実行機名併記
- Phase 3/4 のレポート規約（`docs/missions/issue9-phase4/phase4-report.md` 冒頭）に従う

---

## 5. 全タスク共通の制約

- **`src/beme/lattice.py` と `src/beme/converter.py` は変更禁止**（探索コア）
- `git commit` はローカルのみ。`git push` / PR 作成 / `gh issue` への書き込みは禁止
- 作業はこの worktree（`/home/sarashin/work/beme-19`）の中だけ。他のチェックアウトに `cd` しない
- `git stash` は使わない（stash スタックは他 worktree と共有）。退避が要るなら WIP コミット
- 学習済み重み・artifacts・コーパス表は gitignore 対象。レポートからは SHA-256 で参照する
- `uv run pytest -q` のベースラインは **638 passed / 2 failed**。`tests/test_audit_leakage.py` の 2 件は環境固有（§2.7）。**直さない・合否から除外**。それ以外の 638 件は 1 件も落とさない
- 既存 spec（`legacy_v1` / `decomposed_v1` / A1-A8 / M0 / F0 / F1 / X0）の数値を 1 bit も変えないこと
- 日本語で報告。数値と結論を先頭に

---

## 6. 検証コマンド一覧（タスク横断）

```bash
# 全タスク共通の機械的ゲート
uv run pytest -q                      # 638 passed / 2 failed（audit_leakage の 2 件のみ）
git diff --check
git status --short

# 既存条件の数値が動いていないことの確認（task4 の後に必須）
uv run python -c "
from beme.features import registered_spec_ids, resolve_spec
for s in registered_spec_ids():
    spec = resolve_spec(s)
    print(s, len(spec.lexical_features), len(spec.transition_features))
"

# 探索コアが無変更であることの確認
git diff --stat origin/main -- src/beme/lattice.py src/beme/converter.py   # 空であること
```
