# task4 — F2-b: コーパス頻度 3 特徴の実装と feature spec 登録

対象 issue: #19（A 項目・F2 の後半）
実行機: ローカル（`/home/sarashin/work/beme-19`）
依存: task3（`CorpusFrequencyTable` と `data/corpus_freq_v1.bin` が要る）
種別: 実装

---

## 0. 目的

既存の局所モデルに 3 個の特徴を**追加**する。追加重みは 3 個、そこに載る知識は 34 万ペアの外部頻度表である。IPADIC コストの 8,821 段階の量子化では割れない同点を、連続値で割ることを狙う。

| 特徴名 | 意味 |
|---|---|
| `lex.corpus_reading_cond_logprob` | 読み条件付きの表記選択コスト = `-log P(表記 \| 読み)` |
| `lex.corpus_unigram_logprob` | 表記ユニグラムのコスト = `-log P(表記)` |
| `lex.corpus_unseen` | ペア未観測フラグ |

---

## 1. 必ず読むもの

- `gh issue view 19`（コメント §4。3 特徴の定義が正本）
- `docs/missions/issue19/PLAN.md` §2.4（**なぜ `_edge_cache` も k-best も無変更でよいか**）・§2.5（base は A0 でなく A8）・§2.6（表の遅延ロードは matrix 方式）・§3.3（**特徴の定義・符号・スケールの事前登録**）
- `docs/missions/issue19/s-measurement-report.md`（task1。`lex.ipadic_cost_mapped` の標準偏差をここから取る）
- `data/corpus_freq_v1.manifest.json`（task3。`coverage` を見て 4 特徴目が要るか判断する）
- `src/beme/features.py` — 全部。特に:
  - `FeatureExtractor.__init__` / `lexical_features`（`reading_conditional` の扱いが**反面教師**）
  - `_matrix_table()`（222-226 行）— **こちらが手本**。呼び出し側に何も渡させない遅延シングルトン
  - `_SPEC_REGISTRY` / `resolve_spec` / `registered_spec_ids`
  - `FREQ_LEXICAL_FEATURES` / `FREQ_V1_SPEC`（243-255 行）— 特徴群を足す既存パターン
- `src/beme/scorer.py:44-59`（`_edge_cache` が `Token` をキーにしている理由のコメント。この不変条件を壊さないこと）
- `src/beme/weights.py:100-110`（`resolve_spec` 経由の特徴名検証。新 spec を登録しないと重みが全部弾かれる）
- `scripts/build_ablation_weights.py`（新特徴を 0 初期化する既存パターン）
- `tests/test_reading_conditional.py`・`tests/test_feature_registry.py`（テストの作法）

---

## 2. 実装

### 2.1 特徴の値（PLAN §3.3 の事前登録。勝手に変えない）

トークン `t = (reading r, surface s)` について、表から `c(r,s)` / `c(s)` / `c(r)` / `N` を引く。

| 特徴 | 値 |
|---|---|
| `lex.corpus_reading_cond_logprob` | `c(r,s) > 0` のとき `-log(c(r,s)/c(r)) / T`、それ以外 `0.0` |
| `lex.corpus_unigram_logprob` | `c(s) > 0` のとき `-log(c(s)/N) / T`、それ以外 `0.0` |
| `lex.corpus_unseen` | `c(r,s) == 0` のとき `1.0`、それ以外 `0.0` |

**符号**: 局所モデルはコスト最小化であり、他の lexical 特徴はすべて「大きいほど悪い」向きに揃っている（`lex.ipadic_cost_mapped` 等）。よって**負の対数確率**にして向きを揃える。名前は issue 記載どおり `..._logprob` のままにするが、**docstring に「値は負の対数確率（コスト向き）である」と明記する**。

**平滑化はしない**。未観測時の定数オフセットは `lex.corpus_unseen` の学習重みが担う。α を人手で選ばずに済み、学習可能量になる。

**スケール `T = CORPUS_LOGPROB_TEMPERATURE`**: `src/beme/corpus_freq.py` に定数として置く。既定 **10.0**。task1 が報告した `lex.ipadic_cost_mapped` の d1 上の標準偏差に対して、この 3 特徴の標準偏差が **3 倍以内**に収まることを確認する（確認コードは §5 の検証コマンドにある）。収まらなければ `T` を測定値から決め直し、**学習前に固定する**。学習後に動かすのは禁止。決めた値と根拠を報告に書く。

**4 特徴目**: task3 の manifest の `coverage.d1_test_gold_pair_coverage` が **0.70 未満**だった場合のみ、`lex.corpus_surface_unseen`（`c(s) == 0` で `1.0`）を足す。0.70 以上なら足さない（PLAN §3.5 分岐 C）。

### 2.2 表の受け渡しは matrix 方式（PLAN §3.2）

```python
@lru_cache(maxsize=1)
def _corpus_table():
    from .corpus_freq import DEFAULT_CORPUS_FREQ_PATH, CorpusFrequencyTable
    return CorpusFrequencyTable.load(DEFAULT_CORPUS_FREQ_PATH)
```

`FeatureExtractor.__init__` に `corpus_frequency: "CorpusFrequencyTable | None" = None` を**キーワード引数として**足す。`lexical_features` の中で、spec が corpus 特徴を宣言している場合にのみ `self.corpus_frequency or _corpus_table()` を解決する。

**`extractor_for()` の第 2 引数（`reading_conditional`）のシグネチャは変えない。** これが matrix 方式を選ぶ理由である: `extractor_for` の呼び出しは 7 箇所（`train_scorer` / `oracle_recall` / `measure_cost_scale` / `diagnose_cases` / `evaluate` / `rerank_eval` / tests）あり、そのどれも変更しない。`extractor_for` に `corpus_frequency=` を渡せる**任意**キーワードを足すのはよい（テストが注入に使う）。

**表が無いのに corpus spec が要求されたら明示的な例外**を投げる。黙って全件 OOV にしない（それは静かに別モデルを学習させることになる）。

### 2.3 spec の登録

```python
CORPUS_LEXICAL_FEATURES = (
    "lex.corpus_reading_cond_logprob",
    "lex.corpus_unigram_logprob",
    "lex.corpus_unseen",
)
```

`_SPEC_REGISTRY` に 2 つ足す:

| spec id | 中身 | パラメータ数 | 条件名 |
|---|---|---:|---|
| `decomposed_v1-no_script-surface_len+corpus_v1` | **A8**(lex 9 + trans 16) + corpus 3 | 28 | **C0（本命）** |
| `decomposed_v1+corpus_v1` | **A0**(lex 14 + trans 16) + corpus 3 | 33 | C1 |

C0 を本命にするのは、現行 base が A8 だからである（issue #17 / commit `a4b690b` で採用済み。PLAN §2.5）。C1 は issue #19 コメントの記載（「既存 30 特徴に追加」）どおりの条件で、A0 との対比を取るために残す。

`transition_state` は両方 `("pos",)` のまま。**k-best も `state_key_function` も触らない。**

### 2.4 初期重みの生成

`scripts/build_ablation_weights.py` の `for name in ("lex.reading_cond_logprob", "lex.reading_candidate_count")` のループに corpus 3 特徴を足し、**0 初期化**する。既存の生成対象条件（`decomposed_v1-` / `decomposed_v1+` 接頭辞）に新 spec が自動で乗ることを確認する。

生成:

```bash
uv run python scripts/build_ablation_weights.py
```

### 2.5 `tests/test_corpus_features.py`

- 手作りの `CorpusFrequencyTable`（`from_counts`）を注入し、3 特徴の値が §2.1 の式どおりに出ること（1 件ずつ手計算した期待値を書く）
- **未観測ペアで `lex.corpus_unseen == 1.0` かつ cond_logprob が 0.0**、かつ `c(s) > 0` なら unigram だけ値を持つこと
- **spec が corpus 特徴を宣言していない場合、これらの名前が出力に現れないこと**（`decomposed_v1` の出力が 1 bit も変わらないこと）
- 表を注入せず、既定パスにも表が無い状態で corpus spec の `lexical_features` を呼ぶと**例外**になること
- 新 spec id で `Weights.load` が通ること（3 特徴が unknown 扱いされないこと）
- **位置非依存の検査**: 同じ `Token` を異なる `start`/`end` の `LatticeEdge` に載せて、3 特徴が同値であること（`_edge_cache` が `Token` キーで正しいことの根拠）
- 多トークンパスで `path_features` の 3 特徴がトークンごとの和になること

---

## 3. 変えてはいけないもの

- **`src/beme/lattice.py` / `src/beme/converter.py` は 1 行も変更しない**
- `LinearScorer._edge_cache` / `_transition_cache` のキー設計を変更しない（3 特徴は `token.reading` と `token.surface` のみの関数なので、変更は不要である）
- `extractor_for()` の既存の位置引数シグネチャ
- 既存 spec（`legacy_v1` / `decomposed_v1` / A1-A8 / M0 / M1 / F0 / F1 / X0）の特徴集合と出力値。**1 bit も変えない**

---

## 4. 完了条件

- [ ] 3 特徴（分岐 C 次第で 4 特徴）が `src/beme/features.py` に実装されている
- [ ] `decomposed_v1-no_script-surface_len+corpus_v1` と `decomposed_v1+corpus_v1` が `_SPEC_REGISTRY` に登録され、`resolve_spec` / `Weights.load` を通る
- [ ] `_corpus_table()` が遅延シングルトンで、`extractor_for` の既存 7 呼び出し箇所が**無変更**
- [ ] `scripts/build_ablation_weights.py` が新 spec の初期重み（corpus 3 特徴 = 0.0）を生成する
- [ ] `data/weights/ablations/decomposed_v1-no_script-surface_len+corpus_v1.json` と `decomposed_v1+corpus_v1.json` が生成されている
- [ ] `tests/test_corpus_features.py` が §2.5 の 7 項目を検査している
- [ ] `uv run pytest -q` が **638 + 新規テスト数 passed / 2 failed**（失敗は `tests/test_audit_leakage.py` の 2 件のみ）
- [ ] `CORPUS_LOGPROB_TEMPERATURE` の値と、それを選んだ根拠（task1 の標準偏差との比）が報告に書かれている
- [ ] `git diff --stat origin/main -- src/beme/lattice.py src/beme/converter.py` が **空**
- [ ] `git diff origin/main -- scripts/evaluate.py scripts/rerank_eval.py scripts/oracle_recall.py scripts/measure_cost_scale.py scripts/diagnose_cases.py` が **空**（呼び出し側無変更の証明）
- [ ] git commit / push をしていない

---

## 5. 検証コマンド

```bash
cd /home/sarashin/work/beme-19

uv run pytest -q
uv run pytest -q tests/test_corpus_features.py tests/test_feature_registry.py tests/test_reading_conditional.py -v

# 1. spec 一覧とパラメータ数（C0 が 9+16+3=28、C1 が 14+16+3=33 であること）
uv run python -c "
from beme.features import registered_spec_ids, resolve_spec
for s in registered_spec_ids():
    sp = resolve_spec(s)
    print(f'{s:52s} lex={len(sp.lexical_features):2d} trans={len(sp.transition_features):2d} total={len(sp.lexical_features)+len(sp.transition_features):2d}')
"

# 2. 既存 spec の出力が 1 bit も変わっていないこと
uv run python -c "
from beme.features import extractor_for
from beme.lattice import LatticeEdge, Token
t = Token('とうきょう','東京','proper_noun',3.5,raw_cost=4000,left_id=1,right_id=1,source='ipadic')
e = LatticeEdge(0,5,t)
print(extractor_for('decomposed_v1').lexical_features(e))
print(extractor_for('decomposed_v1-no_script-surface_len').lexical_features(e))
"

# 3. 新 spec の重みが load できること
uv run python -c "
from beme.weights import Weights
for s in ('decomposed_v1-no_script-surface_len+corpus_v1','decomposed_v1+corpus_v1'):
    w = Weights.load(f'data/weights/ablations/{s}.json', feature_spec=s)
    print(s, len(w), {k:v for k,v in w.items() if 'corpus' in k})
"

# 4. スケール確認（3 特徴の std が lex.ipadic_cost_mapped の 3 倍以内か）
#    task1 の feature_distributions と、d1 test 上での corpus 特徴の分布を比べる
uv run python scripts/measure_local_ceiling.py \
  --data data/d1/test.jsonl --limit 500 \
  --specs decomposed_v1-no_script-surface_len+corpus_v1 \
  --output /tmp/claude-1000/scale-check.json
uv run python -c "
import json
d=json.load(open('/tmp/claude-1000/scale-check.json'))
print(json.dumps(d['feature_distributions'],ensure_ascii=False,indent=2))
"

# 5. 呼び出し側・探索コアが無変更であること
git diff --stat origin/main -- src/beme/lattice.py src/beme/converter.py     # 空
git diff --stat origin/main -- scripts/evaluate.py scripts/rerank_eval.py \
  scripts/oracle_recall.py scripts/measure_cost_scale.py scripts/diagnose_cases.py  # 空
git diff --check
git status --short
```

---

## 6. 制約（厳守）

- **`src/beme/lattice.py` / `src/beme/converter.py` 変更禁止**（探索コア）
- `_edge_cache` / k-best / `state_key_function` を触らない。触る必要があると判断したら**実装せずに報告する**（PLAN §2.4 の前提が崩れたということなので、設計を見直す）
- `extractor_for` の呼び出し 7 箇所を変更しない
- 既存 spec の数値を 1 bit も変えない
- 特徴の定義・符号・平滑化なし・スケール定数の決め方（PLAN §3.3）を勝手に変えない。変更が要ると判断したら実装せずに報告に含める
- `T`（スケール定数）を**学習後に**動かさない
- `git commit` / `git push` をしない。作業はこの worktree の中だけ。`git stash` を使わない
- `tests/test_audit_leakage.py` の既存 2 件の失敗は**直さない**（環境固有。PLAN §2.7）
- 日本語で報告。数値と結論を先頭に
