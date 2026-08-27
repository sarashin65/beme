# task1 — S 実測: d1 候補集合上の同一特徴ベクトル率と被覆率

対象 issue: #19（S 項目）
実行機: ローカル（`/home/sarashin/work/beme-19`・branch `feat/issue19-f2`）
依存: なし
種別: 実装 + 測定（`src/beme/` は変更しない）

---

## 0. 目的

F2（コーパス頻度特徴）の設計判断に必要な 2 つの数字を、d1 の実データ上で確定させる。

1. **S1**: 局所モデルが実際に順位付けする候補集合の中で、「特徴ベクトルが完全一致していて、どんな重みでも順位付け不能な候補対」がどれだけあるか
2. **S2**: d1 train/test の読み・(読み,表記) が辞書にどれだけ載っているか。gold がそもそも lattice で作れないケースがどれだけあるか

Opus のアドバイザリーは**辞書レベル**で「多候補読み 72,013 個のうち 24.9% が同一 14 次元 lexical ベクトルの候補対を含む」と算出した。これは d1 の実際の候補集合上の値**ではない**。あなたが出すのは後者である。

---

## 1. 必ず読むもの

- `gh issue view 19`（本文 + 2026-08-26 コメント）— 正本
- `docs/missions/issue19/PLAN.md` §2（確定した前提）・§3.1（なぜ 4 粒度 × 2 候補集合か）・§3.5（この測定結果が F2 設計をどう分岐させるか）
- `src/beme/features.py`（`FeatureExtractor.lexical_features` / `path_features` / `resolve_spec`）
- `src/beme/scorer.py`（`LinearScorer`）
- `src/beme/oracle.py`（`constrained_best_path`）
- `src/beme/lattice.py`（読むだけ。`Lattice.k_best` / `build_lattice` の使い方）
- `src/beme/training_data.py`（`load_jsonl` / `TrainingExample`）
- `scripts/diagnose_cases.py`（既存の「d1 を読んで converter を 2 本立てて比較する」スクリプトの書き方。CLI 引数・`_atomic_json`・SHA-256 記録の作法をここから真似る）

---

## 2. 作るもの

### 2.1 `scripts/measure_local_ceiling.py`

既存スクリプト（`scripts/diagnose_cases.py` / `scripts/measure_cost_scale.py`）の作法に合わせる: `argparse`・`REPOSITORY_ROOT` / `SRC_ROOT` の sys.path 挿入・`_atomic_json` での出力・入力ファイルの SHA-256 を出力 JSON に記録。

CLI:

```
--data PATH            測定対象の d1 JSONL（複数回指定可。既定: data/d1/train.jsonl data/d1/test.jsonl）
--dictionary PATH      既定 data/dictionary_v2.tsv
--weights PATH         コスト同点（粒度 d）の評価に使う重み
                       既定 data/weights/ablations/decomposed_v1-no_script-surface_len.json
--specs ID[,ID...]     特徴ベクトル同点を測る spec
                       既定 decomposed_v1,decomposed_v1-no_script-surface_len
--k INT                lattice k-best の k（既定 64）
--beam-width INT       既定 64
--limit INT            デバッグ用に先頭 N 件だけ処理（既定なし = 全件）
--output PATH          必須。JSON 出力先
```

### 2.2 測定内容 S1（同一特徴ベクトル率）

**候補集合を 2 通り作る。**

- `lattice`（**主測定**）: `build_lattice(reading, dictionary, allow_literals=True)` → `lattice.k_best(beam_width, scorer)` の上位 k パス。学習/評価が実際に順位付けする集合そのもの（`src/beme/training.py:178` と同じ張り方にすること）
- `jsonl`（副測定）: JSONL の `candidates` 列。**`data/dictionary.tsv`（v1・sha `efcdf5a1…`）由来**であって v2 ではない（`data/d1/manifest.json` の `candidate_generation.dictionary`）。表記文字列しか無いので、`oracle.constrained_best_path(lattice, surface, scorer)` でその表記を作る最安パスを取り、取れなかった候補は `unreachable` として別カウントする

**各候補パスについて 4 種のキーを作る。**

| 粒度 | キー |
|---|---|
| (a) `lex14` | `decomposed_v1` の **lexical 特徴だけ**を `path_features` で集計したベクトル |
| (b) `lex9` | `decomposed_v1-no_script-surface_len`（A8）の lexical だけ |
| (c) `full25` | A8 の lexical + transition 全部（= `path_features` の全出力） |
| (d) `cost` | `--weights` の重みでのパスコスト |

比較は**厳密一致**で行う。浮動小数の丸めを跨がせないため、キーは `tuple(sorted((name, repr(value)) for ...))` のように**値のビット表現**で作ること（`round()` でまとめない）。(d) のコスト同点だけは `math.isclose(a, b, rel_tol=0, abs_tol=1e-12)` を使ってよい（理由をコードコメントに書く）。

**各粒度について、ケース単位で次の 3 率を出す。**

1. `any_tied_pair_rate` — 候補集合に同点対が 1 組以上あるケースの割合
2. `top2_tied_rate` — パスコスト最安 2 候補が同点であるケースの割合
3. `gold_vs_nongold_tied_rate` — **gold パスと非 gold 候補が同点であるケースの割合**（← §3.5 分岐 A の判定に使う唯一の指標）

gold パスは `constrained_best_path(lattice, example.target_surface, scorer)` で取る。`None`（到達不能）のケースはこの 3 率の分母から外し、件数を別途 `gold_unreachable` として報告する。

**副産物として必ず出す**（分岐 A が「< 3%」に転んだときの主分析材料になる）:

- `gold_rank_histogram` — k-best 内での gold の順位分布（1, 2, 3, 4-10, 11-64, not_in_kbest）
- `misranked_not_tied` — gold が 1 位でないが、gold と 1 位候補が (c) 粒度で同点**でない**ケース数。これは「重みを直せば取れるはずのケース」の件数である
- **特徴ごとの分布統計**: `decomposed_v1` の各 lexical 特徴のパス集計値の mean / std / min / max（split 別）。`lex.ipadic_cost_mapped` の std は F2 のスケール定数 `CORPUS_LOGPROB_TEMPERATURE` を決めるのに task4 が使う（PLAN §3.3）

### 2.3 測定内容 S2（被覆率）

split ごとに次を出す。`--data` で渡された全 split について出すこと。

| 指標 | 定義 |
|---|---|
| `reading_in_dictionary_rate` | `dictionary.entries_for_reading(example.reading)` が空でないケースの割合（読みまるごとが 1 エントリになる率） |
| `reading_lattice_connectable_rate` | `build_lattice(reading, ...)` が始端から終端まで繋がるパスを 1 本以上持つ割合 |
| `gold_pair_in_dictionary_rate` | `(reading, target_surface)` が辞書エントリとして**そのまま**存在する割合 |
| `gold_reachable_rate` | `constrained_best_path` が `None` でない割合（= 1 − gold 到達不能率） |
| `gold_token_count_histogram` | gold パスのトークン数分布（1 / 2 / 3 / 4+）。F2 の 3 特徴はトークンごとに効くので、多トークン gold の比率は F2 の効き方に直結する |
| `unique_readings` / `unique_gold_pairs` | 異なり数。`data/d1/manifest.json` の `unique_readings` と突き合わせて整合を確認する |

さらに **`gold_pair_multiset`**（`(reading, surface, count)` の一覧、split 別）を別ファイル `coverage_pairs.jsonl` に書き出すこと。task3 が「このペアのうち何割がコーパスに現れたか」（PLAN §3.5 分岐 C）を測るのに、d1 を再走査せずに使える。

### 2.4 `tests/test_measure_local_ceiling.py`

既存 `tests/test_measure_cost_scale.py` / `tests/test_diagnose_cases.py` の作法に合わせる。最低限:

- 小さな手作り辞書 + 手作り JSONL（`tmp_path`）でスクリプトを end-to-end 実行し、JSON の必須キーが揃うこと
- **同点が発生する構成を意図的に作って、`any_tied_pair_rate` が 1.0 になること**（同一 cost・同一 pos・同一長の 2 表記を持つ読みを 1 つ作る）
- **同点が発生しない構成で 0.0 になること**
- `gold` が辞書に無いケースで `gold_unreachable` が 1 になり、3 率の分母から外れること
- `data/d1/*.jsonl` に依存しないこと（gitignore されているため CI で存在しない）

---

## 3. 出力仕様

`--output` に JSON を 1 本（既定 `artifacts/issue19/s-measurement/ties.json`）、隣に `coverage_pairs.jsonl`。

```json
{
  "cli": { "...": "..." },
  "git_commit": "...",
  "dictionary_sha256": "...",
  "weights_sha256": "...",
  "inputs": [{"path": "data/d1/test.jsonl", "sha256": "...", "examples": 2000}],
  "ties": {
    "test": {
      "lattice": {
        "cases": 2000,
        "gold_unreachable": 0,
        "lex14":  {"any_tied_pair_rate": 0.0, "top2_tied_rate": 0.0, "gold_vs_nongold_tied_rate": 0.0},
        "lex9":   {"...": "..."},
        "full25": {"...": "..."},
        "cost":   {"...": "..."},
        "gold_rank_histogram": {"1": 0, "2": 0, "3": 0, "4-10": 0, "11-64": 0, "not_in_kbest": 0},
        "misranked_not_tied": 0
      },
      "jsonl": { "...": "同じ構造 + candidate_unreachable" }
    }
  },
  "coverage": { "test": { "reading_in_dictionary_rate": 0.0, "...": "..." } },
  "feature_distributions": { "test": { "lex.ipadic_cost_mapped": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0} } }
}
```

---

## 4. レポート

`docs/missions/issue19/s-measurement-report.md` を書く。

- 先頭に**結論と数値**（PLAN §3.5 の分岐 A・B のどれに該当したか、その根拠となる 1 つの数字を明示）
- 4 粒度 × 2 候補集合 × 3 率の表（train / test）
- 被覆率の表
- **Opus の辞書レベル 24.9% と、あなたが測った d1 候補集合上の (a) `lex14` の率との比較と、差がある場合の説明**（辞書には多候補読みが 72,013 あるが d1 の読みは 8,964 種・異なる分布から引かれている、等）
- 分岐 A / B の判定と、それが F2 の期待値に何を意味するか
- 実行環境（機名・Python version・所要時間）・入力の SHA-256
- 単一 split の値に p 値ラベルを付けない・判別不能なものは「判別不能」と明記する（Phase 3/4 のレポート規約）

---

## 5. 完了条件

- [ ] `scripts/measure_local_ceiling.py` が上記 CLI で動く
- [ ] `tests/test_measure_local_ceiling.py` が追加され、§2.4 の 5 項目を検査している
- [ ] `uv run pytest -q` が **638 + 新規テスト数 passed / 2 failed**（失敗は `tests/test_audit_leakage.py` の 2 件のみ）
- [ ] `data/d1/{train,test}.jsonl` 全件（12,000 ケース）で実測が完走し、`artifacts/issue19/s-measurement/ties.json` と `coverage_pairs.jsonl` が生成されている
- [ ] `docs/missions/issue19/s-measurement-report.md` が書かれ、PLAN §3.5 の分岐 A・B の判定が明記されている
- [ ] `git diff --stat origin/main -- src/beme/` が **空**（`src/beme/` は 1 行も変更していない）
- [ ] `git diff --check` が無出力
- [ ] git commit / push をしていない

---

## 6. 検証コマンド

```bash
cd /home/sarashin/work/beme-19

# 1. テスト
uv run pytest -q
uv run pytest -q tests/test_measure_local_ceiling.py -v

# 2. スモーク（先頭 50 件・数秒で終わること）
uv run python scripts/measure_local_ceiling.py \
  --data data/d1/test.jsonl --limit 50 \
  --output /tmp/claude-1000/smoke.json
uv run python -c "import json;d=json.load(open('/tmp/claude-1000/smoke.json'));print(json.dumps(d['ties'],ensure_ascii=False,indent=2)[:2000])"

# 3. 本測定（train + test 全件）
mkdir -p artifacts/issue19/s-measurement
uv run python scripts/measure_local_ceiling.py \
  --data data/d1/train.jsonl --data data/d1/test.jsonl \
  --output artifacts/issue19/s-measurement/ties.json

# 4. 探索コア無変更の確認
git diff --stat origin/main -- src/beme/            # 空であること
git diff --check
git status --short
```

---

## 7. 制約（厳守）

- **`src/beme/` は 1 行も変更しない。** このタスクは測定であって実装ではない。特徴も spec も重みも追加しない
- **`src/beme/lattice.py` / `src/beme/converter.py` 変更禁止**（探索コア）
- `git commit` / `git push` をしない
- 作業はこの worktree の中だけ。他のチェックアウトに `cd` しない。`git stash` を使わない
- `tests/test_audit_leakage.py` の既存 2 件の失敗は**直さない**（`data/d1/*.jsonl` がこの worktree に置いてあることによる環境固有の失敗。PLAN §2.7）
- 12,000 ケース × 64 パスの測定は数十分かかりうる。長引くならデタッチ起動（`nohup setsid ... < /dev/null &`）で回し、待つ間に別の完了条件を進める
- `artifacts/` は gitignore 対象。レポートからは SHA-256 で参照する
- **設計判断を勝手に変えない。** 4 粒度・2 候補集合・3 率の構成は PLAN §3.1 で決めてある。変更が必要と判断したら実装せずに報告に含める
- 日本語で報告。数値と結論を先頭に
