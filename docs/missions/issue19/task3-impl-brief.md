# task3 実装指示 — F2-a コーパス頻度表ビルダー（luna 用）

対象 worktree: `/home/sarashin/beme-19`（branch `feat/issue19-f2`・クリーン）。**この worktree の中だけで作業する**（他のチェックアウトに移動しない・`git stash` 禁止）。

## 読むもの（この順）

1. `docs/missions/issue19/task3-opus-plan.md` — **実行計画の正本**。設計・工程・完了条件・制約・リスクが全部ここにある。この計画に従って実装すること
2. `docs/missions/issue19/task3-mission.md` — 仕様（指示書）
3. `docs/missions/issue19/PLAN.md` §2.1 / §3.4 / §3.5
4. `scripts/build_d1_data.py`（import 対象の関数群）・`src/beme/matrix_cost.py`（表の作法の先例）

## 作るもの

- `src/beme/corpus_freq.py`（`CorpusFrequencyTable`・決定的な dump/load）
- `scripts/build_corpus_freq.py`（ビルダー CLI）
- `tests/test_corpus_freq.py`（9 件・リーク検査と監査の負テストを含む）
- `data/corpus_freq_v1.bin`（gitignore）・`data/corpus_freq_v1.manifest.json`（コミット対象）
- `.gitignore` に `data/corpus_freq_v1.bin` を 1 行追加

## 計画への修正 3 点（必ず反映）

1. **`os._exit` は HF ストリーミング経路（`--articles-jsonl` 未指定）の `__main__` のみ**。`--articles-jsonl` 経路では通常の `return status`（テストがサブプロセス経由でビルダーを呼ぶが、直接 import 実行する経路で `os._exit` だと pytest プロセスごと死ぬ）
2. **フィルタ順序は `build_d1_data` と同じ（overlap → split）**。指示書 §2.2 は split → overlap の順だが、d1 manifest の `articles_excluded_for_evaluation_overlap=3` と直接比較可能にするため build_d1_data 側を採る（計画 §2.2 手順 5 の判断）。この意図的逸脱は完了報告に明記すること
3. **pytest のベースラインは 645**（638 は task1 着手前の stale 値）。完了条件 = **645 + 新規 9 = 654 passed / 2 failed**（失敗は `tests/test_audit_leakage.py` の 2 件のみ・直さない）

## 工程（計画 §4 の順）

0. 実装（コード 3 本 + テスト 9 件 + .gitignore）→ `uv run pytest -q tests/test_corpus_freq.py -v` が 9 件 green
1. スモーク 200 記事（2 回・決定性確認・所要時間を記録して 5,104 記事へ線形外挿）
2. 本番ビルド（100MiB 窓・`data/corpus_freq_v1.bin` + manifest）
   - 実行前に `wc -l data/d1/{train,dev,test}.jsonl` で 10,000 / 1,000 / 2,000 を確認（symlink の生存確認）
   - 完了基準の 3 点一致: `utf8_bytes_scanned == 104852271` / `articles.seen == 5104` / `evaluation_overlap_skipped == 3`（d1 manifest と一致しなければ止まって原因を特定）
3. 検証コマンド一式（計画 §4 工程 3 + task3-mission.md §5）
4. 分岐 C の判定（計画 §5）: `coverage.d1_test_gold_pair_coverage` の実測値で ≥0.70 / 0.50〜0.70 / <0.50 を判定。**4 倍再ビルドは行わない**（判定結果を報告し、司令塔が判断してから別途指示する）

## 制約（厳守）

- `src/beme/lattice.py` / `src/beme/converter.py` / `src/beme/features.py` **変更禁止**
- **`article_split` のハッシュ式を写経しない。** `build_d1_data` から import（`node_reading` / `overlaps_evaluation` / `load_exclusion_terms` / `iter_*_articles` / `DEFAULT_*` も同様）
- **`overlaps_evaluation` を落とさない**
- 依存パッケージを増やさない（標準ライブラリ + 既存 `dependency-groups.dev` のみ）
- `git commit` / `git push` / `git stash` をしない。成果物はワークツリーに残す
- `tests/test_audit_leakage.py` の 2 件失敗は直さない
- 分岐 C の閾値（0.70 / 0.50）を動かさない

## 完了報告（日本語・数値と結論を先頭に）

- `leak_audit` の全値（`passed` / `counted_article_ids_intersecting_d1_dev_test` / `misclassified`）
- `coverage` の全値（`d1_test_gold_pair_coverage` / `_span1_only` / `d1_train_gold_pair_coverage` / `coverage_pairs_match`）
- 走査窓 3 点一致の実測値
- `uv run pytest -q` の最終数値（passed / failed）
- `sha256sum data/corpus_freq_v1.bin` と manifest の `table_sha256` の一致
- `git diff --stat origin/main -- src/beme/lattice.py src/beme/converter.py src/beme/features.py` が空であること
- スモークの所要時間と本番ビルドの所要時間
- 分岐 C の判定（実測値と該当する行）
- 上記「修正 3 点」の反映確認
