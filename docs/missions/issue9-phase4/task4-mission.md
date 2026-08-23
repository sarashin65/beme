# Task 4: リーク監査の完走（GPU マシン）

## 目的

issue #9 の Phase 4 コメントが「着手前に片付ける小さな前提」として挙げた 3 点目。Phase 3 のリーク監査は miniPC で実行され、`data/d1/*.jsonl` が無いため次の項目が `unverifiable` のまま残っている（`docs/missions/issue9-phase3/phase3-report.md` §8）。

- d1 と評価データの document n-gram overlap
- reading overlap
- train vocabulary overlap
- article ID による split field の再計算

Phase 4 は AJIMEE / Mozc での改善を主張する場面が来るかもしれないので、**d1 JSONL がある GPU 側で `scripts/audit_leakage.py` を完走させる。**

**これは新規実装ではなく実行タスクである。** スクリプトの改造が必要になったら、勝手に広げず報告すること。

最初に `scripts/audit_leakage.py`、`docs/missions/issue9-phase3/phase3-report.md` §8、`docs/missions/issue9-phase3/PLAN.md` §2.3（d1 manifest の除外内容）を読むこと。

## 依存

なし。Task 1・2・3 と並行してよい。Task 5 の前に終わっていること。

## 実行環境

GPU マシン `beme-gpu`（`ssh beme-gpu`）。

- d1 JSONL: `~/beme-d1/repo/data/d1/{train,dev,test}.jsonl`
- **作業ディレクトリは `~/beme-p4` に本ブランチを新規クローンして作る。** `~/beme-p3` は未コミット変更を持つので使わない。`data/d1` へ d1 JSONL を配置する（コピーまたは symlink）
- artifact の出力先: `~/beme-p4/artifacts/issue9-phase4/audit/`

## やること

1. `~/beme-p4` を用意する（クローン + d1 配置）。**`git_commit` が artifact に記録されるよう、必ず git クローンにすること**（Phase 2 の run1 は `git_commit: null` になった）。
2. `scripts/audit_leakage.py` を d1 JSONL ありで全項目実行する。
3. 出力 JSON を `~/beme-p4/artifacts/issue9-phase4/audit/` に保存し、**miniPC 側の worktree にも結果の要約（JSON そのもの、または要点を抜いた JSON）を持ち帰る**。持ち帰り先は `docs/missions/issue9-phase4/` ではなく、Task 7 のレポートから参照できる形にする（置き場所はレポート作成者と揃えること。迷ったら `artifacts/` を miniPC 側にも作り `.gitignore` の扱いを報告する）。
4. `unverifiable` のまま残った項目があれば**名指しで列挙する**。「全部 verified になった」と丸めない。
5. Phase 3 が d1 生成側の主張として記録した **「Mozc は読みと assertion のみで原文を持たないため語彙共有はリークとみなさない」** を、d1 データがある状態で追検証する。追検証できない場合は「できない」と書く。

## 完了条件

- `audit_leakage.py` が d1 JSONL ありで完走し、出力 JSON が保存されている。
- `unverifiable` が残った項目が名指しで列挙されている。
- 監査結果が Phase 4 レポート（Task 7）から参照できる場所にある。

## 検証コマンド

```bash
# GPU マシン側
ssh beme-gpu

# 作業ディレクトリの用意（実際のコマンドは環境に合わせて調整し、実行したものを報告すること）
git clone <repo> ~/beme-p4 && cd ~/beme-p4 && git checkout feat/issue9-phase4
mkdir -p data/d1 && cp ~/beme-d1/repo/data/d1/*.jsonl data/d1/
sha256sum data/d1/*.jsonl   # manifest の値と一致すること

# 監査の完走
uv run python scripts/audit_leakage.py --help
uv run python scripts/audit_leakage.py <全項目が走る引数> \
  --output artifacts/issue9-phase4/audit/leakage.json

# 期待: manifest の SHA-256 と一致
#   test  7b0ad8ff3f9abf11ccfb9488214db0199ecd9b06d07928dea0fb6c8681f6f2ff
#   （train / dev の値は manifest.json の artifacts_sha256 を参照）
```

## 報告に含めること

- 上記コマンドの**実出力**（`sha256sum` の結果を含む）。
- `audit_leakage.py --help` の全文（どの引数で全項目が走るのかを明示するため）。
- 出力 JSON の `unverifiable` / `verified` の項目一覧。
- Mozc の語彙共有に関する追検証の結果、または追検証できない理由。
- `~/beme-p4` の `git rev-parse HEAD` と `git status --short`。

## 共通ルール・制約

- `git commit` / `git push` 禁止（miniPC 側の worktree・GPU 側のクローンの両方）。
- `scripts/audit_leakage.py` を改造しない。改造が必要と判明したら勝手に広げず報告する。
- d1 JSONL を repo にコミットしない。`.gitignore:8` の対象である。
- 監査結果を「リークなし」と丸めない。測れた項目と測れなかった項目を分けて書く。
