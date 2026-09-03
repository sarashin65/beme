# task7 — レポート

対象 issue: #19
実行機: ローカル（`/home/sarashin/work/beme-19`）
依存: task5, task6（task2 は任意・未実施ならその旨を書く）
種別: 文書

---

## 0. 目的

issue #19 が立てた問い（「局所モデルは過少適合しているのか」「取り残しは局所側にあるのか」）に、実測で答える 1 本のレポートを書く。

---

## 1. 必ず読むもの

- `gh issue view 19`（本文の「提案するアクション」3 項目・コメントの着手順序表）
- `docs/missions/issue19/PLAN.md`（特に §3.5 の事前登録した分岐）
- `docs/missions/issue19/s-measurement-report.md`（task1）
- `docs/missions/issue19/lr-sweep-report.md`（task2・あれば）
- `data/corpus_freq_v1.manifest.json`（task3）
- task5 / task6 の artifacts と報告
- **書き方の規約**: `docs/missions/issue9-phase4/phase4-report.md` の冒頭スタイル

---

## 2. 書くもの

`docs/missions/issue19/issue19-report.md`。

### 2.1 構成

| 節 | 内容 |
|---|---|
| 0 | **結論**（3〜5 行・数値つき）。F2 は採用されたか。issue #19 の前提（過少適合・局所側に取り残しがある）は支持されたか |
| 1 | 前提の確認 — task1 の同点率・被覆率・gold 到達率。**Opus の辞書レベル 24.9% と d1 候補集合上の実測値の差**とその説明 |
| 2 | 最適化要因の排除 — task2。**未実施なら「未実施であり、28% が容量由来という解釈は未確定である」と明記する**（書かずに済ませない） |
| 3 | F2 の実装 — 3 特徴の定義・符号・スケール定数とその根拠・リーク対策 3 重・表の manifest 数値 |
| 4 | F2 の結果 — C0 vs A8・C1 vs A0 の McNemar・test 誤り件数（3 seed）・学習後の重み（符号込み）・4 指標回帰 |
| 5 | 語彙成分と文脈成分の分離 — task6。**上限/下限の見積りであって分解ではない旨の注記込み** |
| 6 | issue #19 の「提案するアクション」3 項目への回答 |
| 7 | 事前登録した分岐（PLAN §3.5 の A / B / C）への判定 |
| 8 | 限界と残存リスク |
| 9 | 次の着手（着手順序表の B 項目以降への申し送り） |
| 付 | artifact SHA-256 一覧・実行機名・所要時間・Python/依存バージョン |

### 2.2 §8 に必ず書くこと

- **コーパス側の重複**: article_id ベースのフィルタは、別記事に同一文が重複している場合（Wikipedia のコピペ節）を防げない。d1 生成時の正規化グループ重複除去はケース側の対処であり、コーパス側の重複は追っていない
- **F2 の被覆率が上限を決める**: 表に無いペアには `lex.corpus_unseen` しか効かない。被覆率（task3 の実測値）が F2 の効く範囲の上限である
- **task2 未実施なら**: 「28% は容量由来」は未確定であり、F2 の結果を容量の話として解釈するのは暫定である
- **contrast の 64 組は小さい**: 単一 split の差に p 値ラベルを付けていない理由

### 2.3 §6 の回答（issue 本文の 3 アクション）

1. **「最終的な平均化重みで train split を評価する」** — Hermes が既に実測済み（28.2%）。task2 が回っていれば lr/epoch を振っても帯に収まるかを添える
2. **「語彙化した局所モデルの検討」** — F2 は「疎な語彙重み」ではなく「外部頻度表を 3 個のスカラー特徴で読む」形である。**この違いを明示し**、F2 の結果が疎重み（B 項目）の期待値に何を意味するかを書く
3. **「分離実験」** — task6 の結果

### 2.4 issue への追記案

レポート本体とは別に、**`gh issue comment` で issue #19 に貼る短い要約案**（15 行以内）を `docs/missions/issue19/issue19-comment-draft.md` に用意する。**投稿はしない**（Hermes が行う）。

---

## 3. 完了条件

- [ ] `docs/missions/issue19/issue19-report.md` が §2.1 の 10 節を持つ
- [ ] 先頭に結論が数値つきで書かれている
- [ ] PLAN §3.5 の分岐 A / B / C すべてに判定が書かれている
- [ ] task2 が未実施なら、その旨と解釈への影響が §2 と §8 の両方に書かれている
- [ ] artifact の SHA-256・実行機名・所要時間が付録にある
- [ ] `docs/missions/issue19/issue19-comment-draft.md` がある（投稿はしていない）
- [ ] `git diff --stat origin/main -- src/ scripts/ tests/` が **空**（このタスクは文書のみ）
- [ ] git commit / push をしていない

---

## 4. 検証コマンド

```bash
cd /home/sarashin/work/beme-19

git diff --stat origin/main -- src/ scripts/ tests/    # 空であること
git diff --check
git status --short

# 参照している artifact が実在し、記載の SHA-256 と一致すること
uv run python -c "
import re, hashlib, pathlib
text = pathlib.Path('docs/missions/issue19/issue19-report.md').read_text(encoding='utf-8')
for path, digest in re.findall(r'([\w./+-]+)\s*[:|]\s*\`?([0-9a-f]{64})', text):
    p = pathlib.Path(path)
    if p.is_file():
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        print('OK ' if actual == digest else 'MISMATCH ', path)
    else:
        print('(not local) ', path)
"
```

---

## 5. 制約（厳守）

- **コードを変更しない。** 文書だけのタスク
- **`src/beme/lattice.py` / `src/beme/converter.py` 変更禁止**（探索コア）
- **判別不能なものは「判別不能」と明記する。** 単一 seed / 単一 split の p 値で採用ラベルを付けない（Phase 3/4 のレポート規約）
- 実測していない数値を書かない。参照値（Phase 4 の L/LR 等）は**参照値であることと機・日付が異なることを明記**する
- `gh issue comment` / `gh issue edit` をしない（下書きを用意するだけ）
- `git commit` / `git push` をしない。作業はこの worktree の中だけ。`git stash` を使わない
- 日本語で書く。数値と結論を先頭に
