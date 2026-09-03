# task6 — F2-d: Phase 4 contrast セットの再測（語彙成分と文脈成分の分離）

対象 issue: #19（A 項目・提案アクション 3「分離実験」）
実行機: GPU マシン `ssh ws`
依存: task5（F2 の学習済み重みが要る）
種別: 実行 + 分析

---

## 0. 目的

Phase 4 の LR が買った **+10 pt** のうち、どれだけが「本物の文脈的曖昧性解消」で、どれだけが「大きいモデルが単語を知っているだけ」なのかを分離する。

issue #19 の論理:

- `both_correct`（contrast ペアの両方を当てる）は L=0 → LR=21/64 だった。文脈盲のモデルは原理的に 0 しか取れないので、**本物の文脈解消は起きている**
- しかし overall Top-1 の +10 pt のうち語彙成分がどれだけかは未分離
- **F2 は文脈を見ない局所モデルである**。したがって F2 の `both_correct` は 0 のままのはずで、そこが動かないまま overall がどれだけ上がるかが、そのまま「語彙成分」の見積りになる

---

## 1. 必ず読むもの

- `gh issue view 19`（本文「提案するアクション 3」・コメント §5 の**「both_correct でなく overall / 非 contrast gain で」という明示的な指示**）
- `docs/missions/issue9-phase4/phase4-report.md` — Phase 4 の contrast 測定の完全な手続き。**同じ手続きを再現する**こと
- `docs/missions/issue19/PLAN.md` §1・§5
- `scripts/rerank_eval.py`（356 行付近の extractor 構築。`--weights` の spec が自動解決されること）
- `src/beme/rerank_stats.py`
- `src/beme/training_data.py`（`contrast_pair_id` フィールド）

---

## 2. やること

### 2.1 測る条件

| 条件 | 説明 |
|---|---|
| A8 | 現行 base の局所モデル単体（task5 で同じ日に学習したもの） |
| **C0** | F2 局所モデル単体（task5 の本命） |
| L | Phase 4 の局所モデル単体（過去 artifact・参照値） |
| LR | Phase 4 の局所 + reranker（過去 artifact・参照値） |

L / LR は再実行しない。Phase 4 レポートの数値を**参照値として引く**（機・日付が違うことをレポートに明記する）。

### 2.2 出す指標

d1 test の contrast ペア **64 組**と、非 contrast の残り（2,000 − contrast ケース）に分けて:

| 指標 | 定義 |
|---|---|
| `overall_top1` | test 全体の Top-1 正解率 |
| `non_contrast_top1` | contrast ペアに属さないケースの Top-1 正解率 |
| `contrast_top1` | contrast ペアに属するケースの Top-1 正解率（ケース単位） |
| `both_correct` | contrast ペアの**両方**を当てた組数 / 64 |

**主判定は `non_contrast_top1` の gain**（A8 → C0）である。`both_correct` は**予測の確認**であって判定材料ではない（issue コメント §5 の指示）。

### 2.3 予測（事前登録）

1. **C0 の `both_correct` は A8 と同じ（ほぼ 0）**。F2 は文脈を見ないので、contrast ペアの両側は同じ読みに対して同じ局所スコアを返す。ここが 0 でなければ**測定か実装のどちらかが壊れている**（contrast ペアの定義を読み直すこと）
2. C0 の `non_contrast_top1` は A8 より上がる（task5 で採用判定が出ていれば）
3. LR の overall gain のうち、C0 が A8 に対して稼いだ分は「語彙成分」であり、**その分は CE より安く取れる**ことになる

### 2.4 分離の報告の形

```
LR の overall gain (Phase 4)        : +X.X pt   （参照値・機と日付が異なる）
C0 の overall gain (対 A8・同日同機) : +Y.Y pt   ← 語彙成分の下限見積り
残差 (X - Y)                        : +Z.Z pt   ← 文脈成分 + 交絡の上限見積り
C0 の both_correct                  : N/64      ← 0 であることが予測
```

**この引き算は上限/下限の見積りであって分解ではない。** L と C0 は base も学習日も違い、LR の gain は L 比である。レポートには「この 3 行は厳密な分解ではなく、語彙成分の下限と文脈成分の上限を挟むための粗い見積りである」と**必ず注記する**。単一 split の差に p 値ラベルを付けない。

---

## 3. 実行

```bash
# GPU 側（task5 で環境は整っている）
ssh ws 'cd ~/beme-19 && uv run python scripts/rerank_eval.py \
  --data data/d1/test.jsonl \
  --weights artifacts/issue19-f2/C0/seed9/weights.json \
  --reranker mock ... '   # 実際の引数は scripts/rerank_eval.py を読んで確定すること

scp -r ws:~/beme-19/artifacts/issue19-f2-contrast artifacts/
```

`scripts/rerank_eval.py` の CLI は**コードを読んで確定する**。Phase 4 レポートに実際に使われたコマンド行があるなら、それをそのまま流用する（手続きを揃えるのが目的なので、勝手に変えない）。

3 seed 分（9/17/33）測り、中央値で報告する。

---

## 4. 完了条件

- [ ] A8 / C0 について §2.2 の 4 指標が 3 seed 分測られている
- [ ] `both_correct` が予測どおり（≈0）であることが確認されている。**違っていたら原因を調べて報告する**（そのまま先に進まない）
- [ ] §2.4 の分離の見積りが、注記つきで書かれている
- [ ] Phase 4 の L / LR 参照値との比較で、機・日付が異なることが明記されている
- [ ] artifacts がローカルに回収され、SHA-256 が記録されている
- [ ] `git diff --stat origin/main -- src/beme/` が **空**（このタスクは実行のみ）
- [ ] git commit / push をしていない

---

## 5. 検証コマンド

```bash
cd /home/sarashin/work/beme-19

git diff --stat origin/main -- src/ scripts/ tests/     # 空であること
git status --short

uv run python -c "
import json,glob
for p in sorted(glob.glob('artifacts/issue19-f2-contrast/**/*.json', recursive=True)):
    d=json.load(open(p))
    print(p, {k:v for k,v in d.items() if 'top1' in str(k) or 'both' in str(k)})
"

# contrast ペア数の確認（64 組であること）
uv run python -c "
import json,collections
c=collections.Counter()
for line in open('data/d1/test.jsonl'):
    r=json.loads(line)
    if r.get('contrast_pair_id'): c[r['contrast_pair_id']]+=1
print('contrast pairs:', len(c), 'cases:', sum(c.values()), 'sizes:', collections.Counter(c.values()))
"
```

---

## 6. 制約（厳守）

- **コードを変更しない。** このタスクは実行と分析である。既存 CLI で足りないものがあれば**実装せずに報告する**
- **`src/beme/lattice.py` / `src/beme/converter.py` 変更禁止**（探索コア）
- **`both_correct` で採用判定をしない**（issue #19 コメント §5 の明示的な指示）。主判定は `non_contrast_top1` の gain
- §2.4 の引き算を「分解」と書かない。上限/下限の見積りであると注記する
- Phase 4 の測定手続きを勝手に変えない（比較可能性が目的）
- 単一 split・単一 seed の差に p 値ラベルを付けない
- `git commit` / `git push` をしない。作業はこの worktree（と GPU 側の `~/beme-19`）の中だけ。`git stash` を使わない
- 日本語で報告。数値と結論を先頭に
