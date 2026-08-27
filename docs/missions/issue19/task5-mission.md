# task5 — F2-c: 学習実行とアブレーション判定（C0/C1 × 3 seed）

対象 issue: #19（A 項目）
実行機: GPU マシン `ssh ws`（学習は CPU 律速だが、Phase 3/4 と同じ機で回して比較の足場を揃える）
依存: task4
種別: 実装（`run_ablations.py` に条件追加）+ 実行

---

## 0. 目的

F2 の 3 特徴が d1 test の Top-1 誤りを減らすかを、Phase 3/4 と同じ手続きで判定する。

判定は**事前登録**する。McNemar（3 seed ペアリング）で C0 vs A8 に有意な改善があり、かつ test 誤り件数の中央値が下がっていれば採用。

---

## 1. 必ず読むもの

- `gh issue view 19`（コメント §4・§5）
- `docs/missions/issue19/PLAN.md` §2.5（base は A8）・§3.3（特徴定義）・§3.5（分岐）
- `docs/missions/issue19/s-measurement-report.md`（task1。分岐 A の判定 = F2 の理論上限）
- `data/corpus_freq_v1.manifest.json`（task3。被覆率 = F2 が触れるケースの割合）
- `scripts/run_ablations.py` — 全部。`CONDITIONS` / `SEEDS` / 1 run の完全なコマンド列（216-232 行）/ `--summarize-only` の McNemar 集計
- `docs/missions/issue9-phase3/phase3-report.md`・`docs/missions/issue9-phase4.1/phase4.1-report.md`（判定の書き方・事前登録の作法）
- `src/beme/rerank_stats.py`（`mcnemar`）

---

## 2. やること

### 2.1 条件の追加

`scripts/run_ablations.py` の `CONDITIONS` に足す:

```python
"C0": "decomposed_v1-no_script-surface_len+corpus_v1",   # 本命（A8 + corpus 3）
"C1": "decomposed_v1+corpus_v1",                          # A0 + corpus 3
```

既存条件は 1 つも消さない・並べ替えない。`tests/test_run_ablations.py` が通ることを確認する。

### 2.2 実行する run

| 条件 | spec | 比較対象 | seed |
|---|---|---|---|
| **C0** | `...-no_script-surface_len+corpus_v1` | A8 | 9, 17, 33 |
| **C1** | `decomposed_v1+corpus_v1` | A0 | 9, 17, 33 |
| A8 | `decomposed_v1-no_script-surface_len` | — | 9, 17, 33 |
| A0 | `decomposed_v1` | — | 9, 17, 33 |

**A8 / A0 も同じ日・同じ機で再実行する**（issue #17 Phase 4.1 と同じ流儀）。過去の artifact と比べると、機・日付・依存バージョンの差が交絡する。

計 12 run。学習設定は Phase 3/4 と同一（`--epochs 5 --learning-rate 0.1`）。**task2 が「別の lr を採用すべき」と結論していた場合のみ**その設定に変え、変えたことを報告に明記する。

### 2.3 表の供給（リーク経路になるので慎重に）

`data/corpus_freq_v1.bin` は gitignore なので bundle に入らない。**scp で運ぶ**:

```bash
scp data/corpus_freq_v1.bin ws:~/beme-19/data/
scp data/corpus_freq_v1.manifest.json ws:~/beme-19/data/
```

**転送後に GPU 側で SHA-256 がローカルと一致することを確認する**。一致しない表で学習した数値は無効である。

### 2.4 判定（事前登録・事後に動かさない）

| 判定 | 条件 |
|---|---|
| **採用** | C0 vs A8 の McNemar（3 seed ペアリング）で p < 0.05 かつ C0 の test 誤り件数の中央値が A8 より小さい |
| **不採用（効果なし）** | 上記を満たさない |
| **要調査** | C1 vs A0 と C0 vs A8 で符号が逆になる（base 依存が出る）場合 |

不採用でも**それは結果である**。F0（`lex.reading_cond_logprob`）が効かなかったのは「新情報ゼロ」だったからで（issue #19 コメント §3-2）、F2 が効かなかった場合はそれとは別の理由（被覆率が低い / 同点がそもそも主因でない）を、task1 の分岐 A と task3 の分岐 C の実測値で切り分けて報告する。**これが F2 を「効かなくても走らせる」理由である。**

### 2.5 追加で必ず記録するもの

- **学習後の corpus 3 特徴の重み**（3 seed 分）。符号が期待どおりか（`lex.corpus_reading_cond_logprob` は負の対数確率 = コスト向きなので**正の重み**が期待値）
- **`lex.corpus_unseen` の重み**。これが未観測ペアへの学習可能なペナルティである
- C0 と A8 の test 誤りケースの差分（C0 が直したケース / 壊したケース）を数件、実例で
- 4 指標回帰の確認（`run_ablations.py` が回す `evaluate.py --suite all` / `oracle_recall` / `diagnose_cases` / `measure_cost_scale`）

---

## 3. 実行手順

```bash
# ローカル: bundle を作って転送（task2 で clone 済みなら fetch でよい）
git bundle create /tmp/beme-19.bundle feat/issue19-f2
scp /tmp/beme-19.bundle ws:/tmp/
ssh ws 'git clone --branch feat/issue19-f2 /tmp/beme-19.bundle ~/beme-19 && cd ~/beme-19 && uv sync'

# d1 と corpus 表を供給
ssh ws 'cd ~/beme-19 && ln -s ~/beme-d1/repo/data/d1 data/d1'
scp data/corpus_freq_v1.bin data/corpus_freq_v1.manifest.json ws:~/beme-19/data/
ssh ws 'cd ~/beme-19 && sha256sum data/corpus_freq_v1.bin'   # ローカルと一致すること

# 学習（12 run・デタッチ推奨）
ssh ws 'cd ~/beme-19 && nohup setsid uv run python scripts/run_ablations.py \
  --conditions decomposed_v1-no_script-surface_len+corpus_v1,decomposed_v1+corpus_v1,decomposed_v1-no_script-surface_len,decomposed_v1 \
  --seeds 9,17,33 > /tmp/f2.log 2>&1 < /dev/null &'

# 集計
ssh ws 'cd ~/beme-19 && uv run python scripts/run_ablations.py --summarize-only'

# 回収
scp -r ws:~/beme-19/artifacts/issue19-f2 artifacts/
```

`run_ablations.py` の実際の CLI 引数名は**コードを読んで確認する**（上は形の目安）。artifact の出力先が `artifacts/issue9-phase3` 固定になっている場合は、出力先を引数化するか、GPU 側で回収後にディレクトリ名を変える（**既存条件の artifact を上書きしない**こと）。

---

## 4. 完了条件

- [ ] `scripts/run_ablations.py` に C0/C1 が追加され、既存条件が 1 つも消えていない
- [ ] `uv run pytest -q` が **638 + 新規テスト数 passed / 2 failed**
- [ ] GPU 側で表の SHA-256 がローカルと一致していることを確認した
- [ ] 12 run（C0/C1/A8/A0 × seed 9/17/33）が完走し、`report.json` が揃っている
- [ ] `--summarize-only` の McNemar 判定が出ている（C0 vs A8・C1 vs A0）
- [ ] 学習後の corpus 3 特徴の重み（3 seed 分・符号込み）が記録されている
- [ ] §2.4 の判定（採用 / 不採用 / 要調査）が明示されている
- [ ] artifacts がローカルに回収され、SHA-256 が記録されている
- [ ] `git diff --stat origin/main -- src/beme/lattice.py src/beme/converter.py` が **空**
- [ ] git commit / push をしていない

---

## 5. 検証コマンド

```bash
cd /home/sarashin/work/beme-19

uv run pytest -q tests/test_run_ablations.py -v
uv run pytest -q

# 条件一覧に C0/C1 があり既存が残っていること
uv run python -c "
import sys; sys.path.insert(0,'scripts')
from run_ablations import CONDITIONS, SEEDS
for k,v in CONDITIONS.items(): print(k, v)
print('seeds', SEEDS)
"

# 回収した結果の主要数値
uv run python -c "
import json,glob
for p in sorted(glob.glob('artifacts/issue19-f2/*/seed*/report.json')):
    d=json.load(open(p))
    m=d['selected_model']['test']
    w=json.load(open(p.replace('report.json','weights.json')))['weights']
    print(p, m['end_to_end_error_rate'],
          {k:round(v,4) for k,v in w.items() if 'corpus' in k})
"

git diff --stat origin/main -- src/beme/lattice.py src/beme/converter.py
git diff --check
git status --short
```

---

## 6. 制約（厳守）

- **`src/beme/lattice.py` / `src/beme/converter.py` 変更禁止**（探索コア）
- `src/beme/features.py` はこのタスクでは変更しない（実装は task4 で完了しているはず。足りないものがあれば**実装せずに報告する**）
- 既存条件（A0-A8 / M0 / M1 / F0 / F1 / X0）の定義と過去 artifact を書き換えない
- **判定基準（§2.4）を事後に動かさない。** 不採用も結果である
- 表の SHA-256 が GPU 側で一致しなければ**学習を回さない**
- 単一 seed の p 値で採用ラベルを付けない（Phase 3/4 のレポート規約）
- `git commit` / `git push` をしない。作業はこの worktree（と GPU 側の `~/beme-19`）の中だけ。`git stash` を使わない
- `tests/test_audit_leakage.py` の既存 2 件の失敗は**直さない**（環境固有。PLAN §2.7）
- 学習が長引くならデタッチ起動（`nohup setsid ... < /dev/null &`・PPID=1 確認）で回し、待つ間に別の完了条件を進める
- 日本語で報告。数値と結論を先頭に
