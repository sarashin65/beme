# P2/#1 評価性能問題 — 原因特定レポート

対象: `feat/p2-kbest-and-exact` (worktree `~/work/beme-p2`) / 調査のみ・実装なし
再現スクリプト: `investigation/probe1.py` … `probe6.py`

## 結論（1行）

**`backward_viterbi()` のヒューリスティックが「許容的だが極端に緩い」ため、A\* が実質 Dijkstra 化して爆発している。**
原因は `continuation` を **POS だけをキーにした dict** で持っていること — 同一 POS でも**長さの違うエッジ**の継続コストが 1 スロットに `min` で潰され、短いエッジが長いエッジの「終端に近い」継続コストを借用してしまう。

---

## 1. 測定事実

### 1-1. 爆発の実測（probe1.py）

| case | 長さ | エッジ数 | K | pops | pushes | heap最大 | 時間 |
|---|---|---|---|---|---|---|---|
| 0 | 45 | 1391 | 5 | 42,284 | 1,122,860 | 1,080,577 | **20s でも未完了** |
| 2 | 55 | 1379 | 64 | 39,362 | 1,040,958 | 1,001,605 | 18.1s |
| 3 | 34 | 710 | 64 | 678 | 13,502 | 12,826 | 0.12s |

`pushes / pops ≈ 26` = 1 位置あたりの平均辞書エッジ数。1 回 pop するたびに 26 個 push されるので、
pop 数がわずかに増えるだけでヒープが百万件規模になる。

### 1-2. ヒューリスティックの誤差（probe2.py）

`backward_viterbi()` の値 `h[p]` と、POS を考慮した**真の残余コスト** `R[p][prev_pos]` の比較:

| case | 長さ | `h[0]` | 真値 | **下界の緩み** |
|---|---|---|---|---|
| 0 | 45 | 21.378 | 28.128 | **6.75** |
| 2 | 55 | 47.424 | 51.562 | **4.14** |
| 3 | 34 | 22.219 | 24.382 | **2.16** |
| 1 | 10 | 4.300 | 6.410 | **2.11** |

docstring は「遷移コストを含む」と書いているが、実際には **文字数に比例して約 0.15/文字ずつ下振れ**する。
A\* は `f ≤ f_K` の状態を全部展開するため、この緩みがそのまま展開数の指数的増加になる。

### 1-3. 決定的証拠 — ヒューリスティックだけ差し替えた比較（probe3.py）

`R[pos][prev_pos]`（POS 別の厳密後向き DP）に差し替えた結果。**出力は完全に同一**（top surface・コストとも一致）:

| case | 現行 (loose) | 厳密 (exact) | 改善 |
|---|---|---|---|
| 0 (len45) | 25s でも未完了 / heap 1.46M | **0.023s** / pops 353 / heap 7,827 | ∞ |
| 2 (len55) | 18.07s / pops 39,362 | **0.014s** / pops 381 | **約1,300倍** |
| 3 (len34) | 0.123s / pops 678 | 0.058s / pops 505 | 2倍 |

→ **これが原因。他は全部これの二次被害。**

### 1-4. 最小再現（probe4.py）— バグの機序

```
node1 から出る 2 本の "noun" エッジ（長さが違う）:
  A: cost 10, 1->4 (終端)      → after_edge(A) = 0.0
  b: cost  1, 1->2 → c(9) → d(9) → after_edge(b) = 18.16

continuation[1]["noun"] = min(0.0, 18.16) = 0.0   ← 長さの違いを潰す
```
前任ノードが `h` を計算するとき:
```
A -> 10.0 + 0.08 + continuation[1]["noun"](=0.0) = 10.08   (正しい)
b ->  1.0 + 0.08 + continuation[1]["noun"](=0.0) =  1.08   ← b が A の継続を借用
min = 1.08
```
実測: `h[0] = 1.08`、真値 `10.08` → **9.0 の過小評価**。

該当箇所は `src/beme/lattice.py:291-294`:
```python
continuation[node.position][edge.token.pos] = min(
    continuation[node.position].get(edge.token.pos, math.inf),
    after_edge,
)
```
`after_edge` は「そのエッジが**どこで終わるか**」に強く依存するのに、キーが `pos` だけ。
辞書は 1 位置から長さ 1〜8 の同 POS エッジを大量に出す（平均 26 本）ので、実データでは常に最長エッジの継続コストが全員に配られる。

---

## 2. 依頼された 4 つの仮説への回答

| # | 仮説 | 判定 |
|---|---|---|
| 1 | A\* が指数的に状態展開 | **YES（結果）**。pops 4万・pushes 110万・heap 146万。ただし原因は #2。パス重複展開の除去（dominance/dedup）は無いが、h が厳密なら pops は K×L 程度に収まるので不要（実測 353 pops）。 |
| 2 | backward_viterbi が許容的でない | **半分 YES**。許容性(admissible)は**保たれている**（常に下界・結果は正しい）。破れているのは**下界の緊密さ**。原因は遷移コストの扱いではなく、**同 POS・異長エッジの continuation 混同**。 |
| 3 | `_tokens_cache` でメモリ爆発 | **YES（増幅要因）**。probe5: ヒープ上のパスの **100%** が `_tokens_cache` 実体化済み（平均 11.9 トークン）。M1 の遅延復元は完全に無効化されている。**1,471 bytes/エントリ** → 146万件で **約 2.1GB**（+ インタプリタ overhead で実測 3.8GB RSS と整合）。 |
| 4 | M4「非かな＝固定エッジ」でラティス巨大化 | **NO**。probe5: case2 は総エッジ 1,379 のうち literal は **わずか 2 本**。M4 が悪いのではなく、M4 が**探索対象を「短いスパン」から「文全体」に伸ばした**ことで、1-2 の緩さが初めて顕在化した（長さに比例して緩みが累積するため）。M4 の設計自体は正しい。 |

### 3 の機序（`k_best_viterbi` 内、lattice.py:370-372）
```python
child = LatticePath(cost, edge.token, path)
tokens_key = tuple((t.reading, t.surface, t.pos, t.cost) for t in child.tokens)
```
`child.tokens` を **push のたびに** 呼ぶ → `tokens` プロパティが親を全部辿って `_tokens_cache` に書き込む。
結果、(a) push あたり O(L) の時間、(b) 全パスが O(L) のタプルを保持。M1 の永続リンクリスト化の効果がゼロになる。
さらに `child.surface` も同じくヒープキーに入れており、これも O(L) の文字列連結。

---

## 3. 修正方針（優先順位順）

### P0 — `backward_viterbi` を「(位置, 直前POS)」の厳密後向き DP にする ★これだけで解決
`continuation` を捨て、`R[pos][prev_pos]` を返す。

```python
def backward_viterbi_exact(lattice):
    n = len(lattice.nodes); term = n - 1
    poss = {None} | {e.token.pos for node in lattice.nodes for e in node.edges}
    R = [dict() for _ in range(n)]
    for p in poss:
        R[term][p] = 0.0
    for pos in range(n - 2, -1, -1):
        edges = lattice.nodes[pos].edges
        for p in poss:
            best = math.inf
            for e in edges:
                nxt = R[e.end].get(e.token.pos, math.inf)
                if nxt == math.inf:
                    continue
                c = pos_transition_cost(p, e.token.pos) + e.token.cost + nxt
                if c < best:
                    best = c
            R[pos][p] = best
    return R
```
- コスト: `O(nodes × |POS| × edges_per_node)`。POS は辞書全体で **17 種**、実測で 1 文あたり数万演算 = 無視できる。
- `k_best_viterbi` 側: `remaining[edge.end]` → `R[edge.end].get(edge.token.pos, inf)`、ルートは `R[0][None]`。
- **互換性の注意**: 公開 API の `Lattice.backward_costs()` / `backward_viterbi()` / `backward_costs` / `viterbi_backward` は `tuple[float, ...]` を返す契約でテストもある。**既存関数はそのまま残し**、A\* 用に新関数を足すのが安全。既存の `tuple` 版は `min(R[pos].values())` で導出できる。

### P1 — ヒープキーから `surface` と `tokens_key` を外す
```python
heapq.heappush(queue, (cost + h, cost, next(sequence), edge.end, child))
```
- 効果（probe3, case2）: 0.105s → **0.014s**（同一結果）。P0 適用後もなお **3〜7倍**。
- メモリも 1,471 B/エントリ → 数百 B に。
- **決定性は壊れない**: `build_lattice` が `_edge_key`（end, cost, surface, pos, reading）でエッジをソート済みなので展開順は辞書挿入順に依存しない。かつ `Converter.convert` は最終的に `(cost, surface)` で並べ直している（converter.py:152）。
- `tests/test_converter.py:59` の決定性テストは「2回実行が一致」＋「(cost,surface)順」だけを見るので影響なし。

### P2 — `LatticePath._tokens_cache` を捨てるか、`tokens` を非キャッシュ化
P1 を入れれば探索中に `tokens` は呼ばれなくなるので `_tokens_cache` は害を及ぼさなくなるが、
`slots` に 1 フィールド余計に持つだけになる。M1 の意図（遅延復元）を活かすなら、
`tokens` は毎回計算する純プロパティにして `_tokens_cache` を削除するのが素直（呼ばれるのは最終 K 件だけ）。

### P3 — `Converter.convert` が `k` ではなく `self.beam_width`(=64) で探索している
`converter.py:142` は常に 64 件取る。P0 適用後は安価（K=5→64 で pops 353→381 程度）なので急がないが、
`max(k, ...)` 程度に絞ると素直。**重複 surface を潰すため k 件ちょうどでは足りない**点に注意（現状 64 は妥当なマージン）。

### P4 — 保険: 探索上限
`pops`/`pushes` に上限（例: `200 * beam_width * len(nodes)`）を設け、超えたら `remaining` を打ち切って
その時点の結果を返す。P0 のリグレッションで再び OOM しないためのガード。テストで上限到達時の挙動を固定する。

### 参考: `_paths_by_node`（lattice.py:312）
`Lattice.paths_at()` 専用に旧ビーム DP が残っており、`k_best_viterbi` と別アルゴリズムになっている。
今回の性能問題とは無関係（評価経路で呼ばれない）だが、意味論の二重管理として要整理。

---

## 4. 修正後の実測（probe6.py — P0 + P1 を当てた等価実装で全件評価）

| suite | 件数 | 修正前 | 修正後 | 指標 |
|---|---|---|---|---|
| ajimee | 200 | **300s+ 未完了** | **2.75s** | mean_cer **0.2901** / exact 0.0700 |
| mozc | 498 | **OOM kill (3.8GB)** | **1.75s** | pass_rate 0.1767 |

`mean_cer = 0.2901` は直前コミット `ba66606`（P0 カタカナ表層抑制）の **CER 0.290 と一致**。
つまり M4 の文全体ラティス化は品質を落としておらず、**性能問題だけが単独の原因**だった。

---

## 5. 実装順序の推奨

1. **P0** を入れて `pytest` 56件が通ることを確認 → `--suite ajimee` が数秒で完了するはず
2. **P1** を入れて再度全テスト + ajimee/mozc
3. **P4** のガードとテストを追加
4. P2/P3 は余力で
