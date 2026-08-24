# Issue #9 Phase 4（文脈モデルへの接続）実行プラン

作成日: 2026-08-24
対象: `feat/issue9-phase4` @ `8ee6cab`（`origin/main`・Phase 3 マージ済み）
状態: **プランのみ。実装未着手。**
正本: GitHub issue #9 本文の Phase 4 チェックリスト、[Phase 4 の進め方コメント](https://github.com/sarashin65/beme/issues/9#issuecomment-5386363007)、`docs/missions/issue9-phase3/phase3-report.md` §10 Limitations and Phase 4 Handoff

---

## 0. 最初の worker への指示文の案

> Issue #9 Phase 4 のタスク 1 を実施してください。
>
> リポジトリ: `/home/sarashin/work/beme-9p4`（ブランチ `feat/issue9-phase4`・クリーン）
>
> まず次を順に読んでください。
> 1. `docs/missions/issue9-phase4/PLAN.md` の §1〜§2.1 と §3.1（Phase 4 の前提と、なぜ Top-1 の定義統一が最優先なのか）
> 2. `docs/missions/issue9-phase4/task1-mission.md`（あなたの指示書。目的・対象ファイル・変更内容・完了条件・検証コマンド・制約がすべて書いてあります）
> 3. `src/beme/training.py`（`evaluate_split`）・`src/beme/lattice.py`（`k_best_viterbi` と `_paths_by_node` の `_path_key`）・`src/beme/converter.py`（`convert` の dedup と `(cost, surface)` ソート）
>
> タスク 1 は「Top-1 の入口の定義を 1 つに統一する」実装です。Phase 4 は全部が「リランカ前後の Top-1 差」の測定なので、入口が 2 種類あるまま先に進むと Phase 3 と同じ罠（A3 で 208 件の乖離）を踏みます。
>
> **`k_best_viterbi` のヒープキーは変えないでください。** ヒープは A\* の f 値 `(cost + after_edge, cost, sequence, ...)` で pop しており、`sequence` は単なる挿入順カウンタです。表層は終端に着くまで確定しないので、探索中に表層でタイブレークすることは原理的にできません。統一は「同点の最小コスト経路群を列挙してから `(cost, surface)` の最小を取る」共有ヘルパで行ってください。設計は PLAN §3.1 にあります。
>
> 落とし穴を 2 つ先に伝えます。
> - `evaluate_split` は学習ループの内側で train 10,000 + dev 1,000 に対して毎 epoch 走ります。`beam_width=1` を `64` にすると学習時間が跳ねます。同点が無い限り 1 回目の pop で確定する early-stop を必ず入れて、PLAN §3.1 の性能ゲート（1 epoch の wall time が現行比 ×1.2 以内）を実測で示してください。
> - `Converter.convert` は dedup 後に `(cost, surface)` でソートします（`converter.py:169`）。`_paths_by_node` の `_path_key` は `(cost, surface, tokens)` で既に表層を見ていますが、`k_best_viterbi` は見ていません。**直すのは `k_best_viterbi` を使う側であって `_path_key` ではありません。**
>
> 既存の 12 条件の数値のうち、同点が発生しない条件（A0 を含む 11 条件）は 1 bit も変わらないはずです。`uv run pytest -q` は現在 585 passed です。実装完了後、指示書の検証コマンドを実行し、実際の出力を報告してください。`git commit` / `git push` はしないでください。

---

## 1. 結論

Phase 4 は次の **7 タスク**に分ける。

| # | タスク | 種別 | 依存 |
|---:|---|---|---|
| 1 | Top-1 定義の統一（`evaluate_split` と converter） | 実装 | なし |
| 2 | リランカ評価ハーネス（score / combine の 2 段分離） | 実装 | 1 |
| 3 | 集計・統計判定モジュールと回帰テスト | 実装 | なし（2 と並行可） |
| 4 | リーク監査の完走（GPU 側・d1 JSONL あり） | 実行 | なし |
| 5 | 測定実行: 3 条件 × d1 dev/test（GPU マシン） | 実行 | 1–4 |
| 6 | cascade ゲート配線とレイテンシ予算（**5 の結果次第**） | 実装 | 5 |
| 7 | Phase 4 レポート | 文書 | 5（6 が実施されたならそれも） |

**Phase 4 の設計上の中核は「タスク 2 の 2 段分離」である。** cross-encoder の relevance は `(context, surface)` だけの関数であり、`viterbi_weight` にも `rerank_k` にも依存しない（`reranker.py:293`）。したがって **relevance を 1 回だけ計算してキャッシュすれば、`viterbi_weight` × `rerank_k` × ゲート閾値の全グリッドが追加の推論ゼロで厳密に評価できる。** これをやらないと、1 グリッド点あたり d1 test 1 周（miniPC 実測で約 1.8 時間）が必要になり、Phase 3 と同じ「条件を増やしてから足りないと気づく」失敗を繰り返す。

もう 1 つの中核は **primary metric を contrast pair 128 件にする**ことである（§3.3）。全体 2,000 件の Top-1 は局所モデルが既に `.731` まで上げており、リランカの効果に対して鈍感である。

---

## 2. 調査で確定した前提

以下はすべて本ワークツリー・`beme-gpu` で実際に測った値である。**推測ではない。**

### 2.1 Top-1 の入口が 2 種類あることの機序（コード実測）

Phase 3 レポート §10.3 が「A3 で `end_to_end_errors` 588 と diagnose の誤り 796 が 208 件乖離した」原因として記録した現象を、コードで特定した。

| 経路 | タイブレーク | 該当行 |
|---|---|---|
| `evaluate_split` → `k_best_viterbi(beam_width=1)` | **ヒープの挿入順**（`sequence` カウンタ） | `training.py:100` / `lattice.py:466-469,494-503` |
| `Converter.convert` → dedup 後ソート | **`(cost, surface)`**（表層の辞書式順） | `converter.py:169` |

`k_best_viterbi` のヒープキーは `(cost + after_edge, cost, next(sequence), position, path)` で、第 3 要素は「結果を辞書の挿入順から独立させるためだけ」のカウンタである（`lattice.py:464-465` のコメント）。**表層は終端ノードに着くまで確定しないので、探索中に表層でタイブレークすることは原理的にできない。**

一方、`_paths_by_node` の `_path_key`（`lattice.py:413-418`）は `(cost, surface, tokens)` で既に表層を見ている。**つまり「表層でタイブレークする」規約はリポジトリ内に既に 1 つ存在しており、`k_best_viterbi` だけがそれに従っていない。**

**帰結**: 統一先は `(cost, surface)` の最小、つまり converter 側の規約である。§3.1 で実装方針を決める。

### 2.2 リランカ配線の実測

| 事実 | 根拠 |
|---|---|
| リランクの発火条件は `not context.is_empty and self.reranker is not None` のみ | `converter.py:176` |
| `needs_rerank`（`confidence < threshold`）は計算されるだけで**何もゲートしていない** | `converter.py:174`。参照は `evaluation.py:250` の集計のみ。Phase 3 が `ast` 検査で確認済み |
| `rerank_k` を指定すると head だけリランクし、`base_candidates[rerank_k:]` をそのまま後ろに連結する | `converter.py:176-192`。**tail の候補が上位へ来ることはない** |
| `viterbi_weight * cost` を使うのは `CrossEncoderReranker` と `LLMLogitsReranker` だけ。`MockContextReranker` は候補コストを混ぜない | `reranker.py:293` / `:433` / `:168` |
| cross-encoder のスコアは `viterbi_weight * cost − cross_encoder_weight * relevance`。**relevance は `(context, surface)` だけの関数** | `reranker.py:275-301`。query は `f"{preceding}【変換対象】{following}"` |
| `oracle_recall.py` は `converter.reranker is not None` を例外にする | `oracle_recall.py:122-123` |
| d1 で context を持つ件数 | test 1,982/2,000・dev 996/1,000・train 9,937/10,000。**空 context の 18 件はリランクが走らない** |

### 2.3 cross-encoder のレイテンシとスループット（本ワークツリーで実測）

`hotchpotch/japanese-reranker-base-v2`・`local_files_only=True`・CPU（Intel N100・4 core）。48 候補 1 セットを 3 回計測した中央値。

| batch_size | 1 呼び出し | 1 候補あたり |
|---:|---:|---:|
| 8 | 3,453 ms | **71.9 ms** |
| 16 | 3,307 ms | 68.9 ms |
| 48 | 3,518 ms | 73.3 ms |
| 64 | 3,522 ms | 73.4 ms |

**バッチサイズを上げても速くならない。** 4 コアが飽和しており、並列度の余地が無い。モデルのロードに 18.8 秒かかる（プロセスあたり 1 回）。

40 例の実走行（後述のパイロット）では 2,052 ペアを 130 秒 = **63 ms/ペア**だった。

**このレイテンシは、そのままでは IME として成立しない。** 1 変換あたりユニーク候補 47.9（Phase 3 実測）× 63 ms = **約 3.0 秒**。README の 48.7 ms/候補（AJIMEE・辞書拡充前・`top_k=5`）より遅いのは、d1 の context が長く（左 40 字前後）query が長いためである。

**帰結**: `rerank_k` による間引きと cascade ゲートは「あとで考える最適化」ではなく、**Phase 4 が実用性を主張できるかどうかの分かれ目**である（issue コメントの指摘どおり）。

### 2.4 パイロット測定: contrast pair 40 件で cross-encoder は実際に効いた

**これが Phase 4 の計画を決める最大の実測である。**

条件: d1 test の contrast pair 128 件の先頭 40 件。局所モデルは A0 seed 9 の学習済み重み（`beme-gpu:~/beme-p3/artifacts/issue9-phase3/A0/seed9/weights.json`・sha256 `70a6ffed…`）、辞書 `data/dictionary_v2.tsv`、`beam_width=64`、候補は `Converter.convert(k=64)`。cross-encoder は `japanese-reranker-base-v2`。relevance を 1 回計算してキャッシュし、`viterbi_weight` と `rerank_k` をオフラインでスイープした。

候補集合の性質: gold は **40/40 すべてが候補集合内**（recall = 1.0）。gold の順位は p50 = 2・p90 = 6。

| 条件 | `viterbi_weight` | `rerank_k` | Top-1 |
|---|---:|---:|---:|
| **L: 局所モデル単独** | — | — | **16/40 = .400** |
| **R: リランカ単独**（CE の relevance のみ） | 0.0 | 64 | **16/40 = .400** |
| R（head 10 件だけ CE 順） | 0.0 | 10 | 22/40 = .550 |
| **LR: 併用** | 1.0 | 10 | **24/40 = .600** |
| LR | 1.0 | 64 | 23/40 = .575 |
| LR | 3.94（Phase 3 の初期倍率） | 10 | 23/40 = .575 |
| LR | 0.5 | 64 | 23/40 = .575 |
| LR | 8.0 | 64 | 22/40 = .550 |

McNemar（`vw=1.0`・`rerank_k=10` vs 局所単独・同一 40 例のペアあり）: **b（失った）= 6 / c（得た）= 14 / discordant = 20 / 両側正確二項検定 p = 0.115**。

**この 40 件から言えること・言えないこと**を分けて書く。

言えること:
1. **cross-encoder の relevance は contrast pair 上で確かに情報を持っている。** 局所単独 16 に対し併用は 24 で、+8（net）は「差が 0.002 以下」だった AJIMEE CER とは桁が違う。
2. **リランカ単独（`vw=0`・全候補）は局所単独と同じ .400 で、併用に負ける。** cross-encoder の relevance だけで 1-best を決めるのは成立しない。**局所コストとの結合が効果の本体である。**
3. **`rerank_k` は効く。** `vw=0` では 10 と 64 で `.550` → `.400` と大きく落ちる。深い候補（表層としてありえない literal 断片）に CE が高い relevance を付け、局所コストの守りが無いと上へ出てくる。
4. **40 件では有意にならない**（p = 0.115）。128 件で同じ効果量なら discordant ≈ 64・b:c ≈ 19:45 となり p は 0.01 を大きく下回るので、**primary を contrast pair 128 件に置く設計は標本サイズとして妥当**である。

言えないこと（レポートで主張してはならない）:
- **`vw=1.0` / `rerank_k=10` が最適である**とは言えない。上の表は同じ 40 件でグリッドを回して最良セルを見ているので選択バイアスがある。**ハイパーパラメータの選択は d1 dev で行い、test は 1 回だけ評価する**（§3.4）。
- **全体 2,000 件で改善する**とは言えない。contrast pair は局所モデルが最も弱い層であり、他の層では悪化しうる。実際 b = 6 件は併用で**失った**ケースである。
- **AJIMEE / Mozc / handmade で改善する**とは言えない。Mozc は context を持たずリランカが走らない。

### 2.5 オフライン再現式が `Converter.convert` と厳密一致することの検証（実測）

タスク 2 の 2 段分離は「`combine` が `Converter.convert` の順位付けを厳密に再現できる」ことに全面的に依存する。**計画立案の段階で、この仮定を実際に検証した。**

方法: `CrossEncoderReranker` の `model` だけを決定的な test double に差し替え（**リランカのコードは本物**）、`Converter(reranker=ce, rerank_k=...)` の `convert` 出力と、リランカ無し converter の候補集合 + 同じ relevance に §3.2 のオフライン式を適用した結果を、全順位で突き合わせた。

条件: 読み 6 種 × context 4 種（空 context を除く 3 種）× `(viterbi_weight, cross_encoder_weight)` 4 組 × `rerank_k ∈ {None, 1, 3, 10, 1000}` = **480 通り**。

結果: **不一致 0。** `rerank_k` が `None` / 候補数未満 / 候補数超過 のいずれでも一致した。

**帰結**: 2 段分離は近似ではなく厳密である。ただしこれは計画立案時の scratchpad 検証であり、**タスク 2 はこの突き合わせをリポジトリ内の恒久的なテストとして持つこと**（task2-mission の必須テスト 1）。式を後から触ったときに壊れたことが分かるのはテストだけである。

### 2.6 confidence と contrast pair の関係（実測）

パイロット 40 件の `confidence = candidates[1].cost − candidates[0].cost` は **40 件すべてが 0.5 未満**だった。Phase 3 の A0 seed 9 実測では d1 test 全体の learned confidence が p25 = `.0344` / p50 = `.1471` / p75 = `2.4535` なので、「conf < 0.5」は全体のおよそ 6 割強を占める。

**したがって「confidence が低い」は contrast pair の必要条件であって十分条件ではない。** cascade ゲートを confidence だけで組むと発火率が 6 割を超え、レイテンシ削減にならない。ゲート設計はタスク 6 で、**測定された「リランカが効いたケース」に対する precision / recall として評価する**（§3.6）。

### 2.7 d1 の層別件数（`beme-gpu` で実測）

| split | 総数 | contrast pair | pair id（対の本数） | context あり |
|---|---:|---:|---:|---:|
| train | 10,000 | 872 | 436 | 9,937 |
| dev | 1,000 | **44** | 22 | 996 |
| test | 2,000 | **128** | 64 | 1,982 |

**dev の contrast pair は 44 件しかない。** これが Phase 4 の実験計画上の最大の制約である。対応は §3.4。

### 2.8 実行環境の実測

| 資産 | miniPC（`/home/sarashin/work/beme-9p4`） | GPU マシン（`beme-gpu`） |
|---|---|---|
| CPU / RAM | 4 core（Intel N100）/ 7 GB | 24 core / 31 GB |
| d1 の `*.jsonl` | **無い**（`.gitignore:8`） | `~/beme-d1/repo/data/d1/`（test 3.6 MB / dev 1.8 MB / train 18 MB） |
| Phase 3 の学習済み重み 36 本 | 無い | `~/beme-p3/artifacts/issue9-phase3/{A0..X0}/seed{9,17,33}/weights.json` |
| cross-encoder キャッシュ | `~/.cache/beme/cross-encoder/`（2.7 GB・2 モデルとも） | **無い** |
| LLM gguf `Qwen2.5-0.5B` | `~/.cache/beme/models/`（380 MB） | **無い** |
| ネットワーク | あり | **あり**（huggingface.co へ到達確認済み） |
| torch | CPU | CPU（`2.13.0+cpu`）。GPU は RTX 3060 Ti 8 GB だが CUDA 版 torch は未導入 |
| `~/beme-p3` の git | — | `194131d` のクローン + 未コミット変更あり。**Phase 4 では `~/beme-p4` に本ブランチを新規クローンする** |

CE スコアリングの所要見積り（63 ms/ペア × コア比で外挿）:

| 実行機 | d1 dev 1,000 件（約 48k ペア） | d1 test 2,000 件（約 96k ペア） | 合計 |
|---|---:|---:|---:|
| miniPC（4 core・実測ベース） | 約 50 分 | 約 1.7 時間 | **約 2.5 時間** |
| beme-gpu（24 core・外挿） | 約 10–15 分 | 約 20–30 分 | **約 30–45 分** |

**1 回スコアリングすれば全グリッドが無料になる**（§1）ので、この所要は Phase 4 を通じて 1 回だけ払えばよい。

---

## 3. 設計

### 3.1 Top-1 定義の統一（タスク 1）

**統一先は converter の規約 = 終端経路のうち `(cost, surface)` が最小のもの。** 理由は §2.1 のとおり、`_paths_by_node` が既にこの規約を持っており、成果物（`diagnose_cases.py`・`oracle_recall.py`・`evaluate.py`）がすべて converter を通るためである。

3 案を検討した。

| 案 | 内容 | 判定 |
|---|---|---|
| A: `k_best_viterbi` のヒープキーに表層を足す | ヒープの pop 順を変える | **却下**。表層は終端に着くまで確定しない。部分経路の表層で並べると A\* の最適性が壊れる |
| B: `evaluate_split` を `Converter.convert(k=1)` に置き換える | 呼び出しを差し替えるだけ | **却下**。`convert` は内部で `k_best_viterbi(beam_width=64)` を回す。学習ループの内側で train 10,000 × 毎 epoch なので現実的でない |
| C: 共有ヘルパ `top1_surface(lattice, scorer)` を置き、最小コスト同点群だけを列挙して `(cost, surface)` 最小を返す | 両方の呼び出し元をこれに向ける | **採用** |

案 C の実装形:

```text
paths = k_best_viterbi(lattice, beam_width=tie_probe, scorer=scorer)
if not paths: return None
best = paths[0].cost
tied = [p for p in paths if p.cost == best]      # 通常は 1 本
return min((p.surface for p in tied))
```

- `tie_probe` の既定は小さい値（例 8）で始め、`len(tied) == len(paths)`（= 全部同点で打ち切られた可能性がある）のときだけ拡張して再取得する。**同点が発生しない通常モデルでは 1 回目で確定する**ので、`beam_width=1` に対する追加コストはほぼゼロになる。
- 比較は `==` の厳密一致でよい。converter 側のタプルソートも float の厳密比較であり、**同じ規約を再現することが目的**なので許容誤差を入れてはならない。入れると converter と再び乖離する。
- `Converter.convert` 側は既にこの規約なので**変更しない**。ヘルパが converter と同じ答えを返すことをテストで固定する。

**性能ゲート**: `scripts/train_scorer.py` の 1 epoch の wall time が現行比 **×1.2 以内**。超えたら `tie_probe` の初期値と early-stop を見直す。実測を報告に含めること。

**不変条件**: Phase 3 の 12 条件のうち同点が発生しない 11 条件（A0 を含む）は、`evaluate_split` の値が 1 件も変わらないはずである。**A3 だけが変わる**（588 → diagnose の 796 側へ寄る）。これが変わらなかったら統一が効いていない。

### 3.2 リランカ評価ハーネスの 2 段分離（タスク 2）

新規 `scripts/rerank_eval.py` を **`score` と `combine` の 2 サブコマンド**に分ける。

**`score` フェーズ**（重い・1 回だけ）

入力: d1 split JSONL・辞書・局所モデルの重み・リランカ種別。
処理: 各例で `Converter.convert(reading, k=beam_width, allow_literals=True)` を呼び、候補 surface・局所コスト・`ConversionContext` を作り、リランカの生スコアを取る。
出力: JSONL（1 例 1 行）。

```json
{"id": "...", "reading": "...", "gold": "...", "contrast_pair_id": "...",
 "context_empty": false,
 "candidates": [{"surface": "...", "cost": 2.25, "relevance": 7.709}, ...],
 "elapsed_seconds": 3.02}
```

- **`relevance` は `viterbi_weight=0.0` / `cross_encoder_weight=1.0` でリランカを呼び、返る `score` を符号反転して得る**（`score = 0*cost − 1*relevance = −relevance`）。`CrossEncoderReranker` を改造してはならない。既存の重みパラメータで純粋な relevance が取れることを利用する。
- `MockContextReranker` は `score = cost − bonus` なので `relevance = cost − score`（= keyword bonus）として同じ形に落とす。`LLMLogitsReranker` は `viterbi_weight=0`・`llm_weight=1` で mean log-likelihood が取れる。**3 種すべてが同じ JSONL 形式に落ちること**が条件。
- `context_empty` の例（test 18 件）は relevance を `null` にして記録し、combine 側で「リランクしない」として扱う。**黙って 0 を入れてはならない。**
- レイテンシは例ごとに記録する。集計時に候補あたりへ割る。
- 決定性: 同じ入力・同じ引数の 2 回実行で byte-identical（浮動小数の repr まで）。CE の推論はスレッド数で結果が揺れうるので、**`torch.set_num_threads` を固定し、揺れる場合はその事実を記録する**（無理に丸めない）。

**`combine` フェーズ**（軽い・何度でも）

入力: `score` の JSONL 群・グリッド指定。
処理: 各 `(viterbi_weight, rerank_k, gate)` について `Converter.convert` の順位付け規則を**厳密に再現**して Top-1 を出す。

```text
head = candidates[:rerank_k]
head = sorted(head, key=lambda c: (vw*c.cost - ce_w*c.relevance, c.cost, c.surface))
order = head + candidates[rerank_k:]
top1 = order[0]
```

**このソートキーは `reranker.py:301` と `converter.py:176-192` の合成である。ハーネス側で書き直すことになるので、`combine` の出力が実際の `Converter.convert(reranker=...)` の出力と一致することを、少数の例で必ず突き合わせるテストを入れること。** ここが乖離すると Phase 4 の全数値が意味を失う。

出力: §3.3 の指標を含む JSON。

**なぜ 2 段に分けるか**（レポートにも書く）: relevance は `(context, surface)` だけの関数で `viterbi_weight` にも `rerank_k` にも依存しない（`reranker.py:293`）。したがってこの分離は**近似ではなく厳密**であり、グリッド探索の推論コストを 1/(グリッド点数) にする。Phase 3 の「12 条件作ってから足りないと気づいた」失敗を構造で防ぐ。

### 3.3 指標の定義: oracle recall と Top-1 改善を分ける（チェックリスト項目 2）

issue のチェックリスト「oracle recall と Top-1 改善を分けて評価」を、**リランキングが recall に対して持つ不変性**として形式化する。

> **リランキングは候補集合を変えない。順序だけを変える。**
> したがって `surface_recall@k` は **k ≥ min(rerank_k, 候補数) のすべての k で不変**である。変わるのは `surface_recall@1`（= Top-1）と、`k < rerank_k` の recall だけである。

ハーネスはこれを**実測してアサートする**（設計の健全性チェック。破れたら候補集合の取り違えがある）。

出力する表は次の 3 段。

| 段 | 指標 | Phase 3 の実測値（A0 learned・d1 test） | Phase 4 で測るもの |
|---|---|---|---|
| 上限 | `surface_recall@64`・`lattice_coverage` | `1.000` / `1.0` | **不変であることを確認** |
| 伸びしろ | `surface_recall@rerank_k − surface_recall@1` | `rerank_k=10` なら `.988 − .731` = **25.7 pt** | rerank_k ごとに提示 |
| 実績 | リランク後 Top-1 − リランク前 Top-1 | — | **これが Phase 4 の答え** |

**primary metric は contrast pair 128 件の Top-1**（基準 `.3281`）。secondary は全体 2,000 件の Top-1（基準 `.731`）。§2.4 の理由により、全体指標だけを見ると効果を見落とす。

追加で必ず出すもの:

- **gold_rank 分布**（リランク前/後）。「1 位に来ないだけでなく上位にも来ない」のか「惜しいところまで来ている」のかを分ける。issue コメントの切り分け 2 に対応。
- **候補あたりレイテンシ**と **1 変換あたりレイテンシ**。実行機を明記する（レイテンシは実験の性質ではなく実行機の性質）。
- **失ったケース（b）の一覧**。パイロットで b = 6 が出ている。リランカは必ず何かを壊すので、壊したものを列挙しないレポートは受け取らない。
- **contrast pair の対（pair id）単位の集計**。64 対のうち両方正解・片方正解・両方誤りが何対か。同音異義対の解消は本来「対として正しく解けたか」で見るべきである。

### 3.4 3 条件の比較設計（チェックリスト項目 3）と選択の規律

**「リランカ単独」の定義**。IME は必ず候補生成を要するので「候補生成なし」は非現実的である。したがって次のように定義する。

| 条件 | 定義 | パラメータ |
|---|---|---|
| **L: 局所モデル単独** | Phase 3 の A0 learned をそのまま。リランカを付けない | — |
| **R: リランカ単独** | 候補生成は L と同一（同じラティス・同じ K-best）。**順位付けは relevance のみ**で決める | `viterbi_weight = 0` / `rerank_k = beam_width` |
| **LR: 併用** | 局所コストと relevance を線形結合 | `viterbi_weight`・`rerank_k` を dev で選択 |

R は「局所モデルのコストを一切使わない上限」であり、「候補集合という形でしか局所モデルに依存しない」条件である。この定義をレポートに明記する。

**選択の規律**（Phase 3 の教訓を最初から織り込む）:

1. **`viterbi_weight` と `rerank_k` は d1 dev で選ぶ。d1 test は選択後に 1 回だけ評価する。** グリッドは `viterbi_weight ∈ {0, 0.25, 0.5, 1, 2, 3.94, 8}` × `rerank_k ∈ {5, 10, 20, 64}` を既定とする。`3.94` は Phase 3 の cost-scale ratio `.2541` の逆数。
2. **選択の主指標は dev 全体 1,000 件の Top-1、副指標は dev contrast 44 件の Top-1。** dev の contrast は 44 件しかないので（§2.7）、**contrast だけで選ばない**。同点は `(dev 全体 Top-1, dev contrast Top-1, rerank_k 小さい方, viterbi_weight 大きい方)` の順で決め、規則を先に固定してから実行する。
3. **選ばれた点が高原（plateau）の上にあることを確認する。** グリッド全面を必ずレポートに載せ、選択点が単独のスパイクなら「選択できなかった」と書く。パイロットの `.600 / .575 / .575` は近傍が平坦で、これは良い兆候である。
4. **train の contrast 872 件は選択に使わない。** 局所モデルは train で学習済みなので候補コストの分布が楽観側に偏り、結合比 `viterbi_weight` の最適値が系統的にずれる。**使わない理由をレポートに書く**（使わないこと自体が判断である）。
5. **判定は McNemar・同一例ペアリング。** 比較は必ず「同じ 1 つの split の同じ例集合」に対する L / R / LR の 3 通り。Phase 3 が踏んだ「未学習 baseline と比較する」バグを構造で防ぐため、**比較対象のペアリングは集計モジュールの契約としてテストで固定する**（§3.5）。
6. **既存 suite（AJIMEE 200 / Mozc 498 / handmade 14）は採否判断に使わない。** Mozc は context を持たずリランカが 1 度も走らない（issue コメント）。handmade は mock で既に Top-1 `1.000`。AJIMEE は `convert_sentence` の文分割が混ざる。**回帰検出の補助としてのみ値を載せる。**

### 3.5 集計・統計判定モジュール（タスク 3）

ユーザーとの方針確認で明示された「統計の教訓を最初から織り込む」への直接の対応。Phase 3 では McNemar の比較対象を間違え（Critical）、ラベルが代表 seed 1 本で決まっていた。**Phase 4 では集計・検定を CLI のスクリプト本文に書かず、テスト可能なモジュールに切り出す。**

`src/beme/` は増やさず、`scripts/` から import できる `scripts/_rerank_stats.py`（または `src/beme/rerank_stats.py`。タスク 3 の指示書で 1 つに決める）に次を置く。

| 関数 | 契約 | 回帰テスト |
|---|---|---|
| `paired_table(before, after)` | 同一 id で突き合わせ、id 集合が一致しなければ **例外**。順序に依存しない | id 欠落・重複・順序入替で例外/同値を固定 |
| `mcnemar(b, c)` | discordant < 25 は正確二項、以上は連続修正 χ²。Phase 3 の実装（`run_ablations.py:61-71`）と同一値 | Phase 3 の既知値を golden で固定 |
| `top1_from_scores(candidates, vw, ce_w, rerank_k)` | §3.2 のソートキー。**`Converter.convert` と一致することをテストで固定** | 実 converter との突き合わせ |
| `stratify(records, axis)` | contrast pair / 読み長 / 固有名詞 / span_tokens | 空層・低支持（n < 30）のフラグ |
| `pair_level(records)` | contrast pair を pair id 単位に畳む（両方正解 / 片方 / 両方誤り） | 対が片側しか無い場合の扱いを固定 |

**同点処理を契約に含める**: Top-1 が同点になるケースは §3.1 の統一後も `(cost, surface)` で一意に決まるが、**リランク後のスコアは実数なので同点はまず起きない代わりに、`vw=0` では relevance が完全同値になりうる**（同一 surface が 2 度出ることは dedup 済みなので無いが、CE が同じ値を返すことはある）。ソートキーの第 2・第 3 要素（`cost`・`surface`）が必ず効くことをテストで固定する。

**レビューは実装者と別モデルで行う**（方針確認 5）。タスク 3 完了後、`/code-review` とは別に、集計・検定部分だけを対象にしたクロスレビューを 1 回入れる。

### 3.6 cascade ゲートとレイテンシ（タスク 6・**結果次第**）

方針確認 4 のとおり、**配線は Phase 4 の最後**に置く。まず「リランカが効くケース」を同定してから、そのケースにだけ適用する設計にする。

前提となる数字（§2.3）: 1 変換あたり約 3.0 秒。全候補に CE を回す設計は成立しない。

削減の手段は 2 つあり、**効果と副作用が違う**ので分けて測る。

| 手段 | 削減率 | 副作用 |
|---|---|---|
| `rerank_k` を絞る | 候補数に比例（`rerank_k=10` なら 47.9 → 10 で **約 4.8 倍高速・約 630 ms**） | `k > rerank_k` の gold が救えなくなる。ただし §3.3 の伸びしろ表で上限が測れる |
| cascade ゲート（`needs_rerank` の配線） | 発火率に比例 | ゲートが誤って落とした例は救えない。**ゲートの recall がそのまま上限になる** |

**ゲートの評価は精度ではなく「リランカが効いたケースに対する recall / precision」で行う。** タスク 5 の測定で「LR が L から救った c 件」が確定するので、ゲート条件 `g` に対して

```text
recall(g)    = |{c 件のうち g が発火}| / |c 件|
fire_rate(g) = |{全例のうち g が発火}| / 全例
```

を出し、`recall ≥ 0.95` を満たす中で `fire_rate` 最小の `g` を選ぶ。**候補となる `g`**: `confidence < θ`（Phase 3 の再較正値 `2.025` を出発点）、`confidence < θ ∧ 上位 2 候補の表層が同じ読み長`、`gold_rank 分布から導く上位 m 候補のコスト差`。§2.6 のとおり confidence 単独では発火率が 6 割を超える見込みなので、**「confidence 単独では足りない」という結論も正式な成果**として扱う。

`converter.py:174-176` の配線変更は**変換の挙動を変える**ので、変更後に Phase 3 の不変条件（`--suite all --reranker mock` の 4 指標）が 1 bit も動かないことを確認する（ゲートを通さない設定が既定であること）。

**タスク 6 を実施しない条件**: タスク 5 で LR が contrast pair 128 件に対して有意な改善を示さなかった場合、ゲートを配線する理由が無い。その場合はタスク 6 を飛ばし、issue コメントの分岐に従って **「取れなかった場合」の 3 つの切り分け**（結合比・識別力・文脈長）をレポートに書いて #2 / #3 へ引き継ぐ。§5 の分岐表を参照。

### 3.7 リーク監査の完走（タスク 4）

issue コメントの「着手前に片付ける小さな前提」3。Phase 3 の監査は miniPC で実行され、d1 JSONL が無いため document n-gram overlap・reading overlap・train vocabulary overlap・split field 再計算が `unverifiable` のまま残っている（phase3-report §8）。

Phase 4 は AJIMEE / Mozc での改善を主張する場面が来る**かもしれない**ので、GPU 側で `scripts/audit_leakage.py` を d1 JSONL ありで完走させる。**新規実装ではなく実行タスク**である。`unverifiable` が `verified` に変わらない項目があれば、その項目を名指しで残す。

### 3.8 やらないこと（明示）

| 項目 | 判断 | 理由 |
|---|---|---|
| feature spec v2（`lex.reading_len` 削除・共線性解消） | **やらない** | issue コメントの「やらないほうがよいこと」。Phase 3 が示したのはこの標本サイズでは局所モデルの改善を判別できないこと。同じ土俵で条件を増やしても `indeterminate` が並ぶ |
| matrix cost（M0/M1）の採否決定 | **やらない** | 同上。リランカを入れた状態で測り直すほうが判断材料として意味がある。**ただし §5 で LR が効いた場合、X0 での再測定を Phase 5 の起票候補として書く** |
| D1 小型 LM（#5）の学習 | **やらない** | `docs/d1-evaluation-2026-08-22.md` で No-Go 済み |
| LLM logits リランカ（#2）の再評価 | **既定ではやらない。ハーネスは対応させる** | 頻度バイアスで劣ると実証済み（handmade Top-1 `1.000 → 0.929`）。タスク 2 の JSONL 形式は 3 種すべてを受けるので、必要になれば `score` を 1 回回すだけで比較できる |
| 局所モデルの再学習 | **やらない** | Phase 3 の A0 seed 9 の重みをそのまま使う。3 seed 分あるので、**LR の効果が seed をまたいで安定するかは seed 9/17/33 の 3 本で確認する**（§4 の実行マトリクス） |
| 辞書・コーパスの再取得 | **やらない** | ミッション制約 |
| 重みの package-data 化 | **やらない。起票のみ** | Phase 1 から 3 フェーズ持ち越し。Phase 3 の結論（学習済み重みは外部パス必須・`legacy_v1`/`decomposed_v1` のみ同梱）を別 issue として起こす |

---

## 4. 実行マトリクス（タスク 5）

局所モデルは Phase 3 の **A0**（Phase 3 §10.1 の結論「改善側がすべて indeterminate なので A0 を既定とする」に従う）。

**`score` フェーズ**（CE 推論が要る。ここだけが重い）

| # | split | 重み | リランカ | ペア数 | 見積り（beme-gpu） |
|---|---|---|---|---:|---:|
| S1 | d1 dev 1,000 | A0 seed 9 | cross-encoder | 約 48k | 10–15 分 |
| S2 | d1 test 2,000 | A0 seed 9 | cross-encoder | 約 96k | 20–30 分 |
| S3 | d1 test 2,000 | A0 seed 17 | cross-encoder | 約 96k | 20–30 分 |
| S4 | d1 test 2,000 | A0 seed 33 | cross-encoder | 約 96k | 20–30 分 |
| S5 | d1 test 2,000 | A0 seed 9 | mock（対照） | 約 96k | 1 分未満 |

S3/S4 は **「LR の効果が seed をまたいで安定するか」**の確認であり、Phase 3 の「代表 seed 1 本でラベルが反転した」教訓への対応である。S1 の選択は seed 9 のみで行い、S3/S4 には**同じ `(viterbi_weight, rerank_k)` をそのまま適用する**（seed ごとに選び直さない）。

**`combine` フェーズ**（推論ゼロ・全グリッド）

| # | 入力 | 出力 |
|---|---|---|
| C1 | S1 | グリッド全面（7 × 4 = 28 点）× {全体 Top-1, contrast Top-1}。**選択はここで 1 回** |
| C2 | S2 | 選択した 1 点で L / R / LR の 3 条件 × {全体, contrast 128, pair 64 対, 層別, gold_rank 分布, 失ったケース一覧} |
| C3 | S3, S4 | C2 と同じ 1 点。seed 安定性 |
| C4 | S5 | mock 対照（context ありでも relevance がほぼ 0 の対照群） |
| C5 | S2 | ゲート候補の recall / fire_rate 曲線（タスク 6 の入力） |

**既存 suite の回帰確認**（採否判断には使わない・§3.4-6）: `--suite all --reranker mock` の 4 指標が Phase 3 と bit 一致すること。タスク 1 と 6 の後に各 1 回。

---

## 5. 結果による分岐

タスク 5 の C2 が出た時点で、Phase 4 の残りは自動的に決まる。**分岐条件を先に固定しておく**（実行後に基準を動かさないため）。

| 分岐 | 条件 | 残りの作業 |
|---|---|---|
| **効いた** | contrast pair 128 件で LR が L に対し McNemar p < 0.05 かつ改善方向、**かつ 3 seed すべてで同じ向き** | タスク 6（cascade ゲート配線・レイテンシ予算）を実施。レポートは配線設計と `rerank_k` / 閾値の根拠を含む。Phase 5 の起票候補: X0 での再測定・AJIMEE への文脈供給・評価セット拡張 |
| **効いたが不安定** | p < 0.05 が seed 9 のみ、または全体 2,000 件で悪化 | **タスク 6 を実施しない。** Phase 3 の A6 と同じ扱い（`indeterminate`）にし、「2,000 件・128 件では判別できない」と書く。次に必要なのは評価セットの拡張であると結論する |
| **効かなかった** | contrast pair でも改善が判別できない | タスク 6 を実施しない。issue コメントの 3 つの切り分けをレポートに書く: (1) 結合比 — `vw` グリッド全面は C1 で既に手元にある、(2) 識別力 — gold_rank 分布は C2 で既に手元にある、(3) 文脈長 — contrast 128 件で文脈長と正誤の関係を追加集計する。課題を #2（頻度バイアス）と #3（誤りの分類）へ移す |

**§2.4 のパイロットは「効いた」側を示唆しているが、40 件・p = 0.115・選択バイアスありなので、これを根拠に分岐を先取りしてはならない。** タスク 6 の着手可否は C2/C3 の実測で決める。

---

## 6. タスク分割

各タスクは worker 1 回の指示で完結するサイズにしてある。指示書は `docs/missions/issue9-phase4/taskN-mission.md`。

| 依存順 | Task | 主対象 | 難易度 | 見積り | 依存 |
|---:|---|---|---|---:|---|
| 1 | 1 | Top-1 定義の統一: `src/beme/training.py` / `src/beme/lattice.py`（読むだけ）/ 新ヘルパ | 中 | 3–5h | なし |
| 1 | 3 | 集計・統計判定モジュールと回帰テスト | 中 | 4–6h | なし |
| 1 | 4 | リーク監査の完走（GPU 実行） | 低 | 1–2h | なし |
| 2 | 2 | リランカ評価ハーネス `scripts/rerank_eval.py`（score / combine） | **高** | 6–9h | 1 |
| 3 | 5 | 測定実行 3 条件 × dev/test（GPU マシン） | 低（実装）/ 中（運用） | 2–3h + 実行 1.5–2h | 1–4 |
| 4 | 6 | cascade ゲート配線とレイテンシ予算（**条件付き**） | 中 | 4–6h | 5 |
| 5 | 7 | Phase 4 レポート | 中 | 4–6h | 5（6） |

タスク 1・3・4 は互いに独立で並行可能。**タスク 2 が唯一の高難度**で、リスクは「`combine` の再現ソートが `Converter.convert` と乖離する」ことに集中している（§3.2）。ここだけは突き合わせテストを必須にしてある。

**Phase 4 合計の見積り**: 実装 約 17–26 時間 + 実行 約 2–3 時間 + レポート 4–6 時間。タスク 6 が実施されない分岐なら 4–6 時間短くなる。

---

## 7. 実行場所の判断

**結論: CE スコアリング（タスク 5）と リーク監査（タスク 4）は GPU マシン（`beme-gpu`）で行う。実装・テスト・レイテンシの参照測定は miniPC で行う。**

| 検討点 | 判断 |
|---|---|
| d1 の `*.jsonl` の所在 | GPU マシンのみ。Phase 3 と同じ理由（gitignore 対象データの複製を増やすと出所の追跡が壊れる）で miniPC へ持ってこない |
| cross-encoder モデルの所在 | miniPC のみ（2.7 GB）。**GPU マシンはネットワークがあるので `local_files_only=False` で 1 回だけ取得する。** 取得後の SHA-256 とリビジョンを artifact に記録すること |
| 計算資源 | 24 core vs 4 core。CE は CPU 飽和型なのでコア数がそのまま効く（§2.8 の見積り） |
| CUDA torch の導入 | **不要**。24 core CPU で 30–45 分に収まる。導入すればさらに速いが、環境変更のリスクに見合わない。**GPU マシンは実質「速い CPU 機」として使う**（Phase 3 と同じ判断） |
| レイテンシの測定 | **miniPC で測る。** レイテンシは実験の性質ではなく実行機の性質であり、IME の実行環境として N100 のほうが現実的である。GPU マシンの値は「スコアリングにかかった時間」として別に記録し、**両者を取り違えないようレポートで実行機を明記する** |
| GPU マシンの作業ディレクトリ | `~/beme-p3` は未コミット変更を持つ。**`~/beme-p4` に本ブランチを新規クローンし、`data/d1` へ `~/beme-d1/repo/data/d1/*.jsonl` を配置する。** git クローンにするのは artifact の `git_commit` を `None` にしないため（Phase 2/3 と同じ） |
| 並列実行 | S2/S3/S4 は独立なので 3 プロセス並列でよい。ただし **`CrossEncoderReranker` は `_lock` で内部直列化されており、1 プロセス内での並列化はできない。** プロセスを分け、`torch.set_num_threads` を 24/3 = 8 に固定して合計コア数を超えないようにすること |

---

## 8. レポート構成（タスク 7・`phase4-report.md`）

1. **Scope and Reproducibility** — 実行機・commit・全コマンドの逐語・入力 SHA-256・CE モデルのリビジョン
2. **Top-1 Definition Unification** — §3.1 の統一結果。A3 の 208 件乖離が解消したことの実測。11 条件が不変であることの実測
3. **Separating Oracle Recall from Top-1** — §3.3 の 3 段の表。**リランキングが recall@k を不変にすることの実測アサーション**
4. **Grid Selection on dev** — C1 のグリッド全面（28 点）。選択規則と選択点。高原かスパイクか
5. **Local / Reranker-only / Combined** — C2 の 3 条件比較。primary（contrast 128）・secondary（全体 2,000）・pair 単位（64 対）
6. **Seed Stability** — C3。seed 9/17/33 で向きが一致するか
7. **What the Reranker Broke** — 失ったケース（b）の全件一覧と分類。**集計値だけで済ませない**（Phase 2 の反省）
8. **gold_rank Distribution** — リランク前後。「上位にも来ない」のか「惜しい」のか
9. **Latency** — miniPC 実測（候補あたり / 変換あたり）と `rerank_k` による削減。**実行機を明記**
10. **Cascade Gate**（タスク 6 実施時のみ） — ゲート候補の recall / fire_rate 曲線、選んだ条件、`converter.py` の配線変更、不変条件の再確認
11. **Leakage Audit** — タスク 4 の完走結果。`unverifiable` が残った項目の名指し
12. **Negative and Null Results** — 効かなかった条件、判別できなかったもの、実施しなかったこととその理由
13. **Limitations and Handoff** — §5 の分岐に応じた次の作業。起票する別 issue（package-data・評価セット拡張）

---

## 9. リスクと緩和

| リスク | 影響 | 緩和 |
|---|---|---|
| `combine` の再現ソートが `Converter.convert` と乖離する | **Phase 4 の全数値が無意味になる** | タスク 2 の必須テストで、少数例に対し実 converter（実リランカ注入）と `combine` の Top-1 が完全一致することを固定（§3.2） |
| Top-1 定義の統一で学習が遅くなる | 以降の再学習が現実的でなくなる | ×1.2 の性能ゲートと early-stop（§3.1）。実測を報告に含める |
| dev の contrast が 44 件しかない | ハイパーパラメータ選択が不安定 | 選択の主指標を dev 全体 1,000 件にし、contrast は副指標。高原の確認を必須にする（§3.4-2,3） |
| グリッド最良点を test で報告してしまう | Phase 3 の「同じ集合で選択と評価」を再演 | 選択は dev で 1 回。test は選択後 1 回だけ。**規則を実行前に固定し、レポートに逐語で書く**（§3.4-1） |
| 40 件のパイロットを結果として引用する | 選択バイアスと標本サイズの誤用 | §2.4 に「言えないこと」を明記済み。レポートでは**パイロットを結果として引用しない**。分岐の先取りをしない（§5） |
| contrast pair 128 件でも判別できない | 結論が出ない | それ自体が結論。**「128 件では判別できない」と書く。** パイロットの効果量なら足りる見込みだが、足りなかったときの次の一手（評価セット拡張）を §5 に用意済み |
| レイテンシの実行機を取り違える | 「3.0 秒」と「0.6 秒」が混在して読者が誤解する | レポートの全レイテンシ数値に実行機名を併記（§7・§8-9） |
| CE の推論結果がスレッド数で揺れる | 2 回実行の byte-identical が破れる | `torch.set_num_threads` を固定。それでも揺れるなら**丸めずに、揺れる事実と幅を記録する**（§3.2） |
| `context_empty` の 18 件を 0 として混ぜる | 集計が静かに歪む | `relevance = null` で記録し、combine で「リランクしない」として扱う。**件数を出力に必ず残す**（§3.2） |
| cascade ゲートを confidence 単独で組む | 発火率 6 割超でレイテンシが減らない | §2.6 の実測を前提に、recall / fire_rate 曲線で評価する（§3.6）。「confidence 単独では足りない」も正式な結論 |
| 集計・検定バグ（Phase 3 の再演） | 誤った結論 | 集計を CLI 本文から分離しテスト可能にする。ペアリング契約を例外で守る。**実装者と別モデルでクロスレビュー**（§3.5） |
| GPU マシンで CE モデルの取得に失敗 | タスク 5 が止まる | フォールバック: miniPC へ d1 dev/test（5.4 MB）をコピーして実行（約 2.5 時間）。**その場合は SHA-256 を artifact に記録して出所を追跡可能にする** |

---

## 10. Phase 4 のチェックリスト対応

| issue のチェックリスト項目 | 対応するタスクと設計 |
|---|---|
| 局所モデルの K-best を #2 / #5 のリランカへ渡す | タスク 2（ハーネス）+ タスク 5（実行）。**配線自体は既に在る**（`converter.py:176`）ので、Phase 4 の実質は「渡した結果どうなるか」の測定である。#5（D1 小型 LM）は No-Go 済みなので #2 系（cross-encoder / LLM logits）を対象とし、ハーネスは 3 種すべてを同じ JSONL 形式で受ける（§3.2・§3.8） |
| oracle recall と Top-1 改善を分けて評価 | §3.3。**リランキングは候補集合を変えないので recall@k は k ≥ min(rerank_k, 候補数) で不変**であることを形式化し、実測でアサートする。伸びしろ（`recall@rerank_k − recall@1`）と実績（Top-1 差）を別の行として出す |
| 局所モデル単独・リランカ単独・併用を比較 | §3.4 の L / R / LR。「リランカ単独」は「候補生成は同一、順位付けは relevance のみ（`viterbi_weight = 0`・`rerank_k = beam_width`）」と定義する。**IME は候補生成を省けないので「候補生成なし」は条件にしない** |

---

## 11. Worker 指示文

- `docs/missions/issue9-phase4/task1-mission.md` — Top-1 定義の統一
- `docs/missions/issue9-phase4/task2-mission.md` — リランカ評価ハーネス（score / combine）
- `docs/missions/issue9-phase4/task3-mission.md` — 集計・統計判定モジュールと回帰テスト
- `docs/missions/issue9-phase4/task4-mission.md` — リーク監査の完走
- `docs/missions/issue9-phase4/task5-mission.md` — 測定実行（GPU マシン）
- `docs/missions/issue9-phase4/task6-mission.md` — cascade ゲート配線とレイテンシ予算（条件付き）
- `docs/missions/issue9-phase4/task7-mission.md` — Phase 4 レポート
