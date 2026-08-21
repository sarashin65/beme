# P2 + Issue #1 実装プラン

## 1. P2 と #1 の関係性・実装方針

**結論: 一括実装（同時に着手）が最も効率的**

**理由:**
- Issue #1（ Exact K-best, パス永続化, Pull型DP）は `lattice.py` における経路探索の基盤を根本から書き換えます。
- P2（文レベルK-best）は、現在 `benchmark.py` の `convert_sentence` でスパン（記号等の非かな文字）ごとに `k=5` で切り詰めて合成している問題（局所Pruning）を修正するものです。
- 真の文レベルK-bestを実現するには、内部のラティス抽出が Exact K-best であることが大前提となります。別々に実装すると、LatticeのインターフェースやK-best取得の仕組みの再設計・手戻りが発生するため、同時に実装するのが最適です。

## 2. アルゴリズム選択と設計

### Exact K-best のアルゴリズム選択
- **選択**: **Lazy K-best / Eppstein's Algorithm (A* 探索)**
- **比較**:
  - *Lazy K-best (A*)*: 終端から逆向きに1-best Viterbiコスト（ヒューリスティック）を前計算 (O(V+E))。その後、優先度付きキュー(A*)を用いて始点から Exact にK個抽出します（参照: `ime_nbest_research_2026-08-20.md`）。計算効率が高く、必要なK個だけ展開されるためメモリ効率も優れています。
  - *Push型 Exact DP*: 各ノードで K 個を正確に保持するのは遅く、計算量が跳ね上がります。
  - *文節ごと Top-N*: 経路の一貫性が壊れ、LLM リランク（Phase 2）の文脈崩壊を招くため非推奨です。
- **性能見積もり**:
  - A* 探索により、ビーム幅による「探索誤り」が数学的に 0 となります。
  - 経路展開はオンデマンドになるため、LLMリランカへ渡すために k=64 などに増やしてもレイテンシ悪化を最小限に抑えられます。

### パス永続化・Pull型DPの設計
- **パス永続化**:
  - `LatticePath` における `tokens` (tuple) の保持をやめ、`token` と `parent: LatticePath | None` を持つ**片方向リスト（Linked List）**に変更します。
  - これにより、エッジ追加ごとの tuple 完全コピーが不要になり、DP 遷移が $O(N)$ から $O(1)$ に高速化されます。
  - `surface` や `tokens` は、終端に到達して K-best に選ばれたパスについてのみ、逆順に辿って一度だけ復元します（遅延評価）。
- **Pull型DP化**:
  - 各ノードへのエッジ追加時に都度 `destination.sort()` する Push 型をやめます。
  - まず Backward DP（Pull型で「次ノードの最小コスト」を引いて自身のヒューリスティックを確定）を 1 回だけ回すため、各ノードでのソートは不要になります。

### 文レベル K-best 合成の設計 (P2)
- **現状の問題**: `convert_sentence` 内で、スパン（非かな文字で分割されたチャンク）ごとに `converter.convert(k=k)`（デフォルト k=5）を呼び、各スパンのトップ5同士だけを合成して再びトップ5を取るという局所 Pruning を行っています。これにより正解パスが早期に脱落しています。
- **改善設計**:
  - 非かな文字を分割してスパンごとに処理するのではなく、**文全体を単一の Lattice として扱う**ように `lattice.py` および `converter.py` / `benchmark.py` を改修します。
  - 非かな文字（記号等）は、「コスト0・1文字確定ノード」として Lattice に組み込みます。
  - スパン境界の接続コストも自然に計算できるようになり、文全体を通した Exact K-best が一発の A* 探索で得られるようになります。

## 3. 完了条件・検証コマンド

- **コマンド**: `scripts/evaluate.py --suite ajimee --json`
  - AJIMEE の CER（現在 0.3414付近）および exact_match_accuracy が悪化しない、または改善すること。
- **コマンド**: `scripts/evaluate.py --suite mozc --json`
  - Mozc 回帰テストの pass_rate が維持されること。
- **レイテンシ**:
  - Exact K-best 化を行っても、実行時間が現在と同等、またはパス永続化の恩恵で高速化されていること。
  - k を大きく設定した際の速度低下が緩やかであること。

## 4. 実装のタスク分割（Luna 向けミッション群）

**Mission 1: LatticePath の永続化と片方向リスト化**
- `src/beme/lattice.py` を修正し、`LatticePath` を片方向リスト（`token`, `parent`）構造にする。
- プロパティ `tokens`, `surface`, `reading` は、要求された時のみリストを辿って構築するよう変更する。
- 既存のテスト・コマンドが現状通り動作することを確認する。

**Mission 2: Viterbi 前計算 (Backward DP) の実装**
- `src/beme/lattice.py` に、終端ノードから始点ノードに向かって「終端までの最小コスト」を計算する Backward DP 処理を追加する。
- この値は、次の Mission で実装する A* 探索のヒューリスティック関数として利用する。

**Mission 3: Eppstein (A*) による Exact K-best の実装**
- `k_best_viterbi` を、優先度付きキュー (`heapq`) を用いた A* 探索に置き換える。
- Mission 2 で得た最小コスト（ヒューリスティック）を用いて、始点から K 個のパスが抽出されるまで Exact な順序で探索を行う。
- `beam_width` による打ち切りを廃止する。

**Mission 4: 文全体ラティス化（スパン境界排除）による P2 実現**
- `src/beme/benchmark.py` と `src/beme/lattice.py` を修正する。
- 現在の `convert_sentence` における「非かな文字での分割・各スパン k=5 抽出・再合成」の処理を廃止する。
- 非かな文字・記号も Lattice 内の固定エッジ（確定パス）として通せるよう `build_lattice` を拡張し、文全体を通した Exact K-best を1回の `converter.convert` で出力できるようにする。
