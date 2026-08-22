# Issue #9 Phase 2（正解経路と学習）実装プラン

作成日: 2026-08-22  
対象: `feat/issue9-phase2` @ `4a7a868`（`origin/main`）  
状態: **プランのみ。実装未着手。**  
正本: GitHub issue #9 本文・owner コメント、および `docs/missions/issue9/PLAN.md` §3.7

## 0. 結論

Phase 2 は次の 6 タスクに分ける。

1. 既存 `Lattice` と `Scorer` を入力にする正解表記制約付きオラクル DP
2. 学習 JSONL と train/dev/test 分割の厳格なデータ契約
3. cost 最小化の符号を明示した averaged structured perceptron
4. seed・入力ハッシュ・選択 epoch を記録する再現可能な学習 CLI
5. 公開探索 API の後方互換、負コスト、別プロセス再現を守る統合テスト
6. 実データでの最終実行と Phase 2 レポート

オラクルは探索コアを改造せず、DP 内だけで `(reading_pos, target_pos, previous_state)` と位置付き edge を保持し、返り値は既存の `LatticePath` とする。これにより `Converter`、K-best、後方 DP の外部契約を変えず、Phase 1 の `path_features` をそのまま使える。

## 1. 調査で確定した前提

### 1.1 Phase 1 の接続点

- 遷移状態は必ず `scorer.state_key(edge)` を使う。遷移特徴 `ψ` はこの state key のみに依存する。
- `path_features(path, extractor, scorer.weights)` と `feature_delta` が実装済み。学習中は**現在の重みを必ず渡す**。
- `LinearScorer.invalidate(new_weights)` で重み差し替えとキャッシュ破棄ができる。
- `LatticePath` は Token 列だけを持ち、位置は持たない。Phase 2 ではオラクル DP の内部状態にのみ位置を持たせる。
- 特徴は位置非依存である。`LinearScorer` のキャッシュが `edge.token` をキーにするため、Phase 2 で `edge.start/end` 依存特徴を追加してはならない。
- 学習重みは負になりうる。ラティスは DAG で、`backward_costs_by_state` は正確な残余 cost を返すため、負 cost 自体は K-best の正しさを壊さない。

### 1.2 更新式の符号

issue 本文は score 最大化形として次を記載している。

```text
θ_score ← θ_score + η(Φ_gold − Φ_pred)
```

一方、現行 `LinearScorer` と探索は `C_w(P)=w·Φ(P)` の**最小 cost**を選ぶ。この `w` へ上式をそのまま適用すると gold cost を上げ、pred cost を下げてしまう。Phase 2 の保存形式は既存の cost weights と互換にするため、実装上の更新は等価な次式に固定する。

```text
w_cost ← w_cost + η(Φ_pred − Φ_gold)
       = w_cost − η feature_delta(gold, pred)
```

これは `θ_score = -w_cost` と置けば issue の式と同一である。単体テストでは 1 更新後に `C(gold)-C(pred)` が小さくなることを必須とし、符号の再反転を防ぐ。

### 1.3 訓練データの現状

依頼にある `data/d1/train.jsonl` / `dev.jsonl` / `test.jsonl` は、対象 commit `4a7a868` の作業ツリーにも git 履歴にも存在しない。#5 の計画書から確認できる候補スキーマは `reading` と `positive`（正解表記）、左右文脈、記事 ID、候補列を持つ 10k/1k/2k の article-disjoint split である。

したがって実装は次の二段構えにする。

- loader は canonical `target_surface` と #5 の `positive` alias の両方を厳格に受ける。
- unit/integration test はリポジトリ内の小さな synthetic fixture で完結させる。
- Task 6 の実データ検証は 3 ファイルが供給されるまで実行不能。データを worker が推測生成して埋めてはならない。
- 実データの件数、SHA-256、source、split 名、article ID の交差を manifest/report に記録する。

## 2. 設計

### 2.1 オラクル DP

新規 `src/beme/oracle.py` に次の純粋関数を置く。

```python
def constrained_best_path(
    lattice: Lattice,
    target_surface: str,
    scorer: Scorer,
) -> LatticePath | None:
    ...
```

DP 状態は `(input_position, target_position, previous_state)`。`input_position` の outgoing edge のうち、`target_surface.startswith(edge.token.surface, target_position)` を満たす edge だけを遷移させる。遷移 cost は既存探索と同じく

```text
scorer.transition_cost(previous_state, edge) + scorer.edge_cost(edge)
```

であり、遷移後 state は `scorer.state_key(edge)`。同一状態には current model で最小 cost の部分経路だけを残す。cost tie は既存 `_path_key` と同じ token 列キーで決定的に解く。終端 `(len(reading), len(target_surface), *)` の最良を latent gold として返し、存在しなければ `None` を返す。

`None` は coverage error の唯一の判定とする。K-best の有限幅で gold を探す実装、正解形態素境界の要求、surface の Unicode 正規化は行わない。入力 JSONL は既に正規化済みの exact surface 契約とする。

### 2.2 データ契約

新規 `src/beme/training_data.py` に immutable `TrainingExample`、JSONL loader、3 split 検証、SHA-256 manifest 作成を置く。

- 必須: 非空 `reading`、正解表記（`target_surface` または `positive`）。両方ある場合は同値必須。
- 任意: `id`, `source`, `article_id`, `left_context`, `right_context` と、#5 が持つ追加 metadata。未知 field は保持せず、許可リスト外はエラーにして schema drift を可視化する。
- JSON duplicate key、blank line、非 UTF-8、重複 ID、不正型を行番号付きで拒否する。
- split 間で `article_id` が交差したらエラー。全行に article ID が無い場合は manifest に `article_disjoint_verified=false` と明記する。
- context は provenance と leakage 検査のため読むが、Phase 2 の局所 scorer へは入力しない。
- manifest は split ごとの path、SHA-256、件数、source 集計、article ID 検証結果を key sort した JSON とする。

### 2.3 学習・平均化・エラー分類

新規 `src/beme/training.py` に学習器と評価器を置く。既定は `dictionary_v2.tsv` + `decomposed_v1.json`。

各 covered train 例で以下を行う。

1. 現在の scorer で unconstrained 1-best `P_pred` を取得。
2. 同じ scorer で constrained latent `P_gold` を取得。
3. `P_gold is None` なら coverage error を加算し、更新も平均化 step も行わない。
4. covered 例は正誤にかかわらず averaging step を 1 増やす。
5. `P_pred.surface != target_surface` のとき model error とし、`path_features(..., scorer.weights)` から `w += η(Φ_pred−Φ_gold)` を適用する。
6. 新しい immutable `Weights` を作り `scorer.invalidate(weights)` する。

平均は「各 covered example 処理後の current weights」の算術平均とする。更新がない正解例も平均へ含め、coverage error は分母に含めない。出力 weights の metadata には `averaged=true`, `updates`, `averaging_steps`, `epochs`, `learning_rate`, `seed`, 初期重み・辞書・split の SHA-256 を残す。

評価指標は split ごとに最低限以下を返す。

- `examples`
- `coverage_errors`, `coverage_error_rate`（分母は全例）
- `covered_examples`
- `model_errors`, `model_error_rate`（分母は covered）
- `end_to_end_errors`, `end_to_end_error_rate`（coverage + model、分母は全例）
- `exact_matches`

dev の model 選択は `(end_to_end_error_rate, model_error_rate, epoch)` の辞書順最小とし、同値なら早い epoch。test は epoch 選択完了後に baseline と選択済み averaged model を各 1 回だけ評価する。

### 2.4 再現可能 CLI

新規 `scripts/train_scorer.py` は少なくとも次を受ける。

```text
--train / --dev / --test
--dictionary
--initial-weights
--epochs
--learning-rate
--seed
--output-weights
--output-report
```

各 epoch の train 順は `random.Random(seed + epoch).shuffle(indices)` とし、集合・辞書の走査順を学習順に使わない。dev だけで epoch を選び、test は最後に一度だけ使う。weights と report は一時ファイルから atomic replace し、JSON は UTF-8・`ensure_ascii=False`・`sort_keys=True`・`indent=2`・末尾改行で決定的に出す。

推奨実行形:

```bash
uv run python scripts/train_scorer.py \
  --train data/d1/train.jsonl --dev data/d1/dev.jsonl --test data/d1/test.jsonl \
  --dictionary data/dictionary_v2.tsv \
  --initial-weights data/weights/decomposed_v1.json \
  --epochs 5 --learning-rate 0.1 --seed 9 \
  --output-weights artifacts/issue9-phase2/weights.json \
  --output-report artifacts/issue9-phase2/report.json
```

同じ commit・入力 bytes・引数で 2 回実行した weights/report はバイト一致させる。wall clock、絶対パス、実行時間は決定的 report に入れない。

### 2.5 後方互換の境界

- `src/beme/lattice.py` の K-best/後方 DP、`src/beme/converter.py` の public interface は変更しない。
- オラクルは `LatticePath` を返すため `path_features` の public contract を変えない。
- 位置依存特徴を追加しない。追加するなら Phase 3 以降に `LatticePath` と scorer cache key を同時再設計する。
- `_edge_key` / `_path_key` と既存 tie-break を変えない。
- 辞書の負値制約は緩めない。負値は学習 weights 側だけで扱う。
- 学習済み weights は CLI の明示 path で読み込む。wheel package-data 問題は Phase 2 の学習正当性とは分離し、Phase 3/配布時の引き継ぎとする。

## 3. タスク分割

| 依存順 | Task | 契約・主対象 | 難易度 | 見積り | 依存 |
|---:|---|---|---|---:|---|
| 1 | 1 | 制約付きオラクル DP: `src/beme/oracle.py` | 高 | 4–6h | Phase 1 |
| 1 | 2 | JSONL/split/manifest: `src/beme/training_data.py` | 中 | 3–5h | なし |
| 2 | 3 | averaged perceptron と指標: `src/beme/training.py` | 高 | 6–8h | 1, 2 |
| 3 | 4 | 再現可能 CLI: `scripts/train_scorer.py` | 高 | 5–7h | 3 |
| 4 | 5 | end-to-end・互換・負 cost テスト: `tests/test_phase2_integration.py` | 高 | 4–6h | 1–4 |
| 5 | 6 | 実データ検証・README・Phase 2 report | 中 | 3–5h + 学習時間 | 5、d1 artifact |

Task 1 と 2 は独立。Task 3 は符号と平均化の中核なので分離し、Task 4 に CLI の副作用、Task 5 に横断的な回帰 gate を集約する。各 worker は対応する `taskN-mission.md` だけで実装できる。

## 4. 全体完了条件

- `(reading, target_surface)` から、複数 gold segmentation のうち現モデル最良の `LatticePath` を取得できる。
- gold surface が lattice に無い例が coverage error として、あるが 1-best でない例が model error として別集計される。
- cost 符号が正しく、1 更新で gold-vs-pred margin が改善する。
- averaged weights を別プロセスの `scripts/evaluate.py --weights ...` で読み、同じ候補を再現できる。
- seed 付き同一コマンド 2 回の weights/report が byte-identical。
- dev のみで epoch を選び、held-out test は選択後 1 回だけ baseline と比較される。
- `uv run pytest -q` が全通し、既存 legacy/decomposed golden と全評価 suite の既存挙動を壊さない。
- handmade は learned weights でケース単位の before/after を report し、aggregate だけで回帰を隠さない。

## 5. リスクと緩和

| リスク | 影響 | 緩和 |
|---|---|---|
| issue 更新式と cost 最小化の符号差 | 学習が逆方向 | score/cost の対応を文書化し、1-step margin test を必須化 |
| d1 が現 branch にない | 実データ gate を実行不能 | synthetic test と実データ Task 6 を分離。欠損時は明示失敗 |
| d1 が #5 の K-best 内 gold のみ | coverage が人工的に高い | 生成辞書・candidate 条件を report。coverage を現在 lattice で再計算 |
| Wikipedia/AJIMEE 重複 | held-out 値の過大評価 | article split と fingerprint/provenance を記録。AJIMEE は補助指標扱い |
| MeCab かな化バイアス | 実入力との分布差 | source・固有名詞等を report。Phase 3 で層別評価 |
| 更新ごとの cache invalidate | 10k×epoch の性能低下 | correctness 優先。Task 6 で wall time を別記録し、最適化は測定後 |
| 負 weights と K-best | 誤った「非負」前提の再導入 | exact remaining-cost の負値 regression test |
| 位置特徴の追加 | token-key cache が誤値を返す | Phase 2 は位置非依存に固定 |
| learned cost scale の変化 | confidence/cascade/reranker blend が変動 | Phase 2 report に明記し、再校正は Phase 3 |
| wheel に weights 非同梱 | 配布後 default scorer が動かない | Phase 2 は明示 path。package-data 方針を Phase 3 へ引き継ぐ |

## 6. Phase 3 への引き継ぎ

- lexical/transition、IPADIC mapped cost、source、script 等のアブレーション。
- reading/surface/POS-detail/right-left ID/conjugation の高 cardinality 特徴を追加する場合の feature spec v2 と外部 weights 格納。
- oracle recall@k、AJIMEE CER、Mozc、handmade、d1 test の統合評価と統計的不確実性。
- learned cost scale に合わせた `cascade_threshold`、reranker の `viterbi_weight`、span 合成 cost の再校正。
- d1 の Wikipedia かな化バイアス、AJIMEE leakage、固有名詞・長さ別の層別評価。
- position-dependent 特徴が必要なら `LatticePath` の edge/position 保持と scorer cache key を同時に再設計。
- 学習済み weights の package-data 同梱または外部モデル配布方針。
- reserved `weights_path` による疎な大規模重み外部化は Phase 1 では未実装。語彙特徴導入前に実装する。

## 7. Worker 指示文

- `docs/missions/issue9-phase2/task1-mission.md`
- `docs/missions/issue9-phase2/task2-mission.md`
- `docs/missions/issue9-phase2/task3-mission.md`
- `docs/missions/issue9-phase2/task4-mission.md`
- `docs/missions/issue9-phase2/task5-mission.md`
- `docs/missions/issue9-phase2/task6-mission.md`
