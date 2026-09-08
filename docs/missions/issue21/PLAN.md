# issue #21: 疎な語彙重み（pair + backoff）実装計画

## 概要

現行の A8 + C0（F2: 3個のコーパス頻度スカラー特徴）に対して、**ペアごとに個別の学習可能な重み**を持つ疎な語彙重みを追加する。F2 は全ペア一律のスカラー特徴だったが、疎重みはペアごとに最適化可能になる。

## 設計判断

### 1. weights I/O — JSON マニフェスト + バイナリ sidecar（v2 format）

- 既存の `Weights` クラス（JSON 1ファイル）は**変更しない**
- 疎重み用の `SparseLexicalWeights` クラスを新設（`src/beme/sparse_weights.py`）
- 形式: JSON マニフェスト + バイナリ sidecar（`.sparse.bin`）
  - マニフェスト: メタデータ（feature_spec, dictionary_sha256 等）+ sidecar への参照
  - バイナリ: 4バイト値のフラット配列（各ペア/表記/読みの重みを float32）
  - インデックス: ペア/表記/読みの文字列 → 配列インデックスのマッピング

### 2. backoff 機構

- 3 レベルの疎重みテーブル:
  1. `w_{r,s}`: (読み, 表記) ペア → 重み（最も密）
  2. `w_s`: 表記のみ → 重み（中間）
  3. `w_r`: 読みのみ → 重み（最も疎）
- フォールバック: w_{r,s} が存在 → 使う → なければ w_s → なければ w_r → なければ 0.0
- 学習時: 3 レベルすべて独立に学習。ただし親レベルの重みが実際に学習されるようにする（子レベルが存在しないケースで backoff が機能する）

### 3. 学習器の拡張

- 現行の `train_scorer.py`（ averaged structured perceptron）をそのまま使う
- 疎重みのパラメータ数: 最大 95,987（pair）+ 93,689（surface）+ 64,389（reading）= 254,065
- 正則化: L2 正則化を追加（`train_scorer.py` に `--l2-regularization` 引数）
- 初期値: 全重み 0.0（既存 28 パラメータは A8 の学習済み値から初期化）

### 4. 特徴量拡張

- 新しい lexical feature: `lex.sparse_lexical`（1 個の特徴名）
- 値 = backoff で辿り着いた重み（親レベルの重みも含む合計値）
- この特徴の重みは常に 1.0（固定）。重みの学習は疎重みテーブル自体が担う

## タスク分割

### S1: SparseLexicalWeights クラス（src/beme/sparse_weights.py）

- `SparseLexicalWeights` クラス
  - 3 つのテーブル: pair_weights, surface_weights, reading_weights
  - 各テーブルは dict[str, float] で保持（学習時は更新可能）
  - `get(reading, surface) -> float`: backoff 機構
  - `load(path) / dump(path)`: JSON マニフェスト + バイナリ sidecar
  - `to_dict() / from_dict()`: JSON 互換の dict 変換（学習の中間保存用）
- テスト: `tests/test_sparse_weights.py`
  - 基本的な get/set
  - backoff の正しいフォールバック（pair → surface → reading → 0.0）
  - load/dump のラウンドトリップ
  - 空のテーブルでの挙動

### S2: 特徴量拡張（src/beme/features.py）

- `FeatureSpec` に `lex.sparse_lexical` を lexical_features に追加した spec を登録
- `FeatureExtractor.__init__` に `sparse_weights: SparseLexicalWeights | None` パラメータを追加
- `lexical_features()` で `lex.sparse_lexical` 特徴を計算:
  - `self.sparse_weights.get(edge.token.reading, edge.token.surface)` を返す
- テスト: `tests/test_features.py` に sparse_lexical 特徴のテストを追加

### S3: 学習器の拡張（src/beme/training.py と train_scorer.py）

- `train_scorer()` に `sparse_weights: SparseLexicalWeights | None` パラメータを追加
- 更新ルール: モデル誤りが発生したとき、`lex.sparse_lexical` のデルタに応じて疎重みテーブルを更新
  - 正解経路の (reading, surface) ペアの重みを増やす（+learning_rate * delta）
  - 予測経路の (reading, surface) ペアの重みを減らす（-learning_rate * delta）
  - 更新は 3 レベルすべてに適用（pair, surface, reading）
- `train_scorer.py` に `--sparse-weights` 引数と `--l2-regularization` 引数を追加
- テスト: `tests/test_training.py` に疎重み学習のテストを追加

### S4: spec 登録と ablation 設定

- `features.py` に以下を追加:
  - `SPARSE_V1_A8_SPEC`: A8 + `lex.sparse_lexical`（28 + 1パラメータ、ただし疎重みテーブルは別）
  - `SPARSE_V1_C0_SPEC`: A8 + corpus 3 + sparse_lexical（28 + 3 + 1 パラメータ）
- `_SPEC_REGISTRY` に登録
- 初期 weights JSON を作成: `data/weights/ablations/decomposed_v1-no_script-surface_len+sparse_v1.json`

### S5: 実験実行（ws の GPU が必要）

- 条件:
  | spec | 説明 | パラメータ |
  |---|---|---|
  | A8 | 現行ベース | 28 |
  | C0 (F2) | A8 + corpus 3 | 31 |
  | S0 | A8 + sparse | 28 + 疎テーブル |
  | S1 | C0 + sparse | 31 + 疎テーブル |
- 学習: d1 train/dev/test, epochs 5, lr 0.1, seed 9/17/33
- 評価: d1 test, AJIMEE, Mozc
- 比較: S0 vs C0 で「疎重み vs 表参照（F2）」の優劣を判定

### S6: レポート

- 既存の `c0-ajimee-deepdive-2026-09-04.html` と同じ形式でレポート
- F2 との比較を重視

## 見積もり工数

| タスク | 工数 | 備考 |
|--------|------|------|
| S1: SparseLexicalWeights クラス | 2h | 新規ファイル。I/O 形式の設計が要点 |
| S2: 特徴量拡張 | 1h | features.py の変更は小さい |
| S3: 学習器拡張 | 2h | training.py の更新ロジックが要点 |
| S4: spec 登録 | 0.5h | 定数追加のみ |
| S5: 実験実行 | 2h | ws で 12 run。GPU 時間 |
| S6: レポート | 1h | HTML レポート |
| **合計** | **8.5h** | |

## 実装上の注意

- Weights クラスは変更しない（後方互換）
- 疎重みテーブルは学習時に更新可能な mutable オブジェクトとして持つ
- `LinearScorer.invalidate()` は疎重みの変更時にも呼ばれる（`_edge_cache` をクリア）
- コーパス頻度表との相互作用: 疎重みは F2 と独立に動作する。両方使うことで相乗効果が期待できる
- テストは既存の pytest 667 passed を壊さないこと