# Task 3: Averaged structured perceptron 中核

## 目的

Phase 1 の scorer/特徴差分と Task 1–2 を接続し、coverage error と model error を分離する決定的な averaged structured perceptron を実装する。

最初に `README.md`、`docs/missions/issue9-phase2/PLAN.md` §1.2・§2.3、`src/beme/features.py`、`scorer.py`、`weights.py`、Task 1–2 の実装を読むこと。

## 依存

Task 1 と Task 2 が完了していること。

## 対象ファイル

- 新規: `src/beme/training.py`
- 新規: `tests/test_training.py`
- 必要な公開 export に限り: `src/beme/__init__.py`

CLI、lattice/converter、データ本体、既存 weight JSON は変更しない。

## 変更内容

1. split 評価結果、epoch 結果、最終学習結果を immutable data class で定義する。
2. prediction は現在 scorer の exact 1-best、gold は `constrained_best_path` を使う。
3. `gold is None` は coverage error。covered だが `pred.surface != target_surface` は model error。分母を PLAN §2.3 どおり分離する。
4. 学習更新は **cost 最小化形** `w += eta * (Phi_pred - Phi_gold)`。既存 `feature_delta(gold,pred)` を使うなら符号を反転する。
5. `path_features` は必ず `path_features(path, extractor, scorer.weights)` と呼ぶ。
6. update ごとに新しい immutable `Weights` を構築し `scorer.invalidate(new_weights)` する。
7. averaged weights は各 covered example 処理後の current weights を平均する。正解例を含み、coverage error は averaging step に含めない。
8. epoch snapshot の averaged model を dev 評価し、`(end_to_end_error_rate, model_error_rate, epoch)` で選択する。
9. test 評価は API 上も選択後にだけ呼べる構造にし、訓練中に参照しない。
10. metadata に updates/steps/epochs/rate/seed と上位層から渡された hash/manifest を保存できるようにする。

## 必須テスト

- 1 update 後に `C(gold)-C(pred)` が小さくなる（符号 gate）。
- correct example は update 0 だが averaging step に入る。
- coverage error は全体件数へ入るが update/averaging step へ入らない。
- coverage/model/end-to-end の各分母とゼロ件境界。
- latent gold を毎回 current model で取り直す。
- scorer cache が update 後に古い cost を返さない。
- 手計算できる 2–3 step の averaged weights と一致。
- 同じ seed/入力で同一 weights と epoch 選択。
- dev tie は早い epoch。test を model selection に使わない。
- negative learned weights を拒否しない。

## 完了条件

- synthetic dataset で model error が減る方向へ学習する。
- coverage/model error が別々に取得できる。
- averaged `Weights` は既存 `dump/load` を往復できる。

## 検証コマンド

```bash
uv run pytest -q tests/test_training.py
uv run pytest -q tests/test_oracle.py tests/test_features.py tests/test_weights.py
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- 対象外ファイルを変更しない。
- issue の式を説明なく cost weight に直適用しない。PLAN の score/cost 対応をコードコメントとテストに残す。
- K-best/後方 DP/Converter の public interface を変えない。
- 位置依存特徴、高 cardinality 特徴、正則化、margin、早期停止など Phase 3 の機能を追加しない。
- 関連テストと検証結果を報告する。
