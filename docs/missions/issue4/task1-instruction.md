# Task 1: コーパス取得スクリプトの実装（issue #4 フェーズ1）

## 指示

`docs/missions/issue4/PLAN.md` の Task 1 を実装してください。プランは既に読める状態です。

- `scripts/fetch_corpus.py` を作成し、Hugging Face datasets などから Wikipedia 日本語テキストを約 100MB 取得して `data/corpus.txt` に保存する
- データ本体はリポジトリにコミットしない（`.gitignore` に `data/corpus.txt` を追記）
- ライセンス: Wikipedia テキストは CC BY-SA 4.0。データ本体の再配布はしない方針（スクリプト実行時に自動取得する形）

## 完了条件

- `scripts/fetch_corpus.py` が動作し、`data/corpus.txt`（約100MB・プレーンテキスト）を生成できる
- `.gitignore` に `data/corpus.txt` が追記されている
- 実行に必要な依存（HuggingFace datasets 等）が `pyproject.toml` / `uv.lock` に正しく追加されている

## 制約

- git commit / push はしない（作業ツリーに変更を残す）
- 対象外ファイルを変更しない
- 実装後は自己確認（スクリプトのヘルプ出力・依存のインストール確認）して報告
- 実際に 100MB フルダウンロードまで実行する必要はないが、小規模で動作確認（スモークテスト）はすること
