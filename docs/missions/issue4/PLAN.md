# Issue #4 フェーズ1（コスト学習 PoC） 実装プラン

## 1. 目的とスコープ

BeME の変換精度向上（特に CER 0.34 問題の要因である常用語の誤変換・誤分割の改善）を目的とし、コーパスから学習した単語頻度を用いてコストを調整する PoC（フェーズ1）を実施する。

具体的には、Wikipedia 日本語サブセット（約 100MB）から単語の出現頻度を集計し、「する」「私」「ある」などの高頻度語（トップ 1,000〜5,000 語）の変換コストを減算（スムージング）する。これにより、常用語がより上位にランクされ、自然な文節分割が選ばれるかを既存の評価スクリプトで検証する。

## 2. 実装アプローチ

### 2.1 データ取得
- **取得元**: Hugging Face の `wikipedia` データセット（言語 `ja`）、または Wikimedia の CirrusSearch ダンプなどを利用し、先頭から約 100MB 程度のプレーンテキストを抽出する。
- **ライセンス**: Wikipedia のテキストは CC BY-SA 4.0 で提供されている。本プロジェクトではデータ本体を再配布・コミットせず、スクリプト実行時に自動ダウンロード・抽出する形をとる。
- **出力**: `data/corpus.txt` （gitignore に追加）。

### 2.2 頻度集計パイプライン
- **解析器**: `fugashi` などの MeCab バインディングと `ipadic` を利用。
- **処理**:
  1. `data/corpus.txt` を読み込み、形態素解析を行う。
  2. 出力から `(reading, surface, pos)` を抽出する。
     - `reading`: カタカナ表記をひらがなに変換（`src/beme/ipadic.py` の `katakana_to_hiragana` 相当を利用）。
     - `pos`: IPADIC の品詞体系を BeME の品詞（`noun`, `verb`, `particle` 等）にマッピング（`map_ipadic_pos` を利用）。
  3. 各タプルの出現回数を `collections.Counter` で集計する。
- **並列化**: `multiprocessing` モジュールを用い、テキストをチャンク分割して並列で解析・集計することで、処理時間を数十秒〜数分に収める。

### 2.3 コスト減算スムージングの設計
- **トップ N 語の選び方**: 集計された頻度の降順にソートし、上位 1,000 〜 5,000 語を対象とする。
- **減算量**: 頻度 $f$ に応じた対数減衰（例: $\Delta C = \alpha \times \log_{10}(f)$）や、定数減算（例: トップ N 語は一律 $-1.0$）から開始し、実験的に調整する。
- **IPADIC コストへの反映方法**:
  - 既存の `data/dictionary.tsv` を読み込む。
  - 対象の N 語に該当するエントリのコストを減算する。
  - BeME の仕様（下限コスト）を下回らないようにクリッピングする（例: 最低コスト `0.1` などの設定）。

### 2.4 TSV 出力と BeME への適用方法
- コスト減算後のエントリ群を、元のフォーマットに従って新しい `dictionary.tsv` として書き出す。
- （あるいは安全のため `data/dictionary_learned.tsv` として出力し、手動で置換または `dictionary_manual.tsv` のような上書きパッチとして扱う形でもよいが、フェーズ1では手軽に `dictionary.tsv` を直接上書き更新するスクリプトを作成する方針とする。）
- BeME アプリケーション自体は `data/dictionary.tsv` を既定で読み込むため、ファイルが更新されれば自動的に推論（`Converter`）にコストが反映される。

### 2.5 評価方法
- 既存の評価スクリプト `scripts/evaluate.py --suite all` を実行する。
- **確認項目**:
  - AJIMEE（データ本体 CC BY-SA 3.0）, Mozc OSS, handmade の各ケースで NDCG や MRR、Top-1 正解率がコスト減算前（main ブランチ相当）よりも向上しているか。
  - 評価ケースに含まれる「する」「私」などの常用語の分割位置や順位が改善されているかを、ログまたは JSONL の出力結果から確認する。

## 3. 完了条件・検証コマンド

- [ ] Wikipedia サブセットの取得と頻度集計がエラーなく完了すること。
- [ ] `data/dictionary.tsv` の特定高頻度語のコストが減算されて更新されること。
- [ ] `evaluate.py` の結果で、常用語の変換ミスが改善し、全体指標が悪化せず向上すること。

**検証コマンドの流れ**:
```bash
# 依存追加（必要な場合）
uv add --dev fugashi ipadic

# 1. コーパス取得
python scripts/fetch_corpus.py

# 2. 頻度集計とコスト更新
python scripts/apply_cost_learning.py

# 3. 評価の実行
python scripts/evaluate.py --suite all
```

## 4. 実装のタスク分割（Luna 向け）

Luna (サブエージェント) に 1 タスクずつ指示を出して進められるよう、以下の3タスクに分割する。

1. **コーパス取得スクリプトの実装**
   - 指示: `scripts/fetch_corpus.py` を作成し、Hugging Face datasets などから Wikipedia 日本語テキストを約 100MB 取得して `data/corpus.txt` に保存せよ。また `.gitignore` に追記せよ。
2. **頻度集計とコスト減算スクリプトの実装**
   - 指示: `scripts/apply_cost_learning.py` を作成し、`fugashi` 等で `data/corpus.txt` を並列解析して頻度集計を行え。上位 5,000 語のコストを減算するロジックを書き、既存の `data/dictionary.tsv` を更新せよ。
3. **評価の実行とパラメータチューニング**
   - 指示: `python scripts/evaluate.py --suite all` を実行し、ベースラインと比較せよ。常用語の改善が見られるよう、対象語彙数（1,000〜5,000）や減算係数を調整して最良の設定を見つけよ。
