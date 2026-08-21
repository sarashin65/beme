# Task 3: 評価の実行とパラメータチューニング（issue #4 フェーズ1）

## 指示

`docs/missions/issue4/PLAN.md` の Task 3 を実施してください。Task 1（fetch_corpus.py・100MB 取得済み `data/corpus.txt`）・Task 2（apply_cost_learning.py・fugashi 動作確認済み）は完了しています。

## やってほしいこと

1. `scripts/apply_cost_learning.py` で本番コーパス（data/corpus.txt・100MB）に対するコスト学習を実行
   - 既定はトップ 5,000 語・対数減衰（α はスクリプト既定値から開始）
   - 更新辞書は `/tmp/` ではなく **worktree 内の一時ファイル**（例: `data/dictionary_learned.tsv`）に出力して評価に使う（`data/dictionary.tsv` は直接上書きしない・元に戻せるようにする）
2. `scripts/evaluate.py --suite all`（または個別に ajimee / mozc / handmade）で、学習前（data/dictionary.tsv）と学習後（data/dictionary_learned.tsv）を比較
   - 辞書を切り替えて評価する方法は `scripts/evaluate.py --help` で確認（--dictionary フラグ等があれば使用）
3. パラメータ調整: 対象語彙数（1,000〜5,000）や減算係数 α を変えて、**AJIMEE の CER が最も改善する設定**を探す（2〜3 パターン程度・過学習に注意しつつ）
4. 結果を `docs/issue4-phase1-report.md` にまとめる:
   - 学習前後の評価比較表（CER・完全一致率・Mozc pass 数・handmade 維持）
   - 採用したパラメータ（対象語彙数・α）
   - 「する」「私」等の常用語の変化の具体例（辞書 diff から）

## 完了条件

- 学習後辞書での評価結果がレポートにまとまっている
- ベースライン（学習前）との比較が明確
- パラメータ調整の結果（何が最良か）が書かれている

## 制約

- git commit / push はしない
- `data/dictionary.tsv` を壊さない（学習後辞書は別ファイルに）
- 評価の実行時間に注意（100MB コーパスの解析は時間がかかる・並列化 workers を活用）
- 実装・実行の自己確認結果を報告に含める
