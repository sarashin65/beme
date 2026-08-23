# Task 7: Phase 4 レポート

## 目的

`docs/missions/issue9-phase4/phase4-report.md` を書き、issue #9 へ Phase 4 完了コメントを起草する（**投稿はユーザーが判断する。worker は起草まで**）。

Phase 2 の反省は「集計値だけで回帰を説明した」、Phase 3 の反省は「統計の比較対象を間違えた」「代表 seed 1 本でラベルを決めた」だった。**Phase 4 のレポートは構造でこれを防ぐ。**

最初に `docs/missions/issue9-phase4/PLAN.md` の §8（レポート構成）と §5（分岐）、`docs/missions/issue9-phase3/phase3-report.md`（体裁の正本）、Task 4・5・6 の全 artifact を読むこと。

## 依存

Task 5。Task 6 が実施された分岐ならそれも。

## 対象ファイル

- 新規: `docs/missions/issue9-phase4/phase4-report.md`
- 変更: `README.md`（リランカ関連の実測値が更新されるなら）
- 新規: issue #9 へのコメント草案（ファイルとして置く。例 `docs/missions/issue9-phase4/issue-comment-draft.md`）

## レポート構成（PLAN §8）

1. **Scope and Reproducibility** — 実行機・commit・全コマンドの逐語・入力 SHA-256・cross-encoder のモデル名とリビジョン
2. **Top-1 Definition Unification** — Task 1 の結果。A3 の 208 件乖離が解消したことの実測。同点が発生しない 11 条件が不変であることの実測
3. **Separating Oracle Recall from Top-1** — 3 段の表（上限 / 伸びしろ / 実績）。**リランキングが `surface_recall@k`（k ≥ min(rerank_k, 候補数)）を不変にすることの実測アサーション結果**
4. **Grid Selection on dev** — C1 のグリッド全面 28 点。選択規則の逐語と選択点。**高原かスパイクか**
5. **Local / Reranker-only / Combined** — C2 の 3 条件比較。primary（contrast 128）・secondary（全体 2,000）・pair 単位（64 対）
6. **Seed Stability** — C3。seed 9/17/33 で向きが一致するか
7. **What the Reranker Broke** — 失ったケース（b）の**全件一覧**と分類
8. **gold_rank Distribution** — リランク前後
9. **Latency** — miniPC 実測。**すべての数値に実行機名を併記**
10. **Cascade Gate** — Task 6 実施時のみ。未実施なら「実施しなかった。理由は §5 の分岐 X」と書く
11. **Leakage Audit** — Task 4 の結果。`unverifiable` が残った項目の名指し
12. **Negative and Null Results** — 効かなかった条件、判別できなかったもの、実施しなかったこととその理由
13. **Limitations and Handoff** — PLAN §5 の分岐に応じた次の作業。起票する別 issue

## 書き方の規約（省略しないこと）

- **判別できないものは「判別できない」と書く。** 「差が無い」と書かない。標本サイズを添える。
- **PLAN §2.4 の 40 件パイロットを結果として引用しない。** 引用する場合は「計画立案時の探索的測定であり、選択バイアスがあり有意でない（p = 0.115）」と明記し、結論の根拠にしない。
- **集計値だけで回帰を説明しない。** 失ったケースは全件列挙する。
- **レイテンシは実行機とセットで書く。** GPU マシンの「スコアリング時間」と miniPC の「IME レイテンシ」を混ぜない。
- **「リランカ単独」の定義を明記する。** IME は候補生成を省けないので「候補生成なし」は条件にしていない。候補生成は局所モデルと同一で、順位付けだけが relevance のみ、という定義である。
- **単一 seed の p 値で有意ラベルを付けない。** Phase 3 と同じ規則（全 seed 一致）。
- 数値は artifact を正本とし、レポート本文には要約表を置く。artifact のパスと SHA-256 を書く。

## issue コメント草案に含めること

Phase 2・3 のコメントと同じ体裁で:

- Phase 4 のチェックリスト 3 項目それぞれに対する結果
- 主要な数値の表（3 条件比較）
- **設計上の決定**（2 段分離・Top-1 統一・「リランカ単独」の定義・primary metric を contrast pair にした理由）
- **レビューで見つけた実バグ**（あれば。Phase 1〜3 のコメントはすべてこの節を持っている）
- **Phase 5 / 次の作業への注意点**（PLAN §5 の分岐に応じて）
- テスト数の推移（Phase 3 完了時点は 585）

## 起票する別 issue（PLAN §3.8）

レポートの Handoff 節に、次を「起票する」と書く（起票そのものはユーザーの判断）。

1. **重みの配布形態** — Phase 1 から 3 フェーズ持ち越し。`pyproject.toml` に package-data 指定が無く、wheel インストールで `default_scorer()` が動かない。Phase 3 の結論は「学習済み重みは外部パス必須、`legacy_v1.json` と `decomposed_v1.json` のみ package-data」
2. **評価セットの拡張** — Phase 3 が「2,000 件では局所モデルの改善を判別できない」、Phase 4 が（分岐によっては）「128 件では判別できない」と示した場合、次に必要なのは標本の拡張である
3. 分岐に応じて: X0（matrix + freq）でのリランカ再測定 / AJIMEE への文脈供給 / #2（頻度バイアス）・#3（誤りの分類）への引き継ぎ

## 完了条件

- `phase4-report.md` が PLAN §8 の 13 節をすべて持つ（実施しなかった節は「実施しなかった。理由は …」と書く）。
- issue チェックリストの 3 項目それぞれに対する結果が明示されている。
- 失ったケースが全件列挙されている。
- すべてのレイテンシ数値に実行機名が付いている。
- issue コメント草案がある。
- `uv run pytest -q` が全通し（レポート作成でコードを触っていないことの確認）。

## 検証コマンド

```bash
uv run pytest -q

# レポート中の再現コマンドが実際に動くことの確認（1 つ以上を実行して出力を貼る）
uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv --weights data/weights/decomposed_v1.json --json

# artifact の SHA-256 がレポートの記載と一致すること
sha256sum <レポートが参照する artifact>
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。
- issue へのコメント投稿をしない（草案まで）。
- 実測していない値をレポートに書かない。推測は「推測である」と明記する。
- Phase 3 レポートの数値を引き写すときは、引き写しであることと出典（`phase3-report.md` の節番号）を書く。
