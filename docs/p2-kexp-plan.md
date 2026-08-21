# プラン: k 拡大ゲート実験（P2 の事前検証）

## 1. 実験設計
**目的**: 候補生成の幅 (k) を広げた際に、実リランカが oracle recall の増分をどの程度 Top-1 へ昇格できるか（実リランカ効果）と、それに伴うレイテンシの増加を測定する。
**データセット**: AJIMEE 200 件 (`data/ajimee_bench_jwtd_v2_v1.json`)
**比較条件**:
- Baseline: P0 実装後 (k=5) + reranker なし
- 提案手法: k=10, 20, 64 + `--reranker cross-encoder` (`japanese-reranker-base-v2`)
**評価指標**:
- Top-1 完全一致率 (Exact Match Rate)
- 平均 CER (Character Error Rate)
- リランカの推論レイテンシ

## 2. 前回（P0 前の 13% 天井）との比較方法
P0 前は oracle recall@5 が 13.0%、Top-1 完全一致が 6.0% であり、P0 実施後もそれぞれ 14.0%, 7.0% と「天井」が存在していました。
しかし、事前調査において k=64 に拡大することで oracle recall が 31.5% に達することが判明しています。
本実験では、この k 拡大で得られた「31.5% の oracle 空間」の中から、実リランカが実際に何件を Top-1 正解として抽出できるか（実質的な Top-1 完全一致率）を測定し、P0 前後 (6.0%, 7.0%) からの「真の改善幅」を直接比較して評価します。

## 3. single-span 候補の扱い
事前の oracle recall 調査により、k=64 に広げて増加した正解 35件のうち、24件は single-span であることが分かっています。
現在のシステムでは文レベルの K-best 合成が不完全（P2 の全面設計が必要）ですが、single-span であればその影響を受けません。
そのため、本実験では「文 K-best 合成の全面再設計」を待つことなく、k 拡大と実リランカの組み合わせだけで、少なくともこれら 24件に対してどこまで実効的な恩恵（Top-1 への昇格）があるかを検証可能です。結果の集計時は single-span と multi-span を層別集計し、リランカ単体のポテンシャルと文合成再設計の必要性を切り分けます。

## 4. 評価スクリプトの置き場所
**判断**: worktree 内の `scripts/` に配置する（または既存の `scripts/evaluate.py` を改修する）。
**理由**: 今回は一時的な診断ではなく、P2 実装の投資判断を目的とした専用の実験ブランチ `exp/k-expansion` (worktree: `~/work/beme-kexp`) での作業です。P1 のように `~/.cache/beme-diag/` に置く使い捨てスクリプトではなく、実験コードもブランチ内でバージョン管理することで、他メンバーとの共有や再現性の確保が容易になるためです。

## 5. 成果物レポートの構成
実験完了後、以下の構成で `docs/p2-kexp-report.md` を作成します。
1. **結論**: 実リランカによる CER および Top-1 の改善幅、レイテンシ。P2（文 K-best 合成再設計）へ進むべきかの最終判断。
2. **指標サマリ**: k=5, 10, 20, 64 における CER、Top-1 完全一致率、平均処理時間の比較表。
3. **span数別分析**: single-span と multi-span におけるリランカ効果の差異（文 K-best 未実装の影響評価）。
4. **定性分析**: リランカによって正解に昇格したケース、および悪化したケースの分析。

## 6. 完了条件・検証コマンド
**完了条件**:
- k=5, 10, 20, 64 の全条件で AJIMEE 200件の評価が完了していること。
- 評価結果が成果物レポートとして markdown でまとめられていること。

**検証コマンド（想定）**:
```sh
cd ~/work/beme-kexp
.venv/bin/python scripts/evaluate.py --suite ajimee --k 10 --reranker cross-encoder --json > results_k10.json
.venv/bin/python scripts/evaluate.py --suite ajimee --k 20 --reranker cross-encoder --json > results_k20.json
.venv/bin/python scripts/evaluate.py --suite ajimee --k 64 --reranker cross-encoder --json > results_k64.json
```

## 7. 実装のタスク分割
Luna へ 1 回の指示で依頼できる単位に分割したタスク一覧です。

- **Task 1: 評価スクリプトの k パラメータ対応**
  `scripts/evaluate.py` や関連する推論エンジンを改修し、候補幅 `k` をコマンドラインから受け取り、`--reranker cross-encoder` への入力候補数を動的に変更できるようにする。
- **Task 2: オフライン実験の実行と結果保存**
  k=5, 10, 20, 64 の設定で AJIMEE の評価を実行し、それぞれの推論結果およびメトリクスを JSON 形式で worktree 内に保存する。
- **Task 3: 結果の集計とレポート作成**
  保存した JSON を読み込み、span 数（single-span / multi-span）ごとの層別集計を行うスクリプトを作成。その結果を基に `docs/p2-kexp-report.md` を執筆する。
