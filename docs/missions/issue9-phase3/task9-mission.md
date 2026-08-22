# Task 9: Phase 3 レポート

## 目的

Task 8 が生成した `artifacts/issue9-phase3/summary.json` と各条件の成果物から、`docs/missions/issue9-phase3/phase3-report.md` を書く。あわせて README に Phase 3 の実行手順を追記する。

**Phase 2 レポートの最大の反省は「集計指標だけで回帰を説明し、原因が仮説どまりになったこと」である**（issue #9 Phase 3 注意点 7）。本レポートは、その再発を構造で防ぐ。

最初に `docs/missions/issue9-phase3/PLAN.md` の §8（レポート構成）・§9（スコープ判断）・§11（Phase 4 引き継ぎ）、`docs/missions/issue9-phase2/phase2-report.md`、`docs/missions/issue9/phase1-report.md`、`README.md` を読むこと。

## 依存

Task 8 が完了していること。

## 対象ファイル

- 新規: `docs/missions/issue9-phase3/phase3-report.md`
- 変更: `README.md`（Phase 3 のセクション追記のみ）

コード・辞書・重み・artifacts は変更しない。

## レポート構成（この順で書く）

### 1. Scope and Reproducibility

実行機・git commit・全コマンドの逐語・入力 SHA-256（辞書・初期重み・d1 split）。**測定していない値を書かない。** 供給された記録に無い値は「記録に含まれていない」と書く（Phase 2 レポートが SHA-256 についてそうしたのと同じ扱い）。

### 2. Ablation Matrix Results

PLAN §4 の 12 条件について、d1 test error（3 seed の中央値と最小–最大幅）・既存 suite・McNemar p 値・判定ラベルの表。

- 判定ラベルは `significant_worse` / `significant_better` / `indeterminate` の 3 値のみ。
- **`indeterminate` を「差が無い」と書き換えないこと。** 「d1 test 2,000 件では判別できない」と書く。
- 既存 suite（AJIMEE 200 / Mozc 498 / handmade 14）の値は載せるが、**採否判断には使っていない**旨を明記する。handmade は 1 件の変動で Top-1 が 7.1 pt 動く。

### 3. Learned Balance Table

設計ノート `docs/beme-learned-path-scorer.html` §8 は、学習の結果として次のバランスを獲得することを**予測している**。

```text
IPADIC生起コスト      弱めに利用
matrix接続コスト      0.2倍程度で利用
カタカナ表層         少し不利
読み条件付き頻度     強く利用
overlay              出所特徴として利用
```

この予測に対する**実測の答え合わせ**を表で書く。予測が外れた項目は外れたと書く。

重みは `lex.ipadic_cost_mapped` を 1 とした正規化値で示し、次の注記を必ず添える。

> `lex.script.*` と `trans.*` は任意の完全経路について `Σ lex.script.* == |P| == Σ trans.*` を満たす（Mozc 200 読み × K-best 5 の 1,000 経路で照合・不一致 0）。特徴空間に厳密な線形従属が 1 本あるため、「トークン数ペナルティ」方向の重みは個別に同定できない。個々の重み値を特徴の重要度として読んではならない。採否判断は §2 のエラー率差で行っている。

`lex.reading_len` が全条件で 0 のままである事実（`Σ len(token.reading) == len(reading)` により gold−pred が常に 0 で構造的に学習されない）も記載する。

### 4. Oracle Recall@k

3 つの定義（lattice coverage / path recall@k / surface recall@k）を分けた表。k と `beam_width` を必ず両方明記する。

- **AJIMEE を対象外とした理由**を書く（`convert_sentence` が文分割するため単一読みの recall と定義が合わない）。
- `unique_surface_count` の分布（`beam_width=64` の経路が何本のユニーク surface に潰れるか）を載せる。**Phase 4 の `rerank_k` はこの値から決まる。**
- 学習が「1-best を上げただけ」なのか「上位集合ごと良くしたのか」を、baseline と学習済みの recall@k 曲線の比較として明示的に述べる。

### 5. Stratified Results

読み長 / 固有名詞 / 数値読み / span_tokens の 4 軸。

**`span_tokens` と contrast pair は「MeCab 変換品質」の代理指標であり、変換品質そのものの層別ではない**旨を明記する。真の品質の根拠は d1 の 200 件抽出監査（`label_accuracy=0.995`）しか無い。

件数 30 未満の層は率とともに `low support` と明記する。

### 6. Case-Level Before/After

**このセクションが Phase 2 の反省への直接の回答である。**

- handmade 全 14 件と Mozc 全 498 件の順位表（Top-1 が変わらなかったケースも省略しない。下位候補の変動も残す）。
- Top-1 が変わったケースについて、特徴ごとの寄与分解 `w[f] * Δf` を絶対値の降順で。「経路が変わった効果」と「重みが変わった効果」を分けて示す。
- **`hashi-no`（`はしの` が `橋の` → `橋野`）を寄与分解で決着させる。** Phase 2 は「Wikipedia 由来の頻度・固有名詞バイアスと `の` の境界選好の変化」と推測したが、これは仮説であって因果測定ではなかった。今回は数値で説明する。分解しても説明がつかない場合は、**つかないと書く**。仮説を測定として書かない。
- d1 test の代表誤り（`fixed` / `regressed` 各 20 件・レコード id 昇順）。

### 7. Cost-Scale Recalibration

confidence 分位点表（baseline / learned × handmade / AJIMEE / Mozc / d1 test）、再較正した `cascade_threshold`、Viterbi ブレンドの初期倍率。

次の 2 つの事実を明記する。

- **`needs_rerank` は計算されるだけで、実際のリランク発火をゲートしていない**（`converter.py:174` が計算、`:176` が `context` と reranker の有無だけで判定、参照は `evaluation.py:255` の `rerank_rate` 集計のみ）。したがって `cascade_threshold` の再較正は統計量の測り直しであって精度改善ではない。Phase 3 ではこれを**直していない**（変換挙動を変えるとアブレーション結果の解釈が汚れるため）。
- **`viterbi_weight` の実較正は Phase 4 へ送った**（実リランカが要り、それ自体が Phase 4 のチェックリスト項目。実モデルは miniPC にのみキャッシュされている）。Phase 3 が出したのは初期倍率のみ。

### 8. Leakage Audit

d1 manifest の除外主張に対する独立検証の結果。

- 文書リーク / 読み重複 / 語彙重複を**分けて**報告する。読み重複と語彙重複はリークではなく分布の重なりである。
- manifest の「Mozc は読みと assertion のみで原文を持たないため語彙共有は文書リークとみなさない」という判断の追検証結果。
- `*.jsonl` が無いために検証できなかった項目があれば、**検証できなかったと書く**。

### 9. Negative and Null Results

**独立したセクションとして必ず置く。**

- 効果が出なかった条件、`indeterminate` だった条件。
- 実施しなかったこと（PLAN §3.3 の F2 = Wikipedia コーパス頻度版、`viterbi_weight` の実較正など）とその理由。
- Task 2 / 3 が未完で skip した条件があればそれも。

### 10. Limitations and Phase 4 Handoff

PLAN §11 の 9 項目を、実測値で埋めて書く。特に:

- **リランカへ渡す局所モデルの確定**（X0 を第一候補、A0 を対照。判定が `indeterminate` なら**単純な A0 を選ぶ** — 複雑な方を選ぶ理由がない）。
- **surface recall@k が頭打ちなら、リランカを強くしても Top-1 は上がらない。** Phase 4 の最初の判断はこれ。
- **重みの配布形態の結論**（PLAN §9）。Phase 1・2 から 2 フェーズ持ち越されている `pyproject.toml` の package-data 問題について、Phase 3 の結論は「学習済み重みは外部パス必須の設計と明示し、`legacy_v1.json` と `decomposed_v1.json` の 2 つだけを package-data に入れて `default_scorer()` を動かす」。Phase 3 で 12 条件 × 3 seed の重みが増えたことがこの結論の根拠になる。Phase 4 で起票する。
- **位置依存特徴の警告**: matrix cost は位置非依存なので Phase 3 では `LatticePath` を再設計していない。文頭・文末距離のような位置依存特徴を Phase 4 で入れるなら、`LinearScorer._edge_cache` が `edge.token` キーである前提（`scorer.py:40-52`）が壊れる。**例外も警告も出ない。** キャッシュキーと `LatticePath` の位置保持を同時に再設計すること。

## README への追記

`README.md` の Phase 2 セクションに続けて、Phase 3 のセクションを追加する。

- アブレーション条件の一覧と spec id。
- `scripts/build_ablation_weights.py` / `build_matrix_table.py` / `oracle_recall.py` / `diagnose_cases.py` / `audit_leakage.py` / `measure_cost_scale.py` / `run_ablations.py` の使い方（実際に動く逐語のコマンド）。
- `data/ipadic_matrix_v1.bin` の生成には IPADIC の `matrix.def` が必要で、`ipadic` pip パッケージには含まれない（`matrix.bin` のみ）という注意。
- 学習は GPU マシンで行う理由（d1 が repo 外）。

## 完了条件

- レポートの 10 セクションがすべて埋まっている。
- **測定していない値を 1 つも書いていない。** 記録に無いものは「記録に無い」と書く。
- 仮説と測定が区別して書かれている。
- `hashi-no` が寄与分解で説明されている（説明できない場合はそう書かれている）。
- `indeterminate` が「差が無い」に書き換えられていない。
- README の追記コマンドが実際に動く（`--help` で確認する）。

## 検証コマンド

```bash
# README に書いたコマンドが実在することの確認
uv run python scripts/build_ablation_weights.py --help
uv run python scripts/build_matrix_table.py --help
uv run python scripts/oracle_recall.py --help
uv run python scripts/diagnose_cases.py --help
uv run python scripts/audit_leakage.py --help
uv run python scripts/measure_cost_scale.py --help
uv run python scripts/run_ablations.py --help

uv run pytest -q
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- コード・辞書・重み・`artifacts/` を変更しない。
- **数値を捏造しない・推測で埋めない。** 供給された成果物に無い値は「記録に含まれていない」と書く。
- 仮説を測定として書かない。因果を主張するなら分解を示す。
- 集計指標だけで回帰を説明しない（Phase 2 の反省）。
- Negative / Null Results を省略しない。
- 上記コマンドの実結果を報告する。
