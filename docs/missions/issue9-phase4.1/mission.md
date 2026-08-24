# Mission: BeME issue #17 Phase 4.1（script 特徴の部分除去アブレーション）

## 材料（必ず読む）
- **正本: `gh issue view 17`** — 本文に事前登録の採用ルール・A7/A8 条件定義・受け入れ条件が全て記載されている。方針はこれに従え（勝手に変えるな）
- 参考実装: `scripts/build_ablation_weights.py`（Phase 3 の A6 = `decomposed_v1-no_shape` が一番近い参考）・`scripts/run_ablations.py`・`tests/test_feature_registry.py`・`tests/test_run_ablations.py`
- Phase 3/4 のレポート書き方規約: `docs/missions/issue9-phase4/phase4-report.md` 冒頭の記述スタイル（判別不能は明記・単一 seed の p 値でラベル付けしない・artifact SHA-256 記録・実行機名併記）

## 作業環境
- worktree: `~/work/beme-4p1`（branch `feat/issue9-phase4.1`・origin/main @ aabd057 起点・クリーン）
- GPU マシン: `ssh ws`（sarashin@192.168.0.28・Arch・Ryzen 9 5900X 24T・RTX 3060 Ti 8GB）
  - d1 データ: `~/beme-d1/repo/data/d1/`（train/dev/test jsonl・gitignore・GPU 側にのみ存在）
  - Phase 3 学習成果物: `~/beme-p3/artifacts/issue9-phase3/`（A0 seed9/17/33 の weights.json 等）

## やること
1. `gh issue view 17` で本文を読む
2. **実装**: `scripts/build_ablation_weights.py` に A7 / A8 の 2 条件を追加し、初期重み JSON を生成する
   - A7 = `decomposed_v1-no_script`（`lex.script.*` 4個のみ除去・`lex.surface_len`・`lex.proper_noun` は保持）
   - A8 = `decomposed_v1-no_script-surface_len`（`lex.script.*` + `lex.surface_len` 除去・`lex.proper_noun` は保持）
   - 既存条件（A6 = `decomposed_v1-no_shape` 等）の命名規則・生成パターンに従う
3. **テスト追加**: 宣言特徴の除去が実際に効くことを検証するテスト（既存の feature registry / run_ablations テストパターンに合わせる）
4. **機械的確認**: `uv run pytest` 全件 green・`git diff --check`
5. **GPU 実行**: `ssh ws` で Phase 4.1 用の新規 clone（`~/beme-p41`）を作る
   - 転送は **git bundle**（プライベート repo・gh 認証が GPU 側に無いため）: `git bundle create /tmp/beme-4p1.bundle feat/issue9-phase4.1` → scp → 先方で `git clone --branch feat/issue9-phase4.1 /tmp/beme-4p1.bundle ~/beme-p41`
   - `data/d1` は symlink（`ln -s ~/beme-d1/repo/data/d1 data/d1`）かコピーで供給
   - `uv sync`（GPU 側の uv 導入済み確認・無ければ curl install）
   - 学習: `run_ablations.py --conditions decomposed_v1-no_script,decomposed_v1-no_script-surface_len,decomposed_v1-no_shape --seeds 9,17,33`（9 run・Phase 3 と同形式。A6 も再実行して同じ日・同じ環境での比較にする）
   - 学習が長引く場合は **デタッチ起動**（`nohup setsid ... < /dev/null &`・PPID=1 確認・ログ確認）で回し、完了を待たずに進められる作業を進める。完了確認は `while kill -0 <PID>; do sleep 30; done` で
6. **集計**: `run_ablations.py --summarize-only` で McNemar 判定（A6/A7/A8 × A0 比較・3 seed ペアリング）
7. **レポート**: `docs/missions/issue9-phase4.1/phase4.1-report.md` を書く（issue #17 の受け入れ条件への回答・事前登録ルール 6 条への判定・3 点比較表・人名 3 件 + proper-noun 層 244 件のケース確認・4 指標回帰・artifact SHA-256）
8. **git commit はローカルに**（push しない・PR は作らない — Hermes が機械的ゲート確認後にやる）
9. 完了報告: 実装内容・テスト件数・学習結果（数値表）・採用ルールの判定・レポートパス

## 制約
- 設計判断は issue #17 の事前登録ルールに従う。追加の設計変更が必要と判断したら勝手に実装せず、その旨を報告に含める
- `src/beme/lattice.py`・`converter.py` など探索コアは触らない（アブレーションは重み・特徴レジストリの変更のみ）
- git commit はローカルのみ。push・PR 作成・gh issue 操作は禁止
- 学習済み重み・artifacts は gitignore 対象。レポートから SHA-256 で参照する形にする
- 日本語で報告（数値・結論を先頭に）
