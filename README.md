# bimodal-skewfit

`bimodal-skewfit` は、二峰性・単峰性・歪度を持つ一次元分布に対して候補分布を最尤推定し、AIC/BIC、グリッド上のモード数、既知生成密度との ISE で比較するための `uv` 前提の Python 実装です。論文 4 本 (ABN, ADN, BSN-FS, NTPN) の密度カーネルを統一記法で実装し、合成データに対する自動回帰テスト 53 件で論文式との対応を担保しています。

## ドキュメント

| ファイル | 内容 |
| --- | --- |
| [`docs/theory_derivations_bimodal_skewfit.md`](docs/theory_derivations_bimodal_skewfit.md) | 全モデルの数式・推定対象・最適化目的・更新式・特殊ケース・サンプリング手順・評価指標を統一記法で導出 (§0〜§14) |
| [`docs/performance_benchmarks.md`](docs/performance_benchmarks.md) | フィット時間、ランダム生成時間 (M-H 含む)、速度 × 精度 (ISE) のトレードオフ計測 |
| [`docs/implementation_inventory.md`](docs/implementation_inventory.md) | パッケージ全ファイルのレイヤー (kernel/driver/orchestration/io) 一覧、品質ゲート結果、検証コマンド |
| [`docs/workdoc_May04-2026_bimodal_skew_fit.md`](docs/workdoc_May04-2026_bimodal_skew_fit.md) | 初版実装時の作業ログ |

## 対象モデル

実装済みは次の 6 モデル + ベースライン skewnorm (生成専用) です。各 model 名のリンクは `docs/theory_derivations_bimodal_skewfit.md` 内の対応する節に飛びます。

| model | 概要 | 論文 / 理論文書 |
| --- | --- | --- |
| `normal` | 単一正規分布の閉形式 MLE | [§2 Normal baseline](docs/theory_derivations_bimodal_skewfit.md#2-normal-baseline) |
| `gmm2` | K=2 混合ガウス、multi-start EM | [§3 GMM K=2](docs/theory_derivations_bimodal_skewfit.md#3-二成分-gaussian-mixture-k2) |
| `abn` | 対称位置・等分散・不等重みの制約付き 2 成分 GMM | [§4 ABN](docs/theory_derivations_bimodal_skewfit.md#4-abn-asymmetric-bimodal-normal) — Mathematics 14 (5), 2026 |
| `adn` | 等重み 2 正規混合核に CDF 型歪化を掛ける | [§5 ADN](docs/theory_derivations_bimodal_skewfit.md#5-adn-asymmetric-double-normal) — Symmetry 17 (6), 2025 |
| `bsn_fs` | Fernández--Steel 型歪化 + 二峰化摂動 | [§6 BSN-FS](docs/theory_derivations_bimodal_skewfit.md#6-bsn-fs-fernándezsteel-型-bimodal-skew-normal) — arXiv:1512.03341 |
| `ntpn` | New Two-Piece Normal 型の二峰・歪分布 | [§7 NTPN](docs/theory_derivations_bimodal_skewfit.md#7-ntpn-new-two-piece-normal-extension) — AIMS Mathematics 11 (1), 2026 |

形状モデル (`abn` / `adn` / `bsn_fs` / `ntpn`) は L-BFGS-B 多項共通ドライバで推定します。詳細は [§8 形状モデル共通の直接最尤推定](docs/theory_derivations_bimodal_skewfit.md#8-形状モデル共通の直接最尤推定) を参照してください。

## インストール

`uv pip` または `pip` でインストールできます。スコープ別に extras を選んでください。

```bash
# (a) ライブラリとして fit 関数だけ使う (core: numpy, scipy のみ)
uv pip install .

# (b) 固定シナリオ実験 / ランダム探索 / 可視化も使う
uv pip install '.[report]'

# (c) Notebook 生成 (build_report_notebook.py) も使う
uv pip install '.[notebook]'

# (d) 開発 (テスト・lint・型チェック・ベンチ)
uv sync --extra dev
```

`bimodal-skewfit` / `bimodal-skewfit-random` の console script、および
`scripts/run_experiment.py` / `scripts/run_random_search.py` は **pandas と matplotlib に依存**するため、`[report]` 以上の extras が必要です。最小ライブラリ用途 `(a)` だけでは `from bimodal_skewfit.experiment import ...` 等が `ModuleNotFoundError` になります (これは Python の標準挙動で、依存分離の意図された結果です)。

公開 API は `__init__.py` から再エクスポートしている次の要素に固定しています:

```python
from bimodal_skewfit import FitResult, fit_model, fit_all, list_models, __version__
```

`fit_model` / `fit_all` の戻り値 (`FitResult`) には `.logpdf(x)` / `.pdf(x)` メソッドがあるため、密度評価のために `distributions` 内部関数を直接呼ぶ必要はありません。

## 基本実行

固定シナリオを実行します。

```bash
uv run python scripts/run_experiment.py --output-dir outputs --sample-size 400 --seed 20260503 --max-iter 60
```

ランダム生成・品質探索を実行します。

```bash
uv run python scripts/run_random_search.py --output-dir outputs --trials 24 --sample-size 300 --seed 20260504 --max-iter 60
```

Notebook を生成・実行します。

```bash
uv run python scripts/build_report_notebook.py --execute
```

matplotlib のバックエンドはこのパッケージ側からは変更しません。Jupyter Lab 等で `%matplotlib widget` を選んでいる場合はその設定がそのまま使われます。`Figure.savefig` は backend に非依存で動くので、`run_experiment.py` などのスクリプトはどの backend でも問題なく PNG を書き出せます。

BLAS スレッド数は `justfile` の各 recipe で `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` に固定しています。本プロジェクトは n=200〜400 程度の小サンプルに対する最適化を多数回反復するワークロードのため、BLAS の thread spawn overhead が並列利得を超え、1 スレッドの方が 2〜3 倍速くなります (実測値は [`docs/performance_benchmarks.md`](docs/performance_benchmarks.md) 参照)。`just test` / `just run` / `just random` / `just quality` 経由なら自動で適用されます。`uv run pytest` 等を直接呼ぶ場合は同じ env を手動で export してください。

## 性能特性

`OPENBLAS_NUM_THREADS=1` 設定の wall time の目安です (詳細表は [`docs/performance_benchmarks.md`](docs/performance_benchmarks.md))。

| 項目 | 所要時間 |
| --- | ---: |
| `fit_normal` (n=200) | 0.12 ms |
| `fit_bsn_fs` (n=200, 最速の shape fit) | 42 ms |
| `fit_gmm2` / `fit_abn` (n=200) | 90〜220 ms |
| `fit_all` 全 6 モデル (1 シナリオ) | 400〜700 ms |
| `pytest -q` (53 テスト) | 約 14 秒 |
| `run_experiment.py` (5 シナリオ × 6 モデル) | 3〜5 秒 |
| `run_random_search.py --trials 24 --sample-size 300` | 40〜60 秒 |

精度面では、全 7 真分布平均で `gmm2` が mean ISE 0.0016 (164 ms) と最良、`ntpn` が 0.0019 (100 ms) でバランス型のチャンピオン、`bsn_fs` が 0.0038 (43 ms) で最速の shape fit という構図です。`fit_all` + BIC argmin の組み合わせが速度・精度・実装シンプルさのトレードオフで最も実用的です。

## 品質チェック / テスト

```bash
uv run ruff format .
uv run ruff check .
uv run ty check src tests scripts --ignore unresolved-import --python .venv
uv run pytest -q
uv run radon cc src scripts -s -a
uv run radon mi src scripts -s
```

`justfile` には上記をまとめたタスクを定義しています。

```bash
just quality
```

テストは 5 ファイル 53 件で構成されます。

| ファイル | テスト数 | 役割 |
| --- | ---: | --- |
| `tests/test_distributions.py` | 5 | 各密度の正規化と特殊ケース (積分テスト) |
| `tests/test_fit_smoke.py` | 2 | 主要フィッタの smoke と明確二峰での GMM 優位 |
| `tests/test_random_search.py` | 2 | ランダム探索の出力 schema |
| `tests/test_regression.py` | 30 | 数値スナップショット (固定シナリオ best-by-BIC、random_search best quality) |
| `tests/test_theoretical_correspondence.py` | 14 | 論文式の特殊ケース (ABN/ADN/NTPN/BSN-FS の縮退点) と family coverage、`fit_gmm2` best-run convergence |

## 出力ファイル

| ファイル | 内容 |
| --- | --- |
| `outputs/fit_summary.csv` | 固定シナリオ別・モデル別の AIC/BIC/推定値/モード数 |
| `outputs/*_fit_overlay.png` | 固定シナリオのヒストグラムと上位フィット密度 |
| `outputs/random_search_summary.csv` | ランダム生成データに対する全モデルの品質スコア |
| `outputs/random_quality_report.md` | ランダム探索における最高品質フィットとファミリ別ベスト |
| `outputs/random_search_best_fit.png` | ランダム探索で最高品質だった試行の可視化 |
| `outputs/validation_log.md` | 品質ゲートの実検証ログ |
| `examples/fitting_report.ipynb` | 結果を確認する実行済み Notebook |

## モード数集計の解釈

`mode_count_grid` 列および各レポート中のモード数は `evaluate.count_density_modes` で算出した **grid-based visual diagnostic** です。理論上のモード数判定ではなく、評価グリッド上の局所最大を `scipy.signal.find_peaks` で抽出し、最大密度に対する相対 prominence (既定 0.02) で雑音を除いた値です。弱分離の二峰や shoulder 形状は 1 峰として分類され得ます。厳密なモード数解析を行いたい場合は密度関数を直接解析してください。理論的背景と prominence 閾値の選び方は [§9 評価指標とモード診断](docs/theory_derivations_bimodal_skewfit.md#9-評価指標とモード診断) を参照。

## アーキテクチャ方針

数値核は状態を持たない関数として `src/bimodal_skewfit/distributions.py` に集約しています。モデル名 → 仕様の写像は `registry.py` に分離 (Rust の `enum Distribution` に直接対応する形)、フィッタは `gmm_fit.py`・`shape_fit.py`・`fit.py` に分割し、推定結果は `results.py` の `FitResult` で統一しています。各モジュール docstring の冒頭に `(layer: kernel | driver | orchestration | io)` を明記しており、Rust 等他言語へ移植する場合は kernel 層を最初に純関数として写し、driver 層の最適化アルゴリズムを差し替える順で進められます。ファイル別の詳細は [`docs/implementation_inventory.md`](docs/implementation_inventory.md) を参照。

暗黙的 fallback は入れていません。未知モデル名、無効スケール、最適化候補なしなどは明示的に例外または `-inf` 対数密度として扱います。

## 論文準拠性と監査上の注意

本実装は各論文の密度関数の主要部分を、フィット比較のために実装したものです。次は **含まれません**:

- ADN 論文の回帰モデル全体 / 観測情報行列 / 標準誤差推定の全再現
- ABN 論文の CDF・分位点・全モーメント・回帰診断の全再現
- AIMS NTPN 論文の信頼性関数・エントロピー・寿命解析指標
- BSN-FS 論文の Student-t 基底やベイズ推定部分

一方で、密度・対数尤度・主要特殊ケース・合成データ生成・最尤推定・モデル比較は実験目的に必要な範囲で self-contained に定義されています。NTPN と BSN-FS のサンプリングは independent Metropolis-Hastings を採用しており、受理率・自己相関・有効サンプルサイズの厳密管理は未実装です。詳細な監査ガイドラインは [§13 監査上の注意点](docs/theory_derivations_bimodal_skewfit.md#13-監査上の注意点) を参照してください。
