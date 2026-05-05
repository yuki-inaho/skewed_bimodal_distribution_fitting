# 実装物一覧 (v4)

## 1. パッケージ構成

| パス | 役割 | レイヤー |
| --- | --- | --- |
| `pyproject.toml` | `uv` 用プロジェクト定義、依存関係、Ruff/pytest/ty 設定 | 設定 |
| `justfile` | セットアップ、品質チェック、固定実験、ランダム探索、Notebook 生成タスク | 設定 |
| `src/bimodal_skewfit/distributions.py` | 正規、GMM2、ABN、ADN、BSN-FS、NTPN の対数密度カーネル | kernel |
| `src/bimodal_skewfit/registry.py` | `DistributionSpec` と `DISTRIBUTIONS` 名前-spec 辞書 (Rust の `enum Distribution` 相当) | kernel |
| `src/bimodal_skewfit/results.py` | `FitResult` と尤度・初期値補助関数 | kernel |
| `src/bimodal_skewfit/evaluate.py` | AIC/BIC、grid-based モード数、評価グリッド、ISE | kernel |
| `src/bimodal_skewfit/gmm_fit.py` | 正規 MLE と K=2 混合ガウス multi-start EM | driver |
| `src/bimodal_skewfit/shape_fit.py` | `TransformedModelSpec` 駆動 L-BFGS-B (ABN/ADN/BSN-FS/NTPN) | driver |
| `src/bimodal_skewfit/fit.py` | モデル名によるフィット dispatch と一括フィット API | driver |
| `src/bimodal_skewfit/simulate.py` | 固定シナリオデータ生成 | driver |
| `src/bimodal_skewfit/random_generators.py` | 7 family のランダム分布生成 (NTPN/BSN-FS は M-H サンプラ) | driver |
| `src/bimodal_skewfit/experiment.py` | 固定シナリオの実験オーケストレーション | orchestration |
| `src/bimodal_skewfit/random_search.py` | ランダム探索、品質スコアリング、Markdown レポート | orchestration |
| `src/bimodal_skewfit/plotting.py` | ヒストグラム・密度 overlay 図の生成 (matplotlib Agg 強制) | io |
| `src/bimodal_skewfit/cli.py` | 固定実験 CLI | io |
| `src/bimodal_skewfit/random_cli.py` | ランダム探索 CLI | io |
| `scripts/run_experiment.py` | 固定実験実行ラッパー | io |
| `scripts/run_random_search.py` | ランダム探索実行ラッパー | io |
| `scripts/build_report_notebook.py` | 実行済み Notebook 生成スクリプト | io |
| `tests/test_distributions.py` | 密度の正規化・特殊ケース単体テスト | テスト |
| `tests/test_fit_smoke.py` | 主要フィッタの smoke test と明確二峰 GMM 検証 | テスト |
| `tests/test_random_search.py` | ランダム生成・ランダム探索出力テスト | テスト |
| `tests/test_regression.py` | 数値スナップショット (固定シナリオ best-by-BIC、ランダム探索 best quality) | テスト |
| `tests/test_theoretical_correspondence.py` | ADN/ABN/NTPN/BSN-FS の特殊ケース、family coverage、`fit_gmm2` best-run convergence | テスト |
| `notebooks/fitting_report.ipynb` | 実行済み Notebook | io |
| `outputs/` | CSV、Markdown、PNG の実験成果物 | io |

## 2. 実装済みモデル

| モデル | パラメータ | フィット方法 | ランダム生成 |
| --- | --- | --- | --- |
| `normal` | `loc`, `scale` | 閉形式 MLE | 直接 sampling |
| `gmm2` | `weight`, `mu1`, `sigma1`, `mu2`, `sigma2` | multi-start EM (実 EM、`np.isfinite(prev_ll)` ガードあり) | 直接 sampling |
| `abn` | `loc`, `scale`, `lambda`, `alpha` | 変換パラメータ L-BFGS-B (spec dispatch) | 直接 sampling |
| `adn` | `loc`, `scale`, `sep`, `skew` | 変換パラメータ L-BFGS-B (spec dispatch) | rejection sampling |
| `bsn_fs` | `loc`, `scale`, `alpha`, `gamma` | 変換パラメータ L-BFGS-B (spec dispatch) | independent M-H (proposal=FS skew normal) |
| `ntpn` | `loc`, `scale`, `alpha`, `lambda` | 変換パラメータ L-BFGS-B (spec dispatch) | independent M-H (proposal=TPN) |

ベースラインの skew-normal は random generator にのみ存在 (`skewnorm`) し、専用フィッタは持ちません。フィッタ族の skew は `adn`/`bsn_fs`/`ntpn` がカバーします。

## 3. 配布物と依存スコープ

`pyproject.toml` 上の依存は次のように分離しています。`pip install` 時に必要な extras を指定してください。

| extras | 含まれるパッケージ | 想定ユースケース |
| --- | --- | --- |
| (none, core) | `numpy`, `scipy` | `from bimodal_skewfit import fit_model, fit_all, FitResult` で密度フィッタだけ使う |
| `report` | core + `pandas`, `matplotlib` | `experiment.py` / `random_search.py` / `plotting.py` / CLI を使う |
| `notebook` | report + `nbformat`, `nbclient`, `ipython`, `ipykernel` | `scripts/build_report_notebook.py --execute` を使う |
| `dev` | notebook + `pytest`, `ruff`, `ty`, `radon` | テスト・lint・型チェック・ベンチを含む全機能 |

Hatchling の `[tool.hatch.build.targets.sdist]` で `outputs/`, `notebooks/`, `examples/`, `temp/`, `dist/` を sdist から除外しています。wheel には `src/bimodal_skewfit` のみが含まれます。

ビルド・配布物検証のコマンド:

```bash
just build              # uv build → dist/*.whl と dist/*.tar.gz を生成
just package-smoke      # 隔離環境で wheel と sdist をインストールし core 依存だけで smoke test
just release-check      # quality + build + package-smoke を一括実行
```

## 4. 検証済みコマンド

```bash
uv run ruff format .
uv run ruff check .
uv run pytest -q
uv run ty check src tests scripts --ignore unresolved-import --python .venv
uv run radon cc src scripts -s -a
uv run radon mi src scripts -s
uv run python scripts/run_experiment.py --output-dir outputs --sample-size 400 --seed 20260503 --max-iter 60
uv run python scripts/run_random_search.py --output-dir outputs --trials 24 --sample-size 300 --seed 20260504 --max-iter 60
uv run python scripts/build_report_notebook.py --execute
```

`justfile` の `test` / `run` / `random` / `quality` recipe には BLAS スレッド制限
(`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`) を入れています。
小さい n=200〜400 では BLAS thread spawn overhead が並列利得を超えるため、絞った方が
2〜3 倍速くなります。直接 `uv run pytest` を実行する場合は手動で同じ env を export してください。

## 5. 品質結果

| 観点 | 結果 |
| --- | --- |
| pytest | 54 passed (5 + 2 + 2 + 31 + 14) |
| Ruff format / check | passed (rules: B, C4, E, F, I, RET, SIM, UP, ANN, N, RUF, PERF, PIE, TID, ARG, NPY) |
| ty static analysis | passed |
| radon cyclomatic complexity | average A (1.97)、93 blocks、最大 B (CC=7) |
| radon maintainability index | all files A |

## 6. 設計上の注意

- 密度関数は stateless かつ副作用なし。Rust などへの移植時に kernel 層を最初に純関数として写し、driver 層 (最適化アルゴリズム) を後で差し替える順序で進められます。
- `DistributionSpec.__post_init__` で `num_parameters == len(param_names)` を検証しており、登録ミスを起動時に検出します。
- `shape_fit.py` は `TransformedModelSpec` データ駆動 dispatch (`SHAPE_MODEL_SPECS` 辞書) で 4 ファミリのフィッタを 1 経路に集約しています。新ファミリを増やす際は spec を 1 つ追加するだけです。
- `random_generators.py` の builders は M-H ベースで NTPN / BSN-FS をカバーします。independent M-H の提案分布は α=0 を入れた既存カーネル (`ntpn_logpdf` / `bsn_fs_logpdf`) を再利用しており、提案密度の独立実装を持たない点で監査が容易です。
- `fit_all()` はモデルごとの失敗を暗黙的に無視しません。未知モデル名は明示的に `KeyError` です。
- `fit_gmm2` の `converged` は最良 NLL run の収束フラグです (旧版は `OR over runs` でした)。
- ランダム探索の品質スコアは、既知生成密度との ISE、真のモード数との一致、収束有無、BIC 差分を使う決定的指標です。
- `count_density_modes` はあくまで grid-based visual diagnostic。弱分離の二峰や shoulder 形状は 1 峰として分類され得ます。厳密なモード数判定が必要な場合は密度関数を直接解析してください。
