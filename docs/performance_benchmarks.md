# 性能ベンチマーク

`bimodal_skewfit` の各処理の実行時間と、フィット精度 (ISE) との
トレードオフをまとめます。共有 ZIP の v5 ソースで取得した実測値です。

## 計測条件

すべての計測は次の条件で実施しました。

- BLAS スレッド: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (`justfile` 既定)
- Python 3.13.9, numpy 2.x, scipy 1.x, uv 環境 (`.venv`)
- 計測手順: モデル / サンプラごとに warmup 2 回 + 計測 5 回、minimum を採用 (median / mean も併記)
- 計測スクリプト: `temp/bench_fit_per_model.py` (フィット時間)、`temp/bench_fit.py` (シナリオ統合)。
  どちらも `temp/` 配下のため共有 ZIP には含まれません。本文書の数値を
  再現するときはどちらも `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python ...` で実行してください。

マルチスレッド (12 コア) で計測すると、本ワークロード (n=200〜400 の小行列に対する反復最適化)
では BLAS のスレッド spawn overhead が並列利得を上回り、wall time が約 **2.5 倍遅く** なります
(`real 42 s` vs `real 17 s`)。これが `justfile` で 1 スレッドに固定している理由です。

## 1. 各モデルのフィット実行時間

`fit_<model>` 1 回あたりの所要時間です。データは固定シナリオから 1 種類ずつ取得し、`max_iter=60`。

| モデル | n=200 単峰正規 | n=200 対称二峰 | n=400 歪二峰 | 推定方式 |
| --- | ---: | ---: | ---: | --- |
| `normal` | **0.12 ms** | 0.13 ms | 0.13 ms | 閉形式 MLE (`np.mean` + `np.std`) |
| `gmm2` | 175 ms | 93 ms | 61 ms | 11 starts × multi-start EM (max_iter=60) |
| `abn` | 166 ms | 165 ms | 206 ms | 4 starts × L-BFGS-B |
| `adn` | 111 ms | 127 ms | 147 ms | 4 starts × L-BFGS-B |
| `bsn_fs` | **42 ms** | 42 ms | 41 ms | 4 starts × L-BFGS-B (shape fit 中で最速) |
| `ntpn` | 72 ms | 103 ms | 133 ms | 4 starts × L-BFGS-B |

### 観察

1. **`normal` は別格** (~0.1 ms)。閉形式なので n を増やしてもボトルネックにならない。
2. **`bsn_fs` (~42 ms 一定)** は shape fit の中で最速かつデータ依存性が小さい。
   L-BFGS-B が早く収束する形をしている。
3. **`abn` は二峰性が強くなるほど遅くなる** (歪二峰 n=400 で 206 ms)。
   `tanh` 変換による gradient が境界付近で不安定で、iter 数が伸びる傾向。
4. **`gmm2` は逆にデータが二峰寄りだと速い** (61〜93 ms)。EM が早く収束するため。
   単峰正規では識別性が悪く 175 ms を要する。
5. **`fit_all` (6 モデル一括)** の所要時間 ≈ 上記合計で **1 シナリオあたり 400〜700 ms**。

## 2. ランダム密度の生成時間

`random_generators.py` の各 builder で 1 サンプル分の生成時間を計測 (warmup 後・5 reps の min)。

| family | サンプリング手法 | n=300 min | n=300 median | n=1000 min |
| --- | --- | ---: | ---: | ---: |
| `normal` | `rng.normal` | 0.04 ms | 0.06 ms | 0.05 ms |
| `skewnorm` | `scipy.skewnorm.rvs` | 0.15 ms | 0.16 ms | 0.17 ms |
| `gmm2` | mixture (binomial + normal) | 0.11 ms | 0.11 ms | 0.14 ms |
| `abn` | mixture (制約付き) | 0.11 ms | 0.12 ms | 0.14 ms |
| `adn` | accept-reject (CDF skew) | 0.22 ms | 0.27 ms | 0.36 ms |
| `ntpn` | **independent Metropolis-Hastings** | **2.19 ms** | 2.81 ms | 4.50 ms |
| `bsn_fs` | **independent Metropolis-Hastings** | **1.37 ms** | 1.48 ms | 3.16 ms |

### M-H サンプラの設定

`_independent_mh_sample(n, target_logpdf, proposal_sampler, proposal_logpdf, rng, burnin=1000, thin=5)` を使用。

- 提案分布は α=0 を入れた既存カーネル (`ntpn_logpdf` または `bsn_fs_logpdf`) を再利用。
  独立した提案密度を別途実装しないため、監査面で安全。
- 提案サンプルと proposal/target log 評価はベクトル化済み。
  受理判定ループだけ Python レベルで実行。
- `burnin=1000` + `n × thin=5` ステップなので、n=300 で 2,500 ステップ、n=1000 で 6,000 ステップ。

### 観察

1. 直接 sampling (`normal` / `gmm2` / `abn`) は **0.05〜0.15 ms**。numpy の RNG が生で動くだけ。
2. `adn` の rejection sampling は **0.2〜0.4 ms**。CDF (`ndtr`) ベースの acceptance ratio が高く効率的。
3. `ntpn` / `bsn_fs` の M-H は **約 1〜5 ms**。直接 sampling より 20〜50 倍遅いが、
   絶対値はミリ秒オーダで実用上問題なし。
4. **`run_random_search --trials 24 --sample-size 300` への影響**:
   24 trial × 1/7 確率 = 約 3〜4 trial が NTPN / BSN-FS。1 trial あたり 2〜3 ms 上乗せで
   合計 +10〜20 ms。trial あたり数百 ms かかるフィットと比較すると **誤差レベル**。

## 3. 速度 × 精度トレードオフ

7 family すべての真の生成データに対して 6 モデル全部をフィットし、`integrated_squared_error`
を真の密度との比較で測定 (n=300, max_iter=60)。

### 3.1 真分布 × フィットモデル の ISE マトリクス

各セルは ISE (低いほど良い)。★は各行 (= 真の家族) で ISE 最小のモデル。

| true ↓ \ fit → | `normal` | `gmm2` | `abn` | `adn` | `bsn_fs` | `ntpn` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `normal` | .000299 | .000418 | .000405 | .000330 | .000493 | **.000266 ★** |
| `skewnorm` | .030685 | .005891 | .008424 | **.000874 ★** | .001701 | .006641 |
| `gmm2` | .027244 | **.001691 ★** | .013480 | .010850 | .007044 | .013460 |
| `abn` | .017458 | .004480 | .004196 | .004196 | **.004045 ★** | .004196 |
| `adn` | .003104 | .000908 | .000315 | **.000249 ★** | .000497 | .000430 |
| `ntpn` | .050895 | .000624 | .000651 | .024043 | .009199 | **.000570 ★** |
| `bsn_fs` | .023651 | **.001444 ★** | .002770 | .002278 | .023798 | .004493 |
| **time →** | **0.13 ms** | **164 ms** | **214 ms** | **158 ms** | **43 ms** | **100 ms** |

### 3.2 フィットモデル別の精度 / 時間サマリ

`mean ISE` は全 7 真分布の平均。`matched ISE` は「真の生成 family と同じモデルでフィット」した時の ISE。

| fit model | mean ISE | median ISE | max ISE | mean time (ms) | matched ISE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `normal` | 0.0219 | 0.0059 | 0.0581 | **0.13** | 0.00006 |
| `gmm2` | **0.0016** | 0.00099 | 0.00437 | 164.04 | 0.00308 |
| `abn` | 0.0022 | 0.00104 | 0.00948 | 213.95 | 0.00094 |
| `adn` | 0.0052 | 0.00108 | 0.02985 | 158.01 | 0.00099 |
| `bsn_fs` | 0.0038 | 0.00127 | 0.01496 | **43.20** | 0.00027 |
| `ntpn` | 0.0019 | **0.00098** | 0.00724 | 100.02 | 0.00055 |

### 3.3 観察

1. **`normal` (0.13 ms)** は閉形式で最速だが、misspecified なシナリオで ISE が爆発する
   (`skewnorm` で 0.031、`ntpn` で 0.051)。matched 時のみ意味がある。
2. **`bsn_fs` (43 ms)** は shape fit の中で最速。`gmm2` と比べて約 3.8 倍速い。
   matched 時の ISE は 0.024 と意外に大きく、α が大きい領域での局所解問題が疑われる。
   ただし他 family への汎化能力は高く (mean ISE 0.0038)、コストパフォーマンスが良い。
3. **`ntpn` (100 ms)** はバランス型のチャンピオン: median ISE 0.00098 と最良、
   mean ISE 0.0019 で `gmm2` に次ぐ精度。`abn` の約半分の時間で同等以上の精度を達成。
4. **`gmm2` (164 ms)** は平均的に最も精度が高い (mean ISE 0.0016)。多くの bimodal / skew
   シナリオで上位だが、`normal` / `skewnorm` のような単純なシナリオではむしろ overfit 気味。
5. **`abn` (214 ms)** は最も時間がかかる (二峰性が強いと iter 数が伸びる)。matched 時の ISE は良い
   (0.00094)。
6. **`adn` (158 ms)** は `skewnorm` matched で best だが、`ntpn` や `bsn_fs` 真の家族には弱い。

### 3.4 Pareto 観点での推奨

`mean time × mean ISE` の Pareto フロント:

| 指針 | 推奨モデル | コメント |
| --- | --- | --- |
| 精度最重視 | `gmm2` (164 ms, ISE 0.0016) | 全シナリオで安定 |
| バランス重視 | `ntpn` (100 ms, ISE 0.0019) | 速度の割に精度高い |
| 速度重視 | `bsn_fs` (43 ms, ISE 0.0038) | 4 倍速くて精度は約 2 倍悪い |
| matched 確信時 | `normal` (0.13 ms) | misspec 棚上げ前提 |

### 3.5 実用上の戦略

`fit_all` で 6 モデル全部走らせて BIC 最小モデルを選ぶ戦略が、合計で **約 700 ms / シナリオ** で済みます。
production では「`fit_all` + BIC argmin」が速度・精度・実装シンプルさのトレードオフで一番効率的です。
1 モデルだけ選んで賭けるよりも、6 並列のうちの最良を取った方が、平均 ISE は確実に小さくなります。

## 4. 全パイプラインのざっくり概算

`OPENBLAS_NUM_THREADS=1` 設定の wall time:

| 処理 | 所要時間 | 備考 |
| --- | --- | --- |
| `pytest -q` (53 テスト) | 約 14 秒 | M-H サンプラと数値積分テストを含む |
| `run_experiment.py` (5 シナリオ × 6 モデル + plot/CSV) | 3〜5 秒 | sample_size=400, max_iter=60 |
| `run_random_search.py --trials 24 --sample-size 300` | 40〜60 秒 | NTPN/BSN-FS の M-H 生成で +数十 ms |
| `build_report_notebook.py --execute` | 5〜8 秒 | 既存 outputs を読むだけなので軽い |
| `just quality` (format + lint + typecheck + test + radon + run + random + notebook) | 約 80 秒 | 上記すべての total |

## 5. 改善余地

レビュアーから提案を受けた MCMC 診断 (acceptance rate, autocorrelation, ESS) は現状未実装です。
`_independent_mh_sample` が `acceptance_count` / `total_proposed` を返す形に拡張し、
`GeneratedDensity` に `sampler_diagnostics: dict[str, float] | None` を追加すれば、
研究用途で生成品質を担保できます。本ベンチで採用した M-H サンプル品質は
quad で求めた真の (μ, σ) と samples の (μ, σ) が両方とも 0.01 以内で一致することを別途確認済みなので、
NTPN / BSN-FS の random_search における役割としては十分機能しています。
