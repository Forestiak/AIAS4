# Brute-Force Resampled Shape Suite

This folder tests Fixed-length resampled CPT segment shape representations.

- Model count: 20
- Clustering model: GMM only
- Cluster selection: fixed 20 clusters for fair comparison with the previous baseline
- Spatial MRF: intentionally disabled for every generated config
- Results location: `results.csv`, `results.json`, and `errors.csv` inside this folder
- Registry visibility: these `bf_sample_*` feature sets and models are also present in the normal `feature_sets/` and `models/bruteforce_sample/` registries, so they are available through the existing UI like ordinary models

## Usage

Dry-run:

```bash
python bruteforce_sample/run_bruteforce.py --dry-run
```

Run a subset:

```bash
python bruteforce_sample/run_bruteforce.py --limit 5
```

Run one model:

```bash
python bruteforce_sample/run_bruteforce.py --model-id bf_sample_raw3_l16_d0_pca20
```

Run all:

```bash
python bruteforce_sample/run_bruteforce.py
```
