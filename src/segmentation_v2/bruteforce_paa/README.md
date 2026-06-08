# Brute-Force PAA Suite

This folder tests Piecewise Aggregate Approximation (PAA) segment shape representations.

- Model count: 20
- Clustering model: GMM only
- Cluster selection: fixed 20 clusters for fair comparison with the previous baseline
- Spatial MRF: intentionally disabled for every generated config
- Results location: `results.csv`, `results.json`, and `errors.csv` inside this folder
- Registry visibility: these `bf_paa_*` feature sets and models are also present in the normal `feature_sets/` and `models/bruteforce_paa/` registries, so they are available through the existing UI like ordinary models

## Usage

Dry-run:

```bash
python bruteforce_paa/run_bruteforce.py --dry-run
```

Run a subset:

```bash
python bruteforce_paa/run_bruteforce.py --limit 5
```

Run one model:

```bash
python bruteforce_paa/run_bruteforce.py --model-id bf_paa_raw3_b4_d0
```

Run all:

```bash
python bruteforce_paa/run_bruteforce.py
```
