# BBO Capstone Project — Black-Box Optimisation

**Emeritus / MIT Professional Education | Professional Certificate in ML & AI**
**Modules 17–24 | April – June 2026**

---

## Overview

This repository contains the full implementation of a thirteen-round Black-Box Optimisation (BBO) capstone project. Eight unknown scalar functions of varying dimensionality (2D to 8D) were optimised sequentially using Bayesian Optimisation, one query per function per week, over thirteen weeks.

The optimisation pipeline combines:
- **Gaussian Process surrogate** (Matern ν=2.5 kernel, ARD length scales)
- **SVM classifier** (candidate region filtering)
- **MC-Dropout Neural Network** (secondary regression surrogate)
- **Expected Improvement acquisition** with adaptive ξ decay
- **Sobol quasi-random sampling** + L-BFGS-B local optimisation

---

## What This Project Is (For Everyone)

This project tackled a challenge that appears throughout science, engineering and business: how do you find the best setting for something when you cannot see inside it and each testing costs time or money? Over thirteen weeks, we ran an intelligent search across eight unknown mathematical functions, using a technique called Bayesian Optimisation. Think of it like a smart trial-and-error process that learns from each result and uses that learning to make a better guess next time rather than searching randomly. The system successfully identified strong-performing regions for all eight functions, with the most rewarding discoveries coming from recognising patterns early and adjusting the search strategy accordingly for each function's unique behaviour.

## Repository Structure

```
CAPSTONE/
│
├── README.md                        ← This file
├── DATASHEET.md                     ← Dataset documentation (Mini-lesson 21.1 framework)
├── MODEL_CARD.md                    ← Model documentation (Mini-lesson 21.2 framework)
│
├── Module24_F1_Optimized_v2.ipynb   ← F1 (2D) — full BO pipeline, 10 rounds
├── Module24_F2_Optimized_v2.ipynb   ← F2 (2D)
├── Module24_F3_Optimized_v2.ipynb   ← F3 (3D)
├── Module24_F4_Optimized_v2.ipynb   ← F4 (4D)
├── Module24_F5_Optimized_v2.ipynb   ← F5
├── Module24_F6_Optimized_v2.ipynb   ← F6
├── Module24_F7_Optimized_v2.ipynb   ← F7
├── Module24_F8_Optimized_v2.ipynb   ← F8 (8D)
│
├── function_1/
│   ├── initial_inputs.npy
│   └── initial_outputs.npy
├── function_2/ ...
└── function_8/
    ├── initial_inputs.npy
    └── initial_outputs.npy
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [DATASHEET.md](./DATASHEET.md) | Dataset documentation following the Gebru et al. (2021) datasheet framework. Covers motivation, composition, collection process, preprocessing, uses, distribution and maintenance. |
| [MODEL_CARD.md](./MODEL_CARD.md) | Model documentation following the Mitchell et al. (2019) model card framework. Covers approach overview, intended use, ten-round strategy, performance results, assumptions, limitations and ethical considerations. |

---

## Key Results

| Function | Dimensions | Best output | Round |
|----------|-----------|-------------|-------|
| F1       | 2D        | 1.5578      | Week 12 |
| F2       | 2D        | 0.7105      | Week 4 |
| F3       | 3D        | -0.0356     | Week 5 |
| F4       | 4D        | 0.6681      | Week 6 |
| F5       | 4D        | 8662.48     | Week 5 |
| F6       | 5D        | -0.2011     | Week 10 |
| F7       | 6D        | 2.6718      | Week 2 |
| F8       | 8D        | 9.9933      | Week 9 |

---

## How to Run

1. Clone this repository
2. Install dependencies: `pip install numpy scipy scikit-learn torch matplotlib`
3. Open any `Module21_FX_Optimized_v2.ipynb` in Jupyter
4. Update `DATA_DIR` to point to the correct `function_X/` folder
5. Run all cells — the notebook will reproduce all 10 rounds of BO and generate visualisations

> **Reproducibility:** All notebooks use `random_state=42` and `SEED=42`. Running on the same data will produce identical GP and NN suggestions.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, data loading |
| `scipy` | Sobol sampling, L-BFGS-B optimisation, EI/UCB computation |
| `scikit-learn` | Gaussian Process, SVM, StandardScaler |
| `torch` | MC-Dropout neural network surrogate |
| `matplotlib` | GP heatmaps, slice plots, convergence curves |

---

## License

MIT License. See [LICENSE](./LICENSE) for details.

---

*Created June 2026 as part of the Emeritus / MIT Professional Education BBO Capstone.*
