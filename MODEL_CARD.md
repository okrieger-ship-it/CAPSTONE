# Model Card — BBO Capstone Optimisation Approach

> Framework: Mitchell et al. (2019), *Model Cards for Model Reporting*

---

## 1. Overview

| Field | Detail |
|-------|--------|
| **Name** | BBO Capstone Optimiser |
| **Version** | v2.0 (Optimised — Modules 17–21) |
| **Type** | Bayesian Optimisation ensemble (GP + SVM + MC-Dropout NN) |
| **Task** | Sequential black-box function maximisation under strict query budget |
| **Developer** | Student, Emeritus / MIT Professional Education (2025) |
| **Language / Stack** | Python 3, scikit-learn, PyTorch, SciPy, NumPy |

---

## 2. Intended Use

**Primary task:**
Identify the maximum of an unknown, unanalysable scalar function f: [0,1]^d → ℝ using the minimum number of function evaluations. Suitable for settings where each evaluation is expensive, slow or irreversible.

**Target users:**
- Researchers studying surrogate-model-based optimisation
- Students learning Bayesian Optimisation under budget constraints
- Practitioners with limited evaluation budgets (≤ 20 queries per problem)

**Appropriate use cases:**
- Low-to-moderate dimensionality (2D–8D) black-box optimisation
- Problems where the function is deterministic and smooth enough for GP interpolation
- Educational benchmarking of acquisition function strategies

**Use cases to avoid:**
- Functions with known discontinuities or extreme non-stationarity (the GP smoothness assumption breaks down)
- Stochastic oracles where repeated queries return different values
- High-dimensional problems (d > 10) without dimensionality reduction — sample efficiency collapses
- Safety-critical or high-stakes real-world deployment without independent validation

---

## 3. Details — Strategy Across Ten Rounds

### Architecture

The optimiser uses a **three-surrogate ensemble**:

| Surrogate | Role | Uncertainty source |
|-----------|------|-------------------|
| Gaussian Process (Matérn ν=2.5) | Primary surrogate; drives EI | Bayesian posterior (exact) |
| SVM (RBF kernel) | Candidate filter; defines "high-performing" region | P(high region) |
| MC-Dropout MLP (2 × 32 ReLU) | Secondary regression surrogate | 200 stochastic forward passes |

### Pipeline (per round)

1. **Data preparation:** load all accumulated (x, y) pairs; apply log-transform for F1 (output range ~10⁻⁷⁹ to ~1); apply StandardScaler for SVM/NN inputs
2. **GP fitting:** Matérn-5/2 ARD kernel + WhiteKernel; 25 L-BFGS-B restarts for kernel hyperparameter optimisation; `normalize_y=True`
3. **Candidate generation:** Sobol quasi-random sequences (10k–150k points depending on dimensionality)
4. **Acquisition:** Expected Improvement with adaptive ξ = ξ₀ × 0.85^t (ξ₀ = 0.05 for 2D/3D, 0.1 for 4D/8D); decays toward pure exploitation
5. **Local optimisation:** L-BFGS-B from top 8–15 Sobol candidates
6. **Ensemble comparison:** GP, SVM-constrained, and NN suggestions compared by EI score and pairwise distance
7. **Override logic:** if empirical trend strongly contradicts GP suggestion (confirmed regression, boundary saturation, plateau), the GP is overridden in favour of the observed directional signal

### Evolution across rounds

| Phase | Rounds | Strategy |
|-------|--------|----------|
| Exploration | 1–3 | High ξ; Sobol coverage; UCB tested for some functions |
| Transition | 4–6 | Adaptive ξ decaying; function-specific regimes begin |
| Exploitation | 7–10 | Low ξ (≈ 0.01); return-to-best for plateaued functions; gradient continuation for trending functions |

**Function-specific regimes by Round 10:**
- **F1:** Return to Week 3 peak neighbourhood [0.646, 0.626]; NN surrogate used as tiebreaker
- **F2:** Return-to-best (Week 4 optimum); GP EI near zero
- **F3/F4:** Retreat to last confirmed high point after recent regressions
- **F8:** Tight exploitation around Week 9 best (y = 9.993); x₅ ≈ 1.0 anchored

---

## 4. Performance

### Best observed outputs by function

| Function | Dimensions | Best y observed | Round achieved |
|----------|-----------|----------------|----------------|
| F1       | 2D        | 0.9153         | Week 3         |
| F2       | 2D        | 0.7105         | Week 4         |
| F3       | 3D        | -0.0349        | Initial data   |
| F4       | 4D        | 0.6681         | Week 6         |
| F5       | 4D        | 8662.48        | Week 5         |
| F6       | 5D        | -0.4028        | Week 9         |
| F7       | 6D        | 2.6718         | Week 2         |
| F8       | 8D        | 9.9933         | Week 9         |

### Metrics used

- **Primary:** Best observed output value per function (maximisation objective)
- **Secondary:** EI score (acquisition function value at suggested point)
- **Diagnostic:** GP-to-NN suggestion distance (consensus proxy); per-dimension length scales (feature importance); convergence plots (best-so-far over rounds)

### Observations on convergence

- **F8:** Monotonically improving trend; clear directional signal; strong exploitation justified
- **F5:** Rapid early gains (Week 5 peak at 8662.48); boundary-saturating landscape — optimal region near [1,1,1,1]; log-transform critical for GP calibration
- **F7:** Best found at Week 2 (y = 2.672); subsequent rounds remained below that peak; volatile 6D landscape with heavy right-skew requiring log-transform
- **F6:** Steady but slow improvement (all outputs negative, maximising toward zero); Week 9 best = -0.4028; x₅ consistently near 0 in high-performing queries
- **F1:** Highly non-stationary; Week 3 spike followed by regression; GP unreliable away from peak
- **F3/F4:** Volatile; regressions in later rounds suggest the GP landscape model is unreliable at this data density

---

## 5. Assumptions and Limitations

### Key assumptions

| Assumption | Impact if violated |
|-----------|-------------------|
| **Smoothness (Matérn kernel):** function varies continuously at a consistent rate | GP misses sharp peaks; EI collapses; F1 partially violates this |
| **Determinism:** same input always returns same output | Noise-chasing in exploitation phase; calibration errors |
| **Stationarity:** function structure is consistent across the input space | GP over-smooths heterogeneous landscapes; performance degrades in high-dimensional functions |
| **Global coverage from Sobol:** 150k candidates adequately represent [0,1]^8 | Acquisition maximum may be missed; local optima returned instead |

### Limitations

- **Query budget vs. dimensionality mismatch:** With 19–49 total observations across 2D–8D functions, the GP operates below the data density required for reliable inference in most of the space. Industry benchmarks suggest 50+ points for low-dimensional problems and several hundred for 8D. The optimiser's theoretical guarantees do not hold at this scale.
- **Sampling bias:** Queries cluster around early-discovered promising regions. Large portions of the input space — especially corners and edges distant from Round 1–3 results — are never sampled.
- **No noise modelling:** The WhiteKernel accounts for observation noise but the pipeline assumes the oracle is deterministic. If the portal introduces stochasticity, exploitation queries may be chasing noise artefacts.
- **Manual overrides are undocumented inline:** Override rationale is described in notebook markdown cells but not in a machine-readable format, reducing automated reproducibility.

---

## 6. Ethical Considerations

**Transparency and reproducibility:**
All pipeline code, hyperparameter choices, weekly query histories and surrogate comparison outputs are documented in public Jupyter notebooks. The fixed random seed (`random_state=42`, `SEED=42`) ensures that any reviewer running the notebooks on the same data will obtain the same GP suggestions and NN outputs. Override decisions are explained in markdown cells alongside the code.

**What additional documentation would improve clarity:**
A structured override log (function, round, GP suggestion, chosen query, override rationale) would make the human-in-the-loop decisions fully auditable. Currently, this information is embedded in narrative markdown and must be read sequentially rather than queried programmatically.

**Limitations of transparency:**
The portal outputs (true function values) cannot be reconstructed without access to the capstone system, meaning a reviewer can follow the methodology but cannot independently verify that the output values are correct.

**Responsible use:**
This optimisation approach makes irreversible, budget-constrained decisions — an analogy to real-world ML deployment where each experiment costs time, compute or money. The habit of cross-checking three surrogates before submitting, and of never blindly following a single model, reflects responsible AI practice: models are tools, not oracles, and human judgment remains essential when their assumptions are violated.

**No fairness or demographic concerns apply** — the dataset contains abstract mathematical inputs with no human subjects.

---

## 7. Distribution

| Item | Detail |
|------|--------|
| Repository | Public GitHub repository (see README for link) |
| License | MIT License |
| Model code | Jupyter notebooks (Python 3) in `/capstone` directory |
| Data files | `.npy` arrays in `/function_X/` subdirectories |
| Datasheet | `DATASHEET.md` (this repository) |
| Model card | `MODEL_CARD.md` (this repository) |

---

*Model card created: June 2025. Framework: Mitchell et al., "Model Cards for Model Reporting," ACM FAccT 2019.*
