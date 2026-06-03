# Datasheet — BBO Capstone Project Dataset

> Framework: Gebru et al. (2021), *Datasheets for Datasets*

---

## 1. Motivation

**Why was this dataset created?**
This dataset was created as part of a Professional Certificate capstone project in Black-Box Optimisation (BBO). The goal is to explore and optimise eight unknown, unanalysable functions by querying them sequentially over ten weekly rounds. Each function accepts a vector of inputs in [0, 1]^d and returns a scalar output. Because the functions are black boxes — their internal structure, gradients and landscape are entirely unknown — the dataset supports the study of surrogate-model-based optimisation under strict query budgets.

**What task does it support?**
The dataset supports Bayesian Optimisation research: fitting Gaussian Process (GP) surrogates, training auxiliary classifiers (SVM) and neural network regressors (MC-Dropout MLP), evaluating acquisition functions (Expected Improvement, Upper Confidence Bound), and comparing multi-model ensemble strategies under real-world constraints of one irreversible query per function per week.

**Who created it and why?**
Created by the student as part of Emeritus / MIT Professional Education coursework. No external funding. The dataset is the direct product of the student's weekly query submissions to the capstone portal.

---

## 2. Composition

**What does the dataset contain?**

| Function | Dimensions | Initial points | Weekly additions | Total observations |
|----------|-----------|---------------|-----------------|-------------------|
| F1       | 2D        | 10            | 9               | 19                |
| F2       | 2D        | 10            | 9               | 19                |
| F3       | 3D        | 15            | 9               | 24                |
| F4       | 4D        | 30            | 9               | 39                |
| F5       | 4D        | 20            | 9               | 29                |
| F6       | 5D        | 20            | 9               | 29                |
| F7       | 6D        | 30            | 9               | 39                |
| F8       | 8D        | 40            | 9               | 49                |

**Format:**
- Inputs: `.npy` NumPy arrays of shape `(n, d)`, all values in [0, 1]
- Outputs: `.npy` NumPy arrays of shape `(n,)`, scalar floats
- Files stored per function: `initial_inputs.npy`, `initial_outputs.npy`
- Weekly additions tracked in notebook code cells

**Output ranges:**

| Function | Output range | Notes |
|----------|-------------|-------|
| F1       | ~1e-192 to 0.915 | Extremely sharp peak; 76 orders of magnitude; log-transform applied |
| F2       | -0.07 to 0.71 | Well-scaled, no transform needed |
| F3       | -0.399 to -0.035 | All negative, near-zero best |
| F4       | -32.6 to +0.67 | High variance, volatile |
| F5       | 0.11 to 8662.48 | ~4 orders of magnitude; log-transform applied (all strictly positive) |
| F6       | -2.57 to -0.40 | All negative; maximising toward zero; no transform needed |
| F7       | 0.003 to 2.672 | Right-skewed; 18/30 initial values below 0.1; log-transform applied |
| F8       | 5.59 to 9.99 | Well-scaled, monotone improvement |

**Gaps and missing data:**
- No observations exist for most of the input space, particularly in higher-dimensional functions. For F8 (8D), the sampled volume represents a negligible fraction of the unit hypercube.
- Queries cluster heavily around regions identified as promising in Rounds 1–4, leaving large portions of the space entirely unsampled.
- No recommended train/test split exists — all data serves as the surrogate training set.

**Sensitive information:** None. All inputs are abstract numerical coordinates; no personal or sensitive data is present.

---

## 3. Collection Process

**How were queries generated?**
Queries were generated through a Bayesian Optimisation pipeline implemented in Jupyter notebooks (Python). The pipeline at each round:

1. Loads all accumulated observations for the function
2. Fits a Gaussian Process (Matérn ν=2.5 kernel, ARD length scales, WhiteKernel noise) to the data
3. Generates 10,000–150,000 quasi-random Sobol candidates in [0,1]^d
4. Evaluates the Expected Improvement (EI) acquisition function with adaptive ξ (starting at 0.05–0.1, decaying by ×0.85 per round)
5. Selects the top 8–15 candidates and refines via L-BFGS-B local optimisation
6. Cross-checks the GP suggestion against an SVM classifier and an MC-Dropout neural network surrogate
7. Submits the final query to the capstone portal; the portal returns the true function output

**Sampling strategy:** Quasi-random (Sobol sequences) for candidate generation; exploitation-dominant after Round 3 for well-behaved functions; manual override applied when GP suggestion contradicted strong empirical evidence.

**Time frame:** Ten weekly rounds (Weeks 1–10, approximately April–June 2025). One query per function per round; 8 queries submitted per week.

**Ethical review:** Not applicable. The black-box functions are mathematical constructs with no human subjects involved.

---

## 4. Preprocessing and Uses

**Transformations applied:**

| Function | Transformation | Reason |
|----------|---------------|--------|
| F1       | Log-transform: `log(|y| + ε)` | Output range spans ~76 orders of magnitude; raw values collapse to near-zero for GP |
| F2, F3, F4, F6, F8 | None (raw values used) | Outputs are well-scaled; GP fits directly |
| F5       | Log-transform: `log(y)` | Strictly positive outputs span ~4 orders of magnitude (0.11 to 8662); plain log is safe |
| F7       | Log-transform: `log(y + ε)` | Right-skewed; 18/30 initial values near zero; epsilon guard for robustness |
| All (NN/SVM) | StandardScaler on inputs | SVM and neural network surrogates require normalised features |
| All (NN targets) | Z-score normalisation `(y - μ)/σ` | Stabilises neural network training |

**Raw data preserved:** Yes. Original `.npy` files are retained alongside processed values.

**Intended uses:**
- Surrogate model benchmarking (GP, SVM, neural network comparison)
- Acquisition function research (EI vs UCB sensitivity analysis)
- Study of Bayesian Optimisation convergence under limited query budgets
- Educational demonstration of exploration–exploitation trade-offs

**Inappropriate uses:**
- Drawing conclusions about the true global optima of the functions (budget too small to guarantee coverage)
- Generalising surrogate model performance to other domains without revalidation
- Any use that treats the dataset as statistically representative of the full input space

---

## 5. Distribution and Maintenance

**Where is the dataset available?**
The dataset (`.npy` files and Jupyter notebooks) is hosted in the student's public GitHub repository for the BBO capstone project. See the repository README for direct links to each function's data folder.

**Terms of use:**
This dataset is shared for educational and research purposes under the MIT License. The black-box function outputs were generated by the Emeritus capstone portal and are subject to any terms of use of that platform. Users may freely use, adapt and redistribute the dataset with attribution.

**Who maintains it?**
The student maintains the dataset for the duration of the capstone project. No automated update pipeline exists; new weekly entries are added manually after each portal submission. Version history is tracked via Git commit history.

**Future updates:**
No further updates are planned after the capstone submission. The dataset is considered complete at 10 rounds × 8 functions.

---

*Datasheet created: June 2025. Framework: Gebru et al., "Datasheets for Datasets," ACM FAccT 2021.*
