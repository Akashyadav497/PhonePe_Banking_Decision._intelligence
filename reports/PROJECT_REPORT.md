# Project Report — PhonePe Banking Decision Intelligence

**Domain:** Banking & Financial Technology  
**Context:** Academic Capstone Simulation  
**Deliverable:** Full Analytics Suite, Inference Engine, Machine Learning Classifiers, and Interactive Decision Dashboard  

---

> **Academic Notice:** This project uses synthetic data generated for academic demonstration and capstone evaluation. It does not contain actual PhonePe customer, transaction, or credit records, nor does it represent proprietary financial or credit policies.

---

## 1. Abstract
Digital consumer credit operations on modern payment platforms require real-time risk intelligence to balance approval volumes against portfolio loss rates. This capstone project develops an end-to-end analytical pipeline simulating lending-risk analytics using synthetic customer loan records. The framework spans four integrated analytical domains:
1. Parametric expenditure estimation, statistical inference, and closed-form linear regression baselines;
2. Ground-up logistic regression, ROC evaluation, cross-validated $K$-NN classification, and asymmetric cost-optimised threshold calibration;
3. Unsupervised deterministic $K$-means clustering, agglomerative hierarchical cohort analysis, and principal component dimensionality reduction;
4. Information-theoretic decision stump splits, hard/soft voting ensembles, manual AdaBoost convergence, and OCR/NLP deep learning architectural sizing.

---

## 2. Introduction & Problem Statement
The scenario models digital consumer lending operations connecting payment platform users to small personal credit lines offered by partner lenders. Risk intelligence operations face three fundamental analytical challenges:
1. **Spend Validation:** Budget allocations assume an average customer monthly spend of INR 8,000. Underwriting teams require rigorous parametric testing of this assumption and determination of statistical sample power.
2. **Delinquency Pre-Warning:** A collections nudge sent to a borrower who would have paid anyway is cheap (INR 50), whereas an unflagged delinquency past 90 days incurs heavy legal recovery expenses (INR 400). Standard 0.5 classification thresholds fail to minimize financial exposure on imbalanced books.
3. **Cohort Segmentation & Dimensional Compression:** Risk managers require interpretable borrower segments to tailor product covenants and sizing estimates for planned cheque OCR and transaction remark NLP systems.

---

## 3. Dataset & Feature Engineering
The pipeline processes `data/records.csv` comprising 800 synthetic borrower records across 10 structured attributes generated using seed 2508 (MD5: `e85e7605b784f4142844f79ad2dd0b8e`):

- **$X_1$ (`account_age_months`):** Account tenure in months ($\mu = 34.0, \sigma = 15.5$).
- **$X_2$ (`failed_txn_count_90d`):** Unsuccessful transactions in past 90 days ($\mu = 3.9, \sigma = 2.2$).
- **$X_3$ (`credit_utilisation_ratio`):** Share of existing credit limit utilized ($\mu = 0.43, \sigma = 0.23$).
- **$X_4$ (`avg_monthly_inflow_inr_thousands`):** Monthly bank credits in thousands of INR ($\mu = 18.8, \sigma = 8.6$).
- **$X_5$ (`repayment_score`):** Historical bureau behaviour score between 0 and 100 ($\mu = 53.3, \sigma = 12.8$).
- **$X_6$ (`irregular_salary_flag`):** Binary indicator for inconsistent payroll credits (38.8% positive).
- **$X_7$ (`autopay_flag`):** Binary indicator for active auto-debit mandate (52.6% positive).
- **Outcome (`loan_default`):** Binary indicator for serious 90+ DPD default (170 of 800, 21.25%).
- **Value (`monthly_spend_inr_thousands`):** Continuous monthly spend ($\mu = 8.9382, \sigma = 9.1723$).

---

## 4. Part-by-Part Analytical Methodology

### Part 1: Estimation, Inference & Regression Baseline
- **Inference & Power:** With sample standard deviation $s = 9.1723$ ($n-1$ divisor), standard error is $\text{SE} = 0.3243$. The 95% confidence interval spans $[8.3026, 9.5738]$. The planning hypothesis $H_0: \mu = 8.0$ is decisively rejected ($z = 2.8930, p < 0.005$). Achieving a margin of error $E = 0.5$ requires $n \ge 1,293$ records.
- **Closed-Form OLS:** Regressing spend on inflow ($X_4$) gives slope $0.6701$, intercept $-3.6598$, $R^2 = 0.5196$, and RMSE $= 6.3537$. Total variance strictly equals explained plus residual sum of squares ($TSS = ESS + RSS$).
- **Gradient Descent Verification:** 500 full-batch updates on standardised $z$-scores ($\eta = 0.1$) converged from $J_0 = 163.9178$ to $J_{500} = 40.3698$, matching OLS parameters with zero gap (`gd_slope_gap = 0.0000`).
- **Funnel Remediation:** Raw residuals exhibited severe heteroskedasticity ($\text{SD Ratio} = 3.7525$), where error variance expands as fitted spend increases. Applying the bivariate log-log transformation eliminated the funnel ($\text{SD Ratio} = 1.0396, R^2 = 0.6565$), restoring homoskedastic residual behaviour.
- **Multiple Regression:** Normal equation $(X^T X)^{-1} X^T Y$ over 8 predictors achieved adjusted $R^2 = 0.7403$, demonstrating that credit utilisation, friction, and tenure significantly boost explanatory power.

### Part 2: Delinquency Classification & Operational Cost Calibration
- **Stratified Partitioning:** Preserving class ratios (21.25%) yields 560 training rows (119 defaults) and 240 test rows (51 defaults). Predictors were standardised strictly on training moments.
- **Logistic Regression from Scratch:** Batch cross-entropy optimisation converged to loss $0.3704$. Feature odds ratios highlight credit utilisation ($\text{OR} = 2.1543$) as the primary default driver.
- **Cross-Validated $K$-NN:** 5-fold cross-validation across odd $K \in [1, 21]$ selected $K=5$ ($\text{Mean CV } F_1 = 0.4541$).
- **Asymmetric Loss Optimization:** Evaluated against PhonePe's loss function ($\text{Cost} = 50 \times \text{FP} + 400 \times \text{FN}$), the cost-optimal alert threshold is $\tau^* = 0.15$. Deploying $\tau^* = 0.15$ slashes test portfolio recovery losses to INR 6,050 compared to INR 11,800 at $\tau = 0.5$ and INR 20,400 for unmanaged defaults.

### Part 3: Cohort Segmentation & Dimensionality Reduction
- **$K$-Means Clustering:** Deterministic centroid seeding over standardised continuous variables ($X_1 \dots X_5$) converged across all $K$. The elbow metric confirmed $K=3$:
  - **Segment A (Prime Established, $N=322$):** Low utilisation (0.2250), high tenure (47.50 mo), default rate 1.86%.
  - **Segment B (High-Volume Transactors, $N=255$):** Highest inflow (INR 28,719), default rate 23.53%.
  - **Segment C (Subprime Strained, $N=223$):** High utilisation (0.7118), low score (38.52), default rate 46.64%.
- **Hierarchical Clustering:** At the dendrogram cut, single linkage identifies 4 clusters ($\{E1\}$, $\{E4\}$, $\{E5\}$, and $\{E2, E3, E6\}$; height gap between merge 2 and 3), while complete linkage identifies 2 macro-clusters ($\{E1, E4, E5\}$ and $\{E2, E3, E6\}$; height gap between merge 4 and 5). Single linkage focuses solely on local nearest-neighbor points, whereas complete linkage enforces maximum inter-cluster compactness.
- **Principal Component Analysis:** PC1 captures 52.27% of total variance, reflecting a stark "Credit Health" axis. Three components capture 86.16% of total variance ($\text{cumulative\_ratio\_3} = 0.8616$).

### Part 4: Ensembles & Neural Network Architectural Sizing
- **Tree Splitting Criteria:** Evaluating four splits identified $S_4$ ($X_1 \le 24$ months) as the purest split, yielding Information Gain $= 0.1252$ and Gini Gain $= 0.0606$.
- **Ensemble Evaluation:** Logistic regression ($\tau=0.3$) achieved test $F_1 = 0.5714$ and recall $= 0.7451$, outperforming voting ensembles burdened by weak $K$-NN predictions ($F_1 = 0.3696$).
- **AdaBoost Hand Convergence:** Two rounds of manual boosting increased weight on misclassified points to 0.5000, weighting hypothesis $h_2$ at $\alpha_2 = 0.8047$ to achieve 75.00% ensemble accuracy.
- **Deep Learning Sizing:**
  - *Model A (Cheque Digit OCR):* MLP requires 13,002 parameters; LeNet-style CNN requires 15,734 parameters while providing spatial translation invariance.
  - *Model B (Payment Remarks NLP):* Recurrent architecture requires 26,768 parameters independent of sequence length due to temporal weight sharing.

---

## 5. Dashboard Implementation
A dedicated Streamlit dashboard (`dashboard/app.py`) provides an executive interface for visualizing findings:
- Executive summary metrics and exploratory scatter/box plots.
- Part 1 gradient descent cost progression curves and residual diagnostics.
- Part 2 ROC curve threshold sweeps and 5-fold cross-validation curves.
- Part 3 cohort segmentation profiles and PCA variance breakdowns.
- Part 4 decision split purity tables and neural network layer parameter charts.
- Raw dataset exploration and direct CSV export functionality.

---

## 6. Academic Conclusion
This capstone research demonstrates how mathematical optimization, statistical inference, and machine learning models operate in concert to solve digital credit operational challenges. The closed-form and iterative equivalence in regression verifies foundational numerical stability; asymmetric cost tuning highlights that statistical accuracy is subordinate to operational business loss; and unsupervised decomposition reveals that behavioral friction and credit utilisation define customer risk far more effectively than account tenure alone.

---

## 7. Limitations & Future Scope
1. **Parametric Synthetic Data:** True payment networks display complex merchant clustering and seasonal UPI festival spending spikes that require nonparametric modeling.
2. **Cross-Sectional Evaluation:** Time-series out-of-time (OOT) validation across macroeconomic quarters should replace static train-test splits.
3. **Sample Imbalance Sensitivity:** With 51 positive default cases in the holdout test set, discrete cell shifts in confusion matrices create moderate percentage volatility.

---

## 8. References
1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer Series in Statistics.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
4. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
5. Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119-139.
