Domain: Banking

## How to run

> **Academic Notice:** This project uses synthetic data generated for academic and demonstration purposes. It does not contain actual PhonePe customer or banking transaction records and is not an official commercial deployment.

To reproduce all results from a clean repository clone, execute the following commands in order from the repository root:

```bash
# 1. Install pinned dependencies
pip install -r requirements.txt

# 2. Generate the synthetic dataset (if data/records.csv is not present)
python generate_data.py

# 3. Execute Part 1 (Estimation, inference, and regression baseline)
python part1.py

# 4. Execute Part 2 (Classification, ROC sweep, KNN CV, and cost optimisation)
python part2.py

# 5. Execute Part 3 (K-means, hierarchical clustering, and PCA)
python part3.py

# 6. Execute Part 4 (Entropy/Gini splits, voting ensemble, AdaBoost, and architecture sizing)
python part4.py

# 7. (Optional) Run the interactive Streamlit analytics dashboard
streamlit run dashboard/app.py
```

All generated text results and CSV files will be written to the `outputs/` directory.

## Part 1 findings

- **R1.1 Hypothesis Test & Planning Claim**: The null hypothesis $H_0: \mu = 8.0$ is rejected at significance level $\alpha = 0.05$ because the test statistic $|z_{\text{stat}}| = 2.8930$ exceeds the two-tailed critical value of $1.96$. In domain terms, a Type I error would occur if the risk operations team falsely concludes that average monthly customer spend differs from the planned benchmark of INR 8,000 when the true population mean is indeed INR 8,000, triggering an unnecessary revision of lending limits and financial projections. The observed difference $\text{mean\_value} - 8.0 = 0.9382$ column units represents an excess spend of INR 938.20 per borrower; in digital lending portfolios operating on tight margins, a ~11.7% deviation is practically meaningful and warrants updating underwriting assumptions.
- **R1.2 Sample Sizing**: The minimum required sample size to estimate the mean spend within a margin of error $E = 0.5$ (INR 500) at 95% confidence is $n_{\text{required}} = 1293$. Because the available book contains only 800 rows ($800 < 1293$), the planning team's current dataset is underpowered to guarantee the desired precision of $\pm\text{INR } 500$, implying that broader confidence bounds must be accepted until ~500 additional accounts are observed.
- **R1.3 Gradient Descent Verification**: The absolute slope gap between closed-form OLS and gradient descent is `gd_slope_gap = 0.0000`. Gradient descent reached the exact closed-form solution because the objective function for linear regression is strictly convex with a single global minimum, and feature standardisation ensured circular contours for smooth, rapid convergence within 500 full-batch steps at learning rate $\eta = 0.1$.
- **R1.4 Residual Funnel & Log Transformation**: The ratio of residual standard deviations in the raw regression model is $\text{resid\_sd\_ratio\_raw} = 3.7525$ (upper half $8.6891$ vs. lower half $2.3155$), displaying severe heteroskedasticity—a widening residual funnel where the spread of the residuals grows substantially with larger fitted spend values. Applying the log-log transformation reduces this ratio to $\text{resid\_sd\_ratio\_log} = 1.0396$ (upper $0.5369$ vs. lower $0.5165$), effectively removing the funnel and restoring near-perfect homoskedastic residual spread across fitted tiers.
- **R1.5 Multiple Regression Coefficients & Adjusted $R^2$**: In Task 1.7, the predictors whose coefficients have an absolute value exceeding the mechanical threshold of $0.05$ are $\ln(X_4)$ ($1.1576$), $X_2$ ($0.0940$), $X_3$ ($0.1067$), and $X_6$ ($0.1518$). Comparing the adjusted $R^2$ of the multiple regression model ($\text{mr\_adj\_r2} = 0.7403$) against the bivariate log-log model ($\text{loglog\_adj\_r2} = 0.6561$) demonstrates an improvement of ~8.4 percentage points of explained variance, confirming that the additional behavioural friction and repayment predictors earn their place in the model.

## Part 2 findings

- **R2.1 Learning Rate Comparison**: Ordering the training losses after 200 epochs from lowest to highest yields $\eta = 1.0$ ($0.3755$) $<$ $\eta = 0.1$ ($0.3879$) $<$ $\eta = 0.01$ ($0.5100$). This demonstrates that an excessively small learning rate ($\eta = 0.01$) progresses sluggishly and fails to converge within 200 epochs, whereas larger rates advance substantially faster along smooth convex surfaces without destabilising when the surface is well-conditioned.
- **R2.2 Key Predictors & Odds Ratios**: The three predictors with the largest absolute weights are $X_3$ (`credit_utilisation_ratio`, weight $= 0.7675$), $X_1$ (`account_age_months`, weight $= -0.5554$), and $X_2$ (`failed_txn_count_90d`, weight $= 0.3540$). The largest odds ratio belongs to $X_3$ ($\text{odds\_ratio\_X3} = 2.1543$): for every increase of one standard deviation in credit utilisation ratio (~0.23), the odds of loan default more than double ($\times 2.1543$), holding all other borrower attributes fixed.
- **R2.3 Classification Threshold Dynamics**: On the test partition, the baseline naive classifier yields `blind_accuracy = 0.7875` (predicting no defaults for all 240 rows), while logistic regression at $\tau = 0.5$ achieves an accuracy of $0.8042$. Blind accuracy illustrates why accuracy is deceptive for imbalanced books (~21.25% defaults): a completely uninformative model achieves nearly 79% accuracy yet detects 0 defaults. Lowering $\tau$ from $0.5$ to $0.3$ increases recall dramatically from $0.4706$ to $0.7451$ (detecting 38 defaults instead of 24), while precision drops moderately from $0.5455$ to $0.4634$.
- **R2.4 Model Comparison (Logistic vs. KNN)**: Comparing the logistic regression model at $\tau = 0.3$ with the cross-validated $K$-NN model ($K = 5$) on the test set, logistic regression achieves an $F_1$ of $0.5714$ and a recall of $0.7451$, whereas $K$-NN attains an $F_1$ of $0.3696$ and a recall of only $0.3333$. Logistic regression heavily outperforms $K$-NN by capturing linear risk trends that metric distance struggles to separate in sparse 7-dimensional space.
- **R2.5 Policy Deployment & Threshold Selection**: In this simulation, selecting the cost-optimal threshold $\tau^* = 0.15$ yields an expected test operational cost of $\text{INR } 6050$, compared to $\text{INR } 11800$ at default $\tau = 0.5$ and $\text{INR } 20400$ for flagging nobody. Threshold $\tau^*$ was chosen strictly on the training set because calibrating decision thresholds on test data introduces data snooping, resulting in overly optimistic cost expectations that fail to generalise out-of-sample.

## Part 3 findings

- **R3.1 K-means WCSS Drops & Elbow Selection**: The four successive WCSS drops are:
  - $\text{WCSS}(2) - \text{WCSS}(3) = 2262.6753 - 1602.8471 = 659.8282$
  - $\text{WCSS}(3) - \text{WCSS}(4) = 1602.8471 - 1494.7732 = 108.0739$
  - $\text{WCSS}(4) - \text{WCSS}(5) = 1494.7732 - 1323.7157 = 171.0575$
  - $\text{WCSS}(5) - \text{WCSS}(6) = 1323.7157 - 1278.1691 = 45.5466$
  
  The mathematical elbow rule identifies $K = 3$ because the drop from $K=3$ to $K=4$ ($108.0739$) is less than half the preceding drop ($0.5 \times 659.8282 = 329.9141$). The non-monotonic drop at $K=5$ highlights that pinned deterministic starting rows yield reproducible partitions rather than guaranteed global optima.
- **R3.2 Segment Profiling & Business Strategy**:
  - **Segment A ("Prime Established Borrowers", size = 322)**: Longest account age ($47.5$ months), lowest friction ($1.95$ failed transactions), low credit utilisation ($0.2250$), high repayment score ($62.16$), and lowest default rate ($1.86\%$). *Action*: Offer automated pre-approved prime credit limit increases and premium cards.
  - **Segment B ("High-Volume Growth Borrowers", size = 255)**: Highest inflow ($\text{INR } 28,719$) and spend ($\text{INR } 17,125$), moderate tenure ($28.6$ months), elevated friction ($6.07$ failed txns), and default rate of $23.53\%$. *Action*: Provide proactive liquidity buffers and smart repayment scheduling linked to salary credits.
  - **Segment C ("High-Risk Subprime Borrowers", size = 223)**: Highest default rate at $46.64\%$, highest credit utilisation ($0.7118$), shortest account tenure ($14.39$ months), and lowest repayment score ($38.52$). *Action*: Implement strict underwriting caps, daily micro-repayments, and mandatory auto-debit covenants.
- **R3.3 Hierarchical Clustering Cut Discrepancy**: At the dendrogram cut, single linkage identifies 4 clusters ($\{E1\}$, $\{E4\}$, $\{E5\}$, and $\{E2, E3, E6\}$ because its largest height gap occurs between merge 2 at height $1.4000$ and merge 3 at height $2.8000$, leaving 2 merges below the cut: $6 - 2 = 4$). In contrast, complete linkage forms 2 clusters ($\{E1, E4, E5\}$ and $\{E2, E3, E6\}$ because its largest height gap is between merge 4 at $2.9000$ and merge 5 at $7.9000$, leaving 4 merges below the cut: $6 - 4 = 2$). The two rules disagree because single linkage evaluates only the minimum pairwise distance between clusters and detects an early plateau between local merges, whereas complete linkage enforces maximum pairwise compactness across all members, preventing cluster merger until the large inter-cohort gap of $7.9000$ is crossed.
- **R3.4 Principal Component Analysis**: The first principal component (PC1) accounts for $52.27\%$ of total variance and is driven in the positive direction by $X_1$ ($+0.5317$) and $X_5$ ($+0.4418$), and in the negative direction by $X_3$ ($-0.5187$) and $X_2$ ($-0.4100$), reflecting a distinct "Credit Health vs. Credit Distress" axis. Retaining 3 components is required to capture at least 80% of total variance ($\text{cumulative\_ratio\_3} = 0.8616$). The segment PC1 means ($\bar{z}_{1,A} = +1.7773$, $\bar{z}_{1,B} = -0.7831$, $\bar{z}_{1,C} = -1.6709$) cleanly separate the borrower segments along this axis, confirming the cluster structure.

## Part 4 findings

- **R4.1 Tree Splits & Regularisation**: Split $S_4$ ($X_1 \le 24.0$ months) is the purest split, yielding the highest information gain ($0.1252$) and Gini gain ($0.0606$). A decision tree grown until every leaf is pure memorises sample noise, idiosyncrasies, and outliers, causing high variance and poor generalisation on unseen records. Common stopping regularisers include establishing a minimum leaf sample count (e.g., `min_samples_leaf >= 20`) or capping maximum tree depth.
- **R4.2 Model Evaluation & Ensemble Comparison**: Among the five models evaluated in Task 4.2, the logistic regression model ($\tau = 0.3$) is the superior model to ship, achieving an $F_1$ of $0.5714$ and a recall of $0.7451$. Ensembling models did not improve performance: hard voting dropped $F_1$ to $0.5344$ and recall to $0.6863$, while soft voting achieved $F_1 = 0.5507$ and recall $= 0.7451$, primarily because the weak $K$-NN model ($F_1 = 0.3696$) degraded the ensemble consensus.
- **R4.3 AdaBoost Round Dynamics**: After round 1 normalisation, the two misclassified points (Points 3 and 7) have a combined weight of $0.2500 + 0.2500 = 0.5000$. Round 2 assigns a higher voting confidence $\alpha_2 = 0.8047$ to hypothesis $h_2$ compared to $\alpha_1 = 0.5493$ for $h_1$ because $h_2$ achieves a lower weighted error ($\varepsilon_2 = 0.1667$ vs. $\varepsilon_1 = 0.2500$) on the reweighted dataset.
- **R4.4 Deep Learning Architectural Sizing**:
  - **Model A (Option 2 CNN Layer Breakdown)**:
    - `cnn_conv1`: output shape $28 \times 28 \times 6$, parameters $= 156$
    - `cnn_pool1`: output shape $14 \times 14 \times 6$, parameters $= 0$
    - `cnn_conv2`: output shape $10 \times 10 \times 16$, parameters $= 2,416$
    - `cnn_pool2`: output shape $5 \times 5 \times 16$, parameters $= 0$
    - `cnn_dense1`: output shape $32$, parameters $= 12,832$
    - `cnn_dense2`: output shape $10$, parameters $= 330$
  - **Model B Recurrent Parameters**: The weight shapes are $W_{xh} \in \mathbb{R}^{128 \times 64}$, $W_{hh} \in \mathbb{R}^{128 \times 128}$, and $W_{hy} \in \mathbb{R}^{16 \times 128}$. The total parameter count (`rnn_total = 26768`) does not scale with sequence length because RNN weights are shared across all unrolled time steps.
  - **Classification Settings & Vanishing Gradients**: For 10-class digit recognition, the hidden activation should be ReLU and the output activation Softmax paired with Categorical Cross-Entropy loss. In very deep networks, repeated backpropagation multiplications by small weight matrices cause gradients to decay exponentially toward zero, halting early-layer learning. Skip connections remedy this by providing identity gradient highways that allow gradients to flow backwards unimpeded.

## Model-choice memo

**To:** Head of Lending-Risk Operations  
**From:** Lead Decision Intelligence Analyst  
**Date:** 07 November 2026  
**Subject:** Delinquency Model Benchmark & Simulation Recommendation  

Based on our benchmark experiments on the synthetic lending dataset, I recommend the **regularised Logistic Regression model** operating at alert threshold $\boldsymbol{\tau = 0.15}$ as the illustrative production baseline for delinquency pre-warning.

On the 240-record holdout test cohort, this policy detects potential defaults with high sensitivity, capturing 46 out of 51 defaults ($\text{recall} = 90.20\%$). Under the simulated operational loss function (INR 50 per false nudge, INR 400 per unflagged default), this policy yields a total test recovery cost of **INR 6,050**, substantially outperforming the standard uncalibrated $\tau = 0.5$ rule (**INR 11,800**) and the unmanaged baseline of flagging nobody (**INR 20,400**).

Ensemble combinations did not provide an advantage: $K$-NN suffered from dimensional sparsity ($\text{recall} = 33.33\%$, $F_1 = 0.3696$), pulling down both soft voting ($\text{recall} = 74.51\%$) and hard voting ($\text{recall} = 68.63\%$). Logistic regression provides superior probabilistic calibration and clear linear interpretability (e.g., credit utilisation doubling default odds, $\text{odds ratio} = 2.1543$).

Importantly, these benchmarks reflect synthetic cohort distributions under simulated credit assumptions; actual commercial deployment requires prospective validation against live merchant settlement dynamics and seasonal UPI payment flows. Furthermore, neither Model A (cheque OCR) nor Model B (payment remark NLP) warrants engineering investment prior to deploying structured baseline scoring, as automated delinquency interventions on core account features generate immediate, quantifiable risk reductions.

## Limitations

1. **Synthetic Data Generation**: The underlying records originate from a parametric generator. Real-world Indian digital payments encompass complex multimodal distributions, merchant settlement latencies, and regional credit behaviours not fully captured here.
2. **Single Cross-Sectional Split**: The models and thresholds are validated on a single 70/30 stratified train-test split rather than rolling out-of-time (OOT) payment windows, leaving sensitivity to temporal drift unmeasured.
3. **Small Positive Test Volume**: With only 51 positive default cases in the test set, cell counts in the confusion matrix are subject to integer sensitivity where a single misclassification moves precision or recall by ~2 percentage points.
