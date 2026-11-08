# Results Summary — PhonePe Banking Decision Intelligence

This document tabulates the verified outputs across all four parts of the capstone project.

---

## Part 1 Results (`outputs/part1_results.txt`)

| Key | Value | Description |
|---|---|---|
| `n` | 800 | Cohort size |
| `mean_value` | 8.9382 | Sample mean monthly spend (thousands INR) |
| `sd_value` | 9.1723 | Sample standard deviation (divisor $n-1$) |
| `se_value` | 0.3243 | Standard error of the mean |
| `ci95_low` | 8.3026 | Lower bound of 95% Confidence Interval |
| `ci95_high` | 9.5738 | Upper bound of 95% Confidence Interval |
| `z_stat` | 2.8930 | Two-tailed Z test statistic vs $\mu = 8.0$ |
| `reject_h0` | yes | Rejection of null hypothesis at $\alpha = 0.05$ |
| `n_required` | 1293 | Sample size required for margin $E = 0.5$ |
| `reg_slope` | 0.6701 | Closed-form OLS slope on $X_4$ |
| `reg_intercept` | -3.6598 | Closed-form OLS intercept |
| `tss` | 67221.4285 | Total Sum of Squares |
| `ess` | 34925.6088 | Explained Sum of Squares |
| `rss` | 32295.8197 | Residual Sum of Squares |
| `tss_equals_ess_plus_rss` | yes | Variance decomposition check |
| `r2` | 0.5196 | Coefficient of determination |
| `rmse` | 6.3537 | Root Mean Squared Error |
| `mae` | 3.7636 | Mean Absolute Error |
| `gd_J0` | 163.9178 | Gradient descent initial loss |
| `gd_J500` | 40.3698 | Gradient descent final converged loss |
| `gd_slope` | 0.6701 | Rescaled gradient descent slope |
| `gd_intercept` | -3.6598 | Rescaled gradient descent intercept |
| `gd_slope_gap` | 0.0000 | Difference between OLS and GD slope ($< 0.001$) |
| `resid_sd_ratio_raw` | 3.7525 | Raw residual standard deviation ratio (funnel) |
| `loglog_r2` | 0.6565 | Bivariate log-log model $R^2$ |
| `resid_sd_ratio_log` | 1.0396 | Log model residual SD ratio (remedy) |
| `mr_r2` | 0.7425 | Multiple regression $R^2$ on 8 predictors |
| `mr_adj_r2` | 0.7403 | Multiple regression adjusted $R^2$ |
| `loglog_adj_r2` | 0.6561 | Bivariate log-log adjusted $R^2$ |

---

## Part 2 Results (`outputs/part2_results.txt`)

| Key | Value | Description |
|---|---|---|
| `train_rows` | 560 | Stratified training observations |
| `train_positives` | 119 | Training default cases (21.25%) |
| `test_rows` | 240 | Stratified test observations |
| `test_positives` | 51 | Test default cases (21.25%) |
| `loss_epoch_0` | 0.6931 | Logistic loss prior to updates |
| `loss_epoch_1000` | 0.3704 | Final converged training loss ($\eta = 0.1$) |
| `weight_X1` | -0.5554 | Account age standardized weight |
| `weight_X2` | 0.3540 | Failed transactions standardized weight |
| `weight_X3` | 0.7675 | Credit utilisation standardized weight |
| `bias` | -1.7454 | Logistic intercept term |
| `test_bce` | 0.3807 | Out-of-sample binary cross-entropy loss |
| `odds_ratio_X3` | 2.1543 | Odds ratio for credit utilisation |
| `blind_accuracy` | 0.7875 | Test accuracy predicting all zero |
| `tau_0.5_accuracy` | 0.8042 | Accuracy at default threshold 0.5 |
| `tau_0.5_recall` | 0.4706 | Recall at default threshold 0.5 |
| `tau_0.3_recall` | 0.7451 | Recall at lowered threshold 0.3 |
| `tau_0.3_f1` | 0.5714 | $F_1$ score at lowered threshold 0.3 |
| `knn_best_k` | 5 | Optimal $K$ via 5-fold cross-validation |
| `knn_f1` | 0.3696 | Test $F_1$ score of $K$-NN classifier |
| `cost_tau` | 0.15 | Cost-minimizing alert threshold on train |
| `cost_train_at_tau` | 12150 | Total operational cost on train at $\tau^*$ |
| `cost_test_at_tau` | 6050 | Out-of-sample test cost at $\tau^*$ (INR) |
| `cost_test_at_0.5` | 11800 | Out-of-sample test cost at $\tau = 0.5$ (INR) |
| `cost_test_flag_nobody`| 20400 | Out-of-sample test cost flagging nobody (INR) |

---

## Part 3 Results (`outputs/part3_results.txt`)

| Key | Value | Description |
|---|---|---|
| `wcss_k2` | 2262.6753 | Within-cluster sum of squares ($K=2$) |
| `wcss_k3` | 1602.8471 | Within-cluster sum of squares ($K=3$) |
| `elbow_k` | 3 | Optimal $K$ determined by elbow rule |
| `size_segment_A` | 322 | Prime established segment count |
| `size_segment_B` | 255 | High-inflow growth segment count |
| `size_segment_C` | 223 | Subprime risk segment count |
| `single_clusters_at_cut` | 4 | Cohort clusters under single linkage |
| `complete_clusters_at_cut`| 2 | Cohort clusters under complete linkage |
| `eigenvalue_sum` | 5.0000 | Sum of covariance eigenvalues |
| `explained_ratio_1` | 0.5227 | Variance explained by PC1 (52.27%) |
| `cumulative_ratio_3`| 0.8616 | Cumulative variance captured by 3 PCs (86.16%) |
| `components_for_80pct` | 3 | Smallest number of PCs exceeding 80% |
| `mean_pc1_segment_A` | 1.7773 | Segment A PC1 mean score |
| `mean_pc1_segment_C` | -1.6709 | Segment C PC1 mean score |

---

## Part 4 Results (`outputs/part4_results.txt`)

| Key | Value | Description |
|---|---|---|
| `root_entropy` | 0.7462 | Entropy of 560 training records |
| `root_gini` | 0.3347 | Gini impurity of 560 training records |
| `best_split_by_info_gain`| S4 | Purest split by Information Gain ($X_1 \le 24$) |
| `best_split_by_gini_gain`| S4 | Purest split by Gini Gain ($X_1 \le 24$) |
| `round1_error` | 0.2500 | AdaBoost Round 1 weighted error |
| `round1_alpha` | 0.5493 | AdaBoost Round 1 hypothesis weight |
| `round2_error` | 0.1667 | AdaBoost Round 2 weighted error |
| `round2_alpha` | 0.8047 | AdaBoost Round 2 hypothesis weight |
| `ensemble_accuracy` | 0.7500 | 2-round AdaBoost final ensemble accuracy |
| `oob_fraction` | 0.3676 | Analytical out-of-bag sample share |
| `mlp_total` | 13002 | Model A Option 1 MLP parameter count |
| `cnn_total` | 15734 | Model A Option 2 CNN parameter count |
| `fc_layer_params` | 50332672 | Fully connected scaling benchmark params |
| `conv_layer_params` | 1792 | Convolutional scaling benchmark params |
| `rnn_total` | 26768 | Model B sequence RNN parameter count |
