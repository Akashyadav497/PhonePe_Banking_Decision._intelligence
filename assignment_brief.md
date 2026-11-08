# Capstone Project — PhonePe Banking Decision Intelligence

**Domain:** Banking · **Program:** Minor in Artificial Intelligence & Data Science, IIT Mandi · **Total marks:** 100 · **One deliverable:** a single public GitHub repository (one branch, `main`)

**Window:** 11 calendar days, 6 November 2026 to 16 November 2026. Submit on your course platform's submission page before the closing time shown there.

## Scenario

You have joined the lending-risk analytics team of **PhonePe**, one of India's largest UPI payments platforms, which also connects its users to small digital loans offered by partner lenders. The team has three open questions and no pipeline that answers them:

1. **Is the planning assumption about monthly spend still right, and how far can a simple model of spend be trusted?** Budgets are built on an average monthly spend per borrower; nobody has tested that assumption against the data or measured how much sample size the estimate needs.
2. **Which borrowers will default or fall 90 days behind?** A collections nudge is only useful if it goes out before the due date. A reminder to a borrower who would have paid anyway is cheap; a default that was never flagged is expensive.
3. **What kinds of borrowers does the book really contain, and what would the team need if it later wanted a model that reads cheque digits or sorts payment remarks?** The team wants the segments first and the sizing of those later models as numbers, not a build.

You will answer all of this in **one connected repository**. Part 1 estimates and tests the average of the value column and builds a regression baseline. Part 2 builds a classifier for the outcome from the ground up and chooses its alert threshold with money. Part 3 segments the records and compresses their features. Part 4 compares models for the structured data, runs a boosting calculation by hand in code, and sizes the neural networks a later project would need. Later Parts reuse the data, the 560/240 split and the models of earlier Parts; no Part may use a number it did not compute itself or receive from an earlier Part, except the constants this brief prints.

> **The data is synthetic.** PhonePe's real records are never used and nothing here reflects its actual data, products, policies or performance. Every row comes from the generator script below. The brand is only the setting for the decisions. Nothing here is financial or credit advice.

## Submission Guidelines (read first)

- **Submit exactly one public GitHub repository link.** The repository's `README.md` is the front door and carries every written answer. There are no per-Part submissions.
- **All work must live on one branch, `main`, and the link you submit must point to `main`.** The accepted forms are the repository root address `https://github.com/<user>/<repo>` (only when `main` is the repository's default branch) or `https://github.com/<user>/<repo>/tree/main`. Check on the repository page that the default branch is `main` before you submit. A link to any other branch, or a link that does not clearly resolve to `main`, **will not be graded**.
- **The one link must contain the entire deliverable.** Do not place any other link inside the repository or the README for a grader to open (no notebook link, no shared-drive folder, no published app, no second repository). Any other link inside the submission will not be opened or graded. Everything a grader needs is the text of the files in this repository.
- **Textual deliverables only.** Code is submitted through the repository and every result is a text or CSV file. **No images, screenshots, PDFs, slides, presentations, videos or audio are required or accepted** (do not commit any; none is graded). Part 4 talks about image-sized inputs only as numbers (28×28 pixels); no picture file is ever used.
- **Graders do not run your code.** They read the files in the repository and compare the numbers in your committed `outputs/` files with their reference. Your code must still be the source of every number: the scripts must run top to bottom on a clean clone, in the order given in your README, and regenerate exactly the committed `outputs/` files.
- **No paid or account-gated service, no API key and no dataset download is needed.** Everything runs offline in plain Python (3.10 or newer) with the free package NumPy. A free GitHub account is needed only to host the repository.
- **Allowed tools.** Python standard library and NumPy. `requirements.txt` must contain a line `numpy==<the version you used>`, plus one pinned line (`<package>==<version>`) for every extra package you use; the reference numbers in this brief were verified on NumPy 1.26.4 and on NumPy 2.2.6 and are identical on both; pin whichever version you use. Other packages may be used for reading files or for cross-checking your own work, but **every graded quantity must be computed by code you wrote**: for example, no ready-made logistic-regression, k-nearest-neighbour, k-means, PCA or metric routine may produce a number you report. NumPy's array arithmetic and its linear-algebra routines (`numpy.linalg`) are expected tools: use them for the matrix solve in Task 1.7 and the eigen-decomposition in Task 3.5. Typing a reference number (including a Checkpoint value) instead of computing it scores that line zero.
- **Money is in Indian Rupees (INR).** Write amounts as `INR 400` or `400 Rupees` in the README; never use a dollar sign for money.
- **Pinned conventions decide the numbers.** Every place where two reasonable people could compute different numbers (split rule, tie-break, starting points, which standard deviation, rounding) is pinned in this brief. Follow the pinned convention exactly; a different convention gives different, ungradable numbers.
- **Pass bands (tolerances).** Counts, yes/no values and text values must match the reference exactly. Other numbers are correct when within `max(0.005, 0.0005 × |reference|)` of the reference (so ±0.005 for values below 10, about ±0.05 near 100 and about ±1 near 2,000). Exceptions: confusion-matrix cells (`tp`, `fp`, `tn`, `fn`) may be off by 1, because a probability within 0.0001 of a threshold can flip with tiny rounding differences; rates computed from those cells may be off by 0.02 (0.03 for k-nearest-neighbour and voting rows); the cross-validated F1 values in `part2_knn_cv.csv` and the `outcome_rate` column of `part3_segments.csv` may be off by 0.01; the PCA loadings and the three-row projections may be off by 0.01 and `mean_pc1_segment_A/B/C` by 0.02; money amounts may be off by INR 450. Two items are stricter than the default: `cost_tau` must match exactly (it is a multiple of 0.05) and `gd_slope_gap` must be below 0.001. One item is more lenient: `n_required` one above or one below the reference earns half of that item's marks as a rounding slip. Results that depend on the pinned split or pinned folds (the best K in Task 2.7, the threshold in Task 2.8) are valid for that split only.
- **What a missing piece costs.** Each results key, each CSV file and each README answer is scored on its own. A missing key or file scores zero for that item only; it never cancels marks for other items. A README answer that quotes a number must quote it from your own committed output.
- **Original work.** The code and the README must be your own work, and you must be able to explain every line of it.
- **Suggested pace.** Part 1 in days 1–3, Part 2 in days 4–6, Part 3 in days 7–8, Part 4 and the README in days 9–11. Write each README section as soon as its Part is finished.

### Required repository layout

```
<repo>/
├── README.md              # first line: Domain: Banking
├── requirements.txt       # numpy==<version>, plus one pinned line per extra package
├── generate_data.py       # the provided script, committed verbatim
├── data/
│   └── records.csv        # generator output, committed
├── part1.py  part2.py  part3.py  part4.py     # one script per Part (shared helper files are fine)
└── outputs/               # every results file and CSV named in this brief
    ├── part1_results.txt  part2_results.txt  part3_results.txt  part4_results.txt
    ├── part2_thresholds.csv  part2_knn_cv.csv
    ├── part3_segments.csv  part3_hierarchical.csv
    └── part4_splits.csv  part4_voting.csv  part4_adaboost.csv  part4_architecture.csv
```

**The first line of the README must be exactly `Domain: Banking`.**

**Results-file format.** Each `partN_results.txt` has one line per key, written `key = value`, with the keys named in the tasks below. Floating-point values are printed with 4 decimal places, integers as integers, true/false values as `yes` or `no`, and text values as given. Compute with full precision and round only when printing. Every CSV file uses the exact header given in its task, a comma separator and one row per item, with floats printed to 4 decimal places.

---

## Provided data — `generate_data.py`

Commit the script below **verbatim** as `generate_data.py` and run `python generate_data.py` once from the repository root. It writes `data/records.csv` and must print `800 10 170 e85e7605b784f4142844f79ad2dd0b8e` (rows, columns, number of records with `loan_default = 1`, and the MD5 hash of the file). Do not edit any constant and do not hand-edit the generated file.

```python
"""Synthetic records for the capstone (all values are invented; no real company data).
Run once from the repository root:  python generate_data.py
It writes data/records.csv and prints:  <rows> <columns> <positive_count> <md5 of the file>
Do not edit anything below the line marked NUMERIC CORE: every number in the brief was
measured on this exact output."""
import csv
import hashlib
import math
import os
import random

# ---------------- DOMAIN SETTINGS (names only; they never touch the numbers) ----------------
DOMAIN = "Banking"
ID_PREFIX = "B"
COLUMNS = [
    "record_id",
    "account_age_months",
    "failed_txn_count_90d",
    "credit_utilisation_ratio",
    "avg_monthly_inflow_inr_thousands",
    "repayment_score",
    "irregular_salary_flag",
    "autopay_flag",
    "loan_default",
    "monthly_spend_inr_thousands"
]
# ---------------- NUMERIC CORE (identical in every domain) ----------------
N = 800
SEED = 2508
GROUP_P = [0.40, 0.35, 0.25]          # three hidden groups
MU_C1 = [48.0, 30.0, 14.0]            # tenure in months
MU_C2 = [2.0, 6.0, 4.0]               # count of friction events
MU_C3 = [0.22, 0.45, 0.72]            # ratio in [0, 1]
MU_C4 = [math.log(12.0), math.log(26.0), math.log(15.0)]   # log of an amount
MU_C5 = [62.0, 55.0, 40.0]            # score
P_B1 = [0.30, 0.55, 0.45]
P_B2 = [0.60, 0.50, 0.40]


def normal(rng):
    """One standard-normal draw from two uniform draws (Box-Muller)."""
    u1 = 1.0 - rng.random()
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def make_rows():
    rng = random.Random(SEED)
    rows = []
    for i in range(N):
        u = rng.random()
        g = 0 if u < GROUP_P[0] else (1 if u < GROUP_P[0] + GROUP_P[1] else 2)
        c1 = round(max(1.0, MU_C1[g] + 9.0 * normal(rng)), 1)
        c2 = int(max(0, round(MU_C2[g] + 1.6 * normal(rng))))
        c3 = round(min(0.99, max(0.01, MU_C3[g] + 0.12 * normal(rng))), 3)
        c4 = round(math.exp(MU_C4[g] + 0.35 * normal(rng)), 1)
        c5 = round(min(100.0, max(0.0, MU_C5[g] + 10.0 * normal(rng))), 1)
        b1 = 1 if rng.random() < P_B1[g] else 0
        b2 = 1 if rng.random() < P_B2[g] else 0
        eta = (-2.05 + 0.9 * (c3 - 0.45) / 0.25 - 0.7 * (c1 - 30.0) / 15.0
               + 0.35 * (c2 - 4.0) / 2.5 + 0.5 * b1 - 0.45 * b2 - 0.25 * (c5 - 55.0) / 12.0)
        p = 1.0 / (1.0 + math.exp(-eta))
        outcome = 1 if rng.random() < p else 0
        mu = 8.0 * math.exp(0.40 * (math.log(c4) - math.log(20.0)) / 0.35 + 0.25 * (c2 - 4.0) / 2.5
                            + 0.15 * b1 - 0.10 * (c1 - 30.0) / 15.0)
        value = round(mu * math.exp(0.45 * normal(rng) - 0.45 * 0.45 / 2.0), 2)
        rows.append([ID_PREFIX + format(i + 1, "04d"), c1, c2, c3, c4, c5, b1, b2, outcome, value])
    return rows


if __name__ == "__main__":
    data = make_rows()
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "records.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(data)
    with open(os.path.join("data", "records.csv"), "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    print(len(data), len(COLUMNS), sum(r[8] for r in data), digest)
```

### Columns and roles

The ten columns, in file order, play the following roles. The brief and the grader's reference call the seven predictors **X1 to X7** and refer to the other two columns as the **outcome** and the **value**. The column order in the file is fixed: X1 is the second column, X7 the eighth, the outcome the ninth and the value the tenth (the first column is only an identifier). In the generator code the same columns appear as `c1` to `c5` (X1 to X5), `b1` and `b2` (X6, X7), `outcome` and `value`.

| Role | Column | Meaning |
|---|---|---|
| (identifier) | `record_id` | Row identifier; not used in any calculation |
| X1 | `account_age_months` | Months since the account was opened |
| X2 | `failed_txn_count_90d` | Failed payment attempts in the last 90 days (count) |
| X3 | `credit_utilisation_ratio` | Share of the credit line in use (0 to 1) |
| X4 | `avg_monthly_inflow_inr_thousands` | Average monthly inflow, in thousands of INR |
| X5 | `repayment_score` | Past repayment behaviour score (0 to 100) |
| X6 | `irregular_salary_flag` | 1 if salary credits arrive irregularly, else 0 |
| X7 | `autopay_flag` | 1 if autopay is set up, else 0 |
| outcome | `loan_default` | 1 if the borrower defaulted or went 90 or more days past due, else 0 |
| value | `monthly_spend_inr_thousands` | Monthly spend, in thousands of INR |

The **outcome** is early loan default or serious delinquency. About one record in five has the outcome (170 of 800). Because of this imbalance, plain accuracy is a misleading score, and the tasks below ask for more informative measures.

---

## Conventions used by every Part

- **Row order.** Rows are in file order and are numbered from 0 where an index is needed. "Row 1", "row 2" and "row 3" in Task 3.5 mean the first three data rows (indices 0, 1, 2).
- **Standard deviation and variance** use the sample form (divide by n − 1) everywhere unless a task says otherwise.
- **ln** means the natural logarithm. **log2** is the base-2 logarithm.
- **Positive** means outcome = 1. A model predicts positive when its probability is **greater than or equal to** the stated threshold τ.
- **Confusion-matrix cells** are `tp` (predicted positive, outcome 1), `fp` (predicted positive, outcome 0), `tn` (predicted negative, outcome 0) and `fn` (predicted negative, outcome 1). Accuracy = (tp + tn) / (tp + fp + tn + fn); precision = tp / (tp + fp); recall (also called sensitivity and the true-positive rate, TPR) = tp / (tp + fn); specificity = tn / (tn + fp); the false-positive rate FPR = fp / (fp + tn); F1 = 2 × precision × recall / (precision + recall). If a denominator is 0, the quantity is 0.
- **Checkpoints.** A few results are printed in this brief so that you can confirm your pipeline early. They are marked **Checkpoint**.

---

## Part 1 — Estimation, inference and a regression baseline  (25 marks)

All of Part 1 uses all 800 rows. Part 1 produces `outputs/part1_results.txt` (script `part1.py`).

**Tasks**

1.1 **Estimate with an interval.** For the value column compute `n`, `mean_value`, `sd_value` (sample SD), the standard error `se_value` = `sd_value` / √n, and the 95% interval `ci95_low` and `ci95_high` = `mean_value` ∓ 1.96 × `se_value`. Because n is far above 30, the sample SD stands in for the population SD and the normal critical value 1.96 applies. **Checkpoint:** `n = 800`, `mean_value = 8.9382`.

1.2 **Test a planning claim.** The planning team's budget assumes that a borrower spends INR 8,000 a month on average (8.0 in the column's units, which are thousands of INR). Test H0: μ = 8.0 against H1: μ ≠ 8.0 at significance level α = 0.05 (two-tailed). Compute `z_stat` = (`mean_value` − 8.0) / `se_value` and `reject_h0` (`yes` when |`z_stat`| > 1.96, otherwise `no`).

1.3 **Size a future sample.** The team wants the average estimated to within a margin of error E = 0.5 column units (INR 500) at 95% confidence. With the `sd_value` from Task 1.1, `n_required` is the smallest integer n such that n ≥ (1.96 × `sd_value` / E)².

1.4 **Simple regression by the closed form.** Regress the value on X4 (`avg_monthly_inflow_inr_thousands`) with one predictor. Compute, with your own code, `reg_slope` = cov(X4, value) / var(X4) (both with divisor n − 1), `reg_intercept` = mean(value) − `reg_slope` × mean(X4), the fitted values, `tss` = Σ(y − mean y)², `ess` = Σ(fitted − mean y)², `rss` = Σ(y − fitted)², `tss_equals_ess_plus_rss` (`yes` when |`tss` − `ess` − `rss`| < 10⁻⁶ × `tss`), `r2` = 1 − `rss` / `tss`, `rmse` = √(`rss` / n) and `mae` = mean |y − fitted|.

1.5 **Fit the same line by gradient descent.** Standardise X4 into z = (X4 − mean X4) / SD of X4 (sample SD). Model: ŷ = m × z + c, starting from m = 0 and c = 0. Cost J = (1/n) Σ e², where e = y − ŷ. The gradients are ∂J/∂m = −(2/n) Σ z × e and ∂J/∂c = −(2/n) Σ e. Update m := m − η ∂J/∂m and c := c − η ∂J/∂c with η = 0.1, using all 800 rows for every update, for 500 updates. Write `gd_J0` (cost with m = c = 0), and `gd_J1`, `gd_J10`, `gd_J100`, `gd_J500` (cost after that many updates). Convert the final line back to the original X4 scale: `gd_slope` = m / (SD of X4) and `gd_intercept` = c − m × (mean X4) / (SD of X4). Write `gd_slope_gap` = |`gd_slope` − `reg_slope`|.

1.6 **Check for a funnel, then apply the log remedy.** Sort the 800 rows by the fitted value of the Task 1.4 model, ascending (ties by row index: use a stable sort, for example `numpy.argsort(..., kind='stable')`, because many fitted values are tied and the default sort does not keep tied rows in row order, which moves `resid_sd_ratio_raw` outside the pass band). The lower group is the first 400 rows of that order, the upper group the last 400. Write `resid_sd_lower_raw` and `resid_sd_upper_raw` (the sample SD of the Task 1.4 residuals in each group) and `resid_sd_ratio_raw` = upper / lower. Then repeat the closed-form regression of Task 1.4 on the logged variables, ln(value) on ln(X4), and write `loglog_intercept`, `loglog_slope`, `loglog_r2`, `loglog_rmse`. Split the rows of this new model into lower and upper halves by its own fitted values in the same way, and write `resid_sd_lower_log`, `resid_sd_upper_log` and `resid_sd_ratio_log`. (R² values from the two models describe different scales; do not compare them with each other.)

1.7 **Multiple regression by the normal equation.** Regress ln(value) on eight columns in this exact order: a constant column of ones, ln(X4), X1, X2, X3, X5, X6, X7. Compute β̂ = (XᵀX)⁻¹ XᵀY (a linear solve is fine) and write `mr_beta_0` … `mr_beta_7` in that order, `mr_r2`, and `mr_adj_r2` where adjusted R² = 1 − (1 − R²) × (n − 1) / (n − p − 1) and p is the number of predictors excluding the constant (7 here). Also write `loglog_adj_r2`, the same adjusted R² for the Task 1.6 log-log model (p = 1).

**Acceptance criteria**

- `outputs/part1_results.txt` contains every key named above, in the format described.
- The closed-form quantities come from the formulas in the tasks, not from a ready-made regression function (for example `numpy.polyfit`, `numpy.cov`, `numpy.corrcoef` or `numpy.linalg.lstsq`); `numpy.linalg.solve` on your own normal-equation matrices is fine.
- `gd_slope_gap` is the difference between the two ways of fitting the same line, computed by your code.

**README for Part 1** (section `## Part 1 findings`):

- R1.1 State whether H0 was rejected, quoting `z_stat` and the critical value 1.96; define a Type I error for this claim in one sentence in domain terms; and state the size of the gap `mean_value` − 8.0 in the column's units and whether you think it matters in practice (statistical significance is not the same as practical size).
- R1.2 State `n_required` next to the 800 rows available and say what the comparison means for the planning team.
- R1.3 Quote `gd_slope_gap` and explain in one sentence why gradient descent reached the closed-form line.
- R1.4 Quote `resid_sd_ratio_raw` and `resid_sd_ratio_log`; say whether the raw model shows a funnel (the spread of residuals growing with the fitted value) and whether the log model removes it.
- R1.5 Name the predictors in Task 1.7 whose coefficient has an absolute value above 0.05 (excluding the constant), and compare `mr_adj_r2` with `loglog_adj_r2` to say whether the extra predictors earn their place. The coefficients are in different units (one unit of ln(X4), of a 0-to-1 ratio, of a count and of a 0/1 flag are not comparable steps), so the threshold of 0.05 is a mechanical rule, not a ranking of importance: apply it exactly as stated, even to a predictor whose coefficient is small relative to its uncertainty, and do not read the list as the set of predictors that matter.

---

## Part 2 — Classification and evaluation  (30 marks)

Part 2 builds a probability model for the outcome from X1 to X7 and judges it honestly. It produces `outputs/part2_results.txt`, `outputs/part2_thresholds.csv` and `outputs/part2_knn_cv.csv` (script `part2.py`).

**Tasks**

2.1 **Split and standardise.** Make a stratified 70/30 split. Take the rows with outcome 0 in file order; the first ⌊7 × count / 10⌋ of them (integer arithmetic) go to the training set and the rest to the test set; do the same for outcome 1. Both sets keep file order. Standardise X1 to X7 (all seven, including the two 0/1 flags) with the **training** mean and training sample SD, and apply those same numbers to the test rows (the test rows must never influence any statistic). Write `train_rows`, `train_positives`, `test_rows`, `test_positives`. **Checkpoint:** `train_rows = 560`, `train_positives = 119`, `test_rows = 240`, `test_positives = 51`.

2.2 **Logistic regression from scratch.** Model: z = Σ wⱼ xⱼ + b, ŷ = σ(z) = 1 / (1 + e⁻ᶻ). Loss = binary cross-entropy averaged over the training rows: −(1/n) Σ [y ln ŷ + (1 − y) ln(1 − ŷ)]. The gradients are ∂L/∂wⱼ = (1/n) Σ (ŷ − y) xⱼ and ∂L/∂b = (1/n) Σ (ŷ − y). Start from all weights and the bias at 0; use η = 0.1; run **1000 updates**, each on all 560 training rows. Write `loss_epoch_0` (before any update), `loss_epoch_1`, `loss_epoch_10`, `loss_epoch_100`, `loss_epoch_1000` (training loss after that many updates), the final weights `weight_X1` … `weight_X7` and `bias`, and `test_bce` (the same loss on the 240 test rows with the final parameters). **Checkpoint:** `loss_epoch_0 = 0.6931`.

2.3 **Learning-rate effect.** Restart from zero with each η in {0.01, 0.1, 1.0} and run 200 updates. Write `loss_epoch_200_eta_0.01`, `loss_epoch_200_eta_0.1`, `loss_epoch_200_eta_1.0` (training loss after 200 updates).

2.4 **Odds ratios.** For each predictor write `odds_ratio_X1` … `odds_ratio_X7` = e^(weight): the factor by which the odds of the outcome are multiplied when that predictor rises by one training SD, other predictors fixed.

2.5 **Evaluate on the test rows at two thresholds.** Using the Task 2.2 model, for τ = 0.5 and τ = 0.3 write `tau_0.5_tp`, `tau_0.5_fp`, `tau_0.5_tn`, `tau_0.5_fn`, `tau_0.5_accuracy`, `tau_0.5_precision`, `tau_0.5_recall`, `tau_0.5_f1`, `tau_0.5_specificity` and the nine same-named keys for `tau_0.3_…`. Also write `blind_accuracy`: the test accuracy of a "model" that predicts 0 for every record.

2.6 **Threshold sweep.** For τ in 0.1, 0.2, … 0.9 (nine values), compute the test confusion cells and write `outputs/part2_thresholds.csv` with header `tau,tp,fp,tn,fn,tpr,fpr`, one row per τ in ascending order. (These are the points of the ROC curve; no area is required.)

2.7 **K-nearest neighbours with cross-validation.** Implement K-nearest neighbours yourself. Distance is the Euclidean distance on the seven standardised features. The neighbours of a query are ranked by (distance, then position in file order among the reference rows; a stable sort such as `numpy.argsort(..., kind='stable')` keeps that order) and the first K are taken; the query is predicted positive when at least (K + 1) / 2 of the K neighbours are positive (K is odd). Choose K from the odd values 1, 3, 5, …, 21 by 5-fold cross-validation on the 560 training rows: the fold of a training row is its 0-based position in the training set (in file order) modulo 5; for each fold the other four folds are the reference rows; the score of a K is the **mean over the five folds of the F1 score** of the held-out fold. Use the Task 2.1 standardised features throughout (the mean and sample SD of the 560 training rows); do **not** recompute any mean or SD inside a fold. Write `outputs/part2_knn_cv.csv` with header `K,mean_cv_f1` and eleven rows in ascending K. The chosen `knn_best_k` is the K with the highest mean F1 (ties go to the smaller K). Refit with all 560 training rows as reference and write the test results `knn_tp`, `knn_fp`, `knn_tn`, `knn_fn`, `knn_accuracy`, `knn_precision`, `knn_recall`, `knn_f1`.

2.8 **Choose the alert threshold with money.** A reminder call to a borrower who would have paid anyway costs about INR 50; a default that was not flagged costs about INR 400 more to recover. In money terms each false alarm costs INR 50 and each missed positive costs INR 400, so cost = 50 × `fp` + 400 × `fn`. Using the Task 2.2 model's probabilities on the **training** rows, evaluate τ = k / 20 for k = 1 … 19 and take the τ with the lowest training cost (ties go to the smaller τ). Write `cost_tau`, `cost_train_at_tau`. Then, on the **test** rows, write `cost_test_at_tau`, `cost_test_at_0.5` (the same cost at τ = 0.5) and `cost_test_flag_nobody` (the cost of flagging nobody: every positive missed).

**Acceptance criteria**

- `part2_results.txt` has every key named in Tasks 2.1 to 2.8, `part2_thresholds.csv` has 9 rows and `part2_knn_cv.csv` has 11 rows, each with the exact header.
- Every probability, distance and metric comes from code you wrote; the test rows are used only to evaluate, never to fit, standardise, choose K or choose τ.
- Rates in the CSV and results files are consistent with the cell counts next to them.

**README for Part 2** (section `## Part 2 findings`):

- R2.1 Order the three Task 2.3 losses from lowest to highest and say in one sentence what that says about a learning rate that is too small and one that is large.
- R2.2 Name the three predictors with the largest absolute weight and interpret the largest odds ratio in one sentence in domain terms.
- R2.3 Quote `blind_accuracy` next to the τ = 0.5 accuracy and say why accuracy alone is a weak score here; then state, with numbers, what lowering τ from 0.5 to 0.3 did to recall and to precision.
- R2.4 Compare the logistic model at τ = 0.3 with the KNN model on `f1` and `recall`, quoting both.
- R2.5 State which model and threshold you would deploy, supported by `cost_test_at_tau`, `cost_test_at_0.5` and `cost_test_flag_nobody`, and say in one sentence why the threshold was chosen on training rows rather than test rows.

---

## Part 3 — Segmentation and dimensionality reduction  (25 marks)

Part 3 uses all 800 rows and the five continuous predictors X1 to X5. It produces `outputs/part3_results.txt`, `outputs/part3_segments.csv` and `outputs/part3_hierarchical.csv` (script `part3.py`).

**Tasks**

3.1 **K-means from scratch.** Standardise X1 to X5 over all 800 rows (mean and sample SD of each column). For each K = 2, 3, 4, 5, 6 run K-means with these pinned rules: the starting centroids are the standardised rows with 0-based indices 0, ⌊800/K⌋, 2 × ⌊800/K⌋, …, (K − 1) × ⌊800/K⌋; each row is assigned to the nearest centroid by squared Euclidean distance (ties go to the lower-numbered centroid); each centroid is then replaced by the mean of its members (a centroid with no members stays where it is); the loop stops at the first assignment pass in which no row changes cluster (there is no iteration cap: keep going until such a pass occurs; on the reference data K = 4 takes 21 assignment passes, K = 5 takes 15 and K = 6 takes 27, counting the final pass that changes nothing). The within-cluster sum of squares WCSS is the sum, over all rows, of the squared distance to the row's final centroid, measured in the standardised space. Write `wcss_k2` … `wcss_k6`. These pinned starting rows give one reproducible solution that is not necessarily the global optimum for every K (other starting rows can reach a lower WCSS, for example for K = 4 and K = 6, so the successive drops need not shrink steadily); grade and answer with the pinned values. **Checkpoint:** `wcss_k2 = 2262.6753`.

3.2 **Choose K with the elbow.** Write `elbow_k`, defined as the smallest K in {3, 4, 5} for which WCSS(K) − WCSS(K + 1) < 0.5 × (WCSS(K − 1) − WCSS(K)). Whatever `elbow_k` is, use K = 3 for the rest of this Part.

3.3 **Profile the K = 3 segments.** Order the three clusters by ascending mean of X3 (in original units) and call them segments A, B and C. Write `size_segment_A`, `size_segment_B`, `size_segment_C`, and `outputs/part3_segments.csv` with header `segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value` and three rows (A, B, C). The means are in original (unstandardised) units; `outcome_rate` is the share of the segment's records with outcome 1 and `mean_value` the mean of the value column.

3.4 **Hierarchical clustering on a supplied table.** The team compared six borrower cohorts, E1 to E6, and produced this table of pairwise dissimilarity scores (symmetric, zero diagonal):

| | E1 | E2 | E3 | E4 | E5 | E6 |
|---|---|---|---|---|---|---|
| **E1** | 0.0 | 7.9 | 7.4 | 2.9 | 2.9 | 6.1 |
| **E2** | 7.9 | 0.0 | 1.4 | 5.9 | 5.3 | 2.0 |
| **E3** | 7.4 | 1.4 | 0.0 | 5.8 | 4.6 | 1.3 |
| **E4** | 2.9 | 5.9 | 5.8 | 0.0 | 2.8 | 4.5 |
| **E5** | 2.9 | 5.3 | 4.6 | 2.8 | 0.0 | 3.4 |
| **E6** | 6.1 | 2.0 | 1.3 | 4.5 | 3.4 | 0.0 |

Implement agglomerative clustering yourself for three linkage rules: **single** (the smallest distance between a member of one cluster and a member of the other), **complete** (the largest such distance) and **average** (the mean of all such member-to-member distances). At each step merge the two clusters with the smallest linkage distance (no ties occur). Write `outputs/part3_hierarchical.csv` with header `linkage,merge_no,cluster_a,cluster_b,height` and 15 rows (single, complete, average, five merges each, `merge_no` 1 to 5); a cluster is written as its member names in ascending order joined by `+` (for example `E3+E6`), `cluster_a` is the cluster whose smallest member number is the smaller one, and `height` is the linkage distance of the merge. To cut a dendrogram, take the four gaps between consecutive merge heights (merge 2 − merge 1, … merge 5 − merge 4); the cut lies in the largest gap, and the number of clusters is 6 minus the number of merges below the cut. Write `single_clusters_at_cut`, `complete_clusters_at_cut`, `average_clusters_at_cut`.

3.5 **PCA by the seven steps.** On the standardised 800 × 5 matrix of Task 3.1: (1) the mean vector (all zeros after standardising; state this in the README); (2) centre the data; (3) the covariance matrix S = (1/(n − 1)) XᵀX; (4) its eigenvalues and eigenvectors (`numpy.linalg.eigh` is the one library routine allowed here); (5) sort by eigenvalue, largest first; (6) keep the top two eigenvectors; (7) project Z = X W. **Sign convention:** flip each eigenvector so that its entry of largest absolute value is positive. Write `eigenvalue_sum`, `eigenvalue_1` … `eigenvalue_5`, `explained_ratio_1` … `explained_ratio_5` (eigenvalue / sum of eigenvalues), `cumulative_ratio_1` … `cumulative_ratio_5`, `components_for_80pct` (the smallest number of components with cumulative ratio at least 0.8), `pc1_loading_X1` … `pc1_loading_X5`, `pc2_loading_X1` … `pc2_loading_X5`, the projections of the first three data rows `row1_pc1`, `row1_pc2`, `row2_pc1`, `row2_pc2`, `row3_pc1`, `row3_pc2`, and `mean_pc1_segment_A`, `mean_pc1_segment_B`, `mean_pc1_segment_C` (the mean first-component score of the records in each Task 3.3 segment). **Checkpoint:** `eigenvalue_sum = 5.0000`.

**Acceptance criteria**

- Every key of Tasks 3.1 to 3.5 is in `part3_results.txt`; `part3_segments.csv` has 3 rows and `part3_hierarchical.csv` has 15 rows, each with the exact header.
- K-means, the three linkage rules and the seven PCA steps are written by you (apart from the eigen-decomposition routine).
- The segment sizes add up to 800.

**README for Part 3** (section `## Part 3 findings`):

- R3.1 Quote the four successive WCSS drops WCSS(2) − WCSS(3), WCSS(3) − WCSS(4), WCSS(4) − WCSS(5), WCSS(5) − WCSS(6) and say which K the elbow supports and why (if the drops do not shrink steadily, say that the pinned starting rows give a reproducible solution, not necessarily the global optimum).
- R3.2 Give each segment a short business name and one concrete action for borrowers, based on its row in `part3_segments.csv`; name the segment with the highest `outcome_rate` and quote it. Think in terms of risk and usage: who borrows heavily, who is new, who is steady.
- R3.3 State, for single and for complete linkage, the clusters obtained at the cut (list the members), and explain in two sentences why the two rules disagree on this table.
- R3.4 Say which predictors drive the first component (use the signs and sizes of the PC1 loadings), how many components keep at least 80% of the variance, and what `mean_pc1_segment_A/B/C` shows about the segments.

---

## Part 4 — Model-choice and architecture memo  (20 marks)

Part 4 works on the **training rows** and **test rows** of Task 2.1 and uses the models of Part 2. It produces `outputs/part4_results.txt`, `outputs/part4_splits.csv`, `outputs/part4_voting.csv`, `outputs/part4_adaboost.csv` and `outputs/part4_architecture.csv` (script `part4.py`; code shared with Part 2 belongs in a helper file that both scripts import, so that importing it does not re-run a Part).

**Tasks**

4.1 **Which split is purest?** On the 560 **training** rows, in original (unstandardised) units, evaluate four candidate yes/no splits: **S1** X6 = 1, **S2** X7 = 1, **S3** X3 > 0.5, **S4** X1 ≤ 24 (the "yes" side is where the condition holds). For a node with P positives out of n records, entropy = −Σ pᵢ log2 pᵢ over the two classes (0 for a pure node) and Gini = 1 − Σ pᵢ². Information gain = parent entropy − weighted child entropy (children weighted by their share of the records), and Gini gain is the same with Gini. Write `root_entropy` and `root_gini` of the 560 training rows, `outputs/part4_splits.csv` with header `split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain` (rows S1, S2, S3, S4), and `best_split_by_info_gain` and `best_split_by_gini_gain` (each `S1`, `S2`, `S3` or `S4`). **Checkpoint:** `root_entropy = 0.7462`.

4.2 **A one-question model and two ways to combine models.** The stump is the best split by information gain: its predicted probability for a record is the share of positive **training** rows on the same side of the split as the record. Write `stump_rate_yes` and `stump_rate_no`. On the 240 test rows build five predictions: **logistic** (Task 2.2 model, positive when probability ≥ 0.3); **knn** (Task 2.7 model with the chosen K, positive when at least (K + 1) / 2 neighbours are positive; its probability is the share of positive neighbours among the K); **stump** (positive when its probability ≥ 0.3); **hard_vote** (positive when at least two of the three above predict positive); **soft_vote** (positive when the mean of the three probabilities — logistic probability, KNN neighbour share, stump probability — is ≥ 0.3). Write `outputs/part4_voting.csv` with header `model,tp,fp,tn,fn,precision,recall,f1` and five rows in the order logistic, knn, stump, hard_vote, soft_vote.

4.3 **Two rounds of AdaBoost by hand.** Eight past borrowers have true classes y (+1 = outcome occurred, −1 = it did not) and two weak rules, h1 and h2, each predicting +1 or −1:

| Point | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| y | +1 | +1 | +1 | −1 | −1 | −1 | +1 | −1 |
| h1 | +1 | +1 | −1 | −1 | −1 | −1 | −1 | −1 |
| h2 | −1 | +1 | +1 | −1 | +1 | −1 | +1 | −1 |

Start with equal weights 1/8. In round t with weights w: the weighted error ε_t is the sum of the weights of the points where h_t ≠ y; α_t = 0.5 × ln((1 − ε_t) / ε_t); the new unnormalised weight of a point is w × exp(−α_t × y × h_t); the new weights are those divided by their sum, so they sum to 1. Round 1 uses h1 and the equal weights; round 2 uses h2 and the normalised weights from round 1. Write `round1_error`, `round1_alpha`, `round2_error`, `round2_alpha`; `outputs/part4_adaboost.csv` with header `round,point,weight_before,weight_unnormalised,weight_after` and 16 rows (round 1 points 1–8, then round 2 points 1–8); and `ensemble_accuracy`, the share of the eight points classified correctly by sign(α1 × h1 + α2 × h2). **Checkpoint:** `round1_error = 0.2500`.

4.4 **Out-of-bag share.** When a training set of 560 rows is sampled with replacement 560 times, a given row is never picked with probability (1 − 1/560)^560. Write it as `oob_fraction`.

4.5 **Size the neural networks a later project would need.** Nothing is trained here: you count shapes and parameters. A layer's parameter count includes one bias per neuron or filter.

- **Model A — an image-sized input.** This is the main use case. The lending team wants a model that reads the digits written on scanned cheques and forms (an OCR-style digit classifier, digits 0 to 9). The input is a 28 × 28 grayscale array (28 × 28 × 1) and there are 10 classes. Option 1 is a fully connected network 784 → 16 → 16 → 10. Option 2 is a convolutional network: convolution with 5 × 5 filters, 6 filters, stride 1, padding 2; ReLU; max-pool 2 × 2 with stride 2; convolution with 5 × 5 filters, 16 filters, stride 1, padding 0; ReLU; max-pool 2 × 2 with stride 2; flatten; fully connected layer with 32 neurons and ReLU; fully connected output layer with 10 neurons. The output size of a convolution or pooling layer is (input − filter + 2 × padding) / stride + 1 per spatial side. Write `outputs/part4_architecture.csv` with header `layer,parameters` and exactly these seven rows in this order: `mlp_hidden1`, `mlp_hidden2`, `mlp_output`, `cnn_conv1`, `cnn_conv2`, `cnn_dense1`, `cnn_dense2`; and the keys `mlp_total` and `cnn_total` (sums over the respective layers).
- **Why convolution scales.** A fully connected layer takes a flattened 128 × 128 × 3 input into 1,024 neurons; a convolutional layer takes the same 128 × 128 × 3 input with 64 filters of size 3 × 3. Write `fc_layer_params` and `conv_layer_params`.
- **Model B — a token-sequence input.** This is a second use case. The team also wants the free-text remark on a payment sorted into one of 16 categories. A recurrent network reads one 64-dimensional word vector X_t per step, keeps a 128-dimensional hidden state, and after the last step outputs 16 class scores: H_t = tanh(W_xh X_t + W_hh H_{t−1} + b_h) and Y = W_hy H_T + b_y. Write `rnn_total`, the number of parameters in W_xh, W_hh, W_hy, b_h and b_y.

4.6 **Memo to the operations lead.** Write a model-choice memo of at most 300 words under the README heading `## Model-choice memo`, addressed to the head of the lending-risk team. It must recommend one structured-data model from Task 4.2 for production, cite at least three numbers from your own `outputs/` files, state the threshold you recommend and what it costs in INR (always report the cost of the logistic-regression model from Task 2.8 as the comparator; if you recommend a different model, give that model's cost only if you computed it yourself with the Task 2.8 cost rule, and otherwise say that it was not computed), name one risk or limitation of the evidence, and say in one sentence whether Model A or Model B is worth building before the structured model is in production.

**Acceptance criteria**

- Every key and CSV named in Tasks 4.1 to 4.5 is present with the exact header and row count (`part4_splits.csv` 4 rows, `part4_voting.csv` 5, `part4_adaboost.csv` 16, `part4_architecture.csv` 7).
- Entropy, Gini, gains, the vote rules and the boosting arithmetic are your own code. No model is trained for Models A or B.
- The memo is at most 300 words.

**README for Part 4** (section `## Part 4 findings`):

- R4.1 In two or three sentences, say why the best split is the best, and why a tree grown until every leaf is pure would overfit; name one rule that stops a tree from growing too deep.
- R4.2 State which of the five Task 4.2 models you would ship and whether combining models helped, quoting `f1` and `recall` values.
- R4.3 State the total weight of the two misclassified points after round 1 normalisation, and say in one sentence why round 2 trusts h2 more than h1.
- R4.4 A table with one row per layer of Model A option 2 giving the output shape (height × width × depth) and the parameter count; the shapes of W_xh, W_hh and W_hy for Model B and a sentence on why `rnn_total` does not grow when the sequences get longer; the hidden-layer activation and the output activation with loss you would choose for 10 classes; and, in two sentences, why a very deep plain network can stop learning in its early layers and how a skip connection helps.

---

## Submission

Submit **one public GitHub repository link** that holds the whole project and points to the **`main`** branch (`https://github.com/<user>/<repo>` when `main` is the default branch, or `https://github.com/<user>/<repo>/tree/main`). The repository must contain `README.md` (first line `Domain: Banking`), `requirements.txt`, `generate_data.py`, `data/records.csv`, `part1.py` to `part4.py` and the `outputs/` files named above, all committed on `main`. A link to any other branch, or a link that does not clearly resolve to `main`, will not be graded. This one link is the whole deliverable: no other link placed inside it will be opened or graded. The first line of the README (`Domain: <name>`) is the only place the domain is declared; no separate domain entry is needed.

**The README must contain, in this order:** the line `Domain: Banking`; `## How to run` (the exact commands, in order, from a clean clone, including `pip install -r requirements.txt`); `## Part 1 findings`; `## Part 2 findings`; `## Part 3 findings`; `## Part 4 findings`; `## Model-choice memo`; and `## Limitations` (at least three honest limitations of this analysis, for example the synthetic data, a single split, or the small number of positive test records).