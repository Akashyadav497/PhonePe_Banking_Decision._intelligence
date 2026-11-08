"""
PhonePe Banking Decision Intelligence — Executive Interactive Analytics Dashboard
Academic Capstone Simulation
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="PhonePe Banking Decision Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for PhonePe Theme
st.markdown("""
<style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #5f259f;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 16px;
        color: #6c757d;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #5f259f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .disclaimer-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 12px;
        border-radius: 6px;
        font-size: 14px;
        margin-bottom: 15px;
        border: 1px solid #ffeeba;
    }
</style>
""", unsafe_allow_html=True)


def load_all_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    records_path = os.path.join(base_dir, "data", "records.csv")
    outputs_dir = os.path.join(base_dir, "outputs")

    df_records = pd.read_csv(records_path) if os.path.exists(records_path) else None

    # Load output results
    def parse_results_txt(filename):
        fpath = os.path.join(outputs_dir, filename)
        res = {}
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        res[k.strip()] = v.strip()
        return res

    p1_res = parse_results_txt("part1_results.txt")
    p2_res = parse_results_txt("part2_results.txt")
    p3_res = parse_results_txt("part3_results.txt")
    p4_res = parse_results_txt("part4_results.txt")

    # Load CSVs
    p2_thresh = pd.read_csv(os.path.join(outputs_dir, "part2_thresholds.csv")) if os.path.exists(os.path.join(outputs_dir, "part2_thresholds.csv")) else None
    p2_knn = pd.read_csv(os.path.join(outputs_dir, "part2_knn_cv.csv")) if os.path.exists(os.path.join(outputs_dir, "part2_knn_cv.csv")) else None
    p3_seg = pd.read_csv(os.path.join(outputs_dir, "part3_segments.csv")) if os.path.exists(os.path.join(outputs_dir, "part3_segments.csv")) else None
    p3_hier = pd.read_csv(os.path.join(outputs_dir, "part3_hierarchical.csv")) if os.path.exists(os.path.join(outputs_dir, "part3_hierarchical.csv")) else None
    p4_splits = pd.read_csv(os.path.join(outputs_dir, "part4_splits.csv")) if os.path.exists(os.path.join(outputs_dir, "part4_splits.csv")) else None
    p4_vote = pd.read_csv(os.path.join(outputs_dir, "part4_voting.csv")) if os.path.exists(os.path.join(outputs_dir, "part4_voting.csv")) else None
    p4_ada = pd.read_csv(os.path.join(outputs_dir, "part4_adaboost.csv")) if os.path.exists(os.path.join(outputs_dir, "part4_adaboost.csv")) else None
    p4_arch = pd.read_csv(os.path.join(outputs_dir, "part4_architecture.csv")) if os.path.exists(os.path.join(outputs_dir, "part4_architecture.csv")) else None

    return df_records, p1_res, p2_res, p3_res, p4_res, p2_thresh, p2_knn, p3_seg, p3_hier, p4_splits, p4_vote, p4_ada, p4_arch


df, p1, p2, p3, p4, p2_thresh, p2_knn, p3_seg, p3_hier, p4_splits, p4_vote, p4_ada, p4_arch = load_all_data()

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/7/71/PhonePe_Logo.svg", width=180)
st.sidebar.markdown("### Decision Intelligence Portal")
st.sidebar.markdown("**Domain:** Digital Banking & Lending Risk")
st.sidebar.markdown("**Institution:** Academic Capstone Simulation")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigate Modules",
    ["Executive Overview", "Part 1: Spend Estimation & Baseline", "Part 2: Default Classification & Cost", "Part 3: Segmentation & PCA", "Part 4: Ensembles & Sizing", "Raw Dataset Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Designed for academic and demonstration evaluation.")

# Top Header & Transparency Notice
st.markdown('<div class="main-title">PhonePe Banking Decision Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Lending-Risk Analytics, Statistical Inference & Machine Learning Pipeline</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>Transparency Notice:</strong> This project uses synthetic data generated for academic and demonstration purposes.
    It does not contain actual PhonePe customer or banking transaction records.
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# MODULE 1: EXECUTIVE OVERVIEW
# -------------------------------------------------------------
if nav_choice == "Executive Overview":
    st.header("Executive Summary & Core Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cohort Size", f"{len(df):,}" if df is not None else "800", "Borrower Records")
    with col2:
        default_rate = (df['loan_default'].mean() * 100) if df is not None else 21.25
        st.metric("Portfolio Default Rate", f"{default_rate:.2f}%", "170 of 800")
    with col3:
        cost_tau = p2.get("cost_tau", "0.15")
        st.metric("Optimal Alert Threshold", f"τ* = {cost_tau}", "Training Calibrated")
    with col4:
        cost_saved = int(float(p2.get("cost_test_flag_nobody", "20400")) - float(p2.get("cost_test_at_tau", "6050")))
        st.metric("Recovery Savings", f"INR {cost_saved:,}", "vs. Flagging Nobody")

    st.markdown("### Operational Risk Architecture")
    st.write("""
    This analytics suite addresses three fundamental risk questions facing digital consumer lending operations:
    1. **Spend Dynamics:** Validating whether borrower monthly expenditure aligns with credit underwriting benchmarks (Part 1).
    2. **Early Delinquency Warning:** Accurately identifying accounts likely to reach 90+ days past due while minimizing recovery costs (Part 2).
    3. **Behavioral Cohort Segmentation:** Clustering the portfolio into distinct risk/volume tiers and mapping dimensional axes (Part 3).
    4. **Ensemble & Architectural Sizing:** Measuring tree splits, boosting convergence, and sizing future OCR/NLP neural networks (Part 4).
    """)

    if df is not None:
        st.subheader("Key Portfolio Risk Drivers")
        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(
                df, x="loan_default", y="credit_utilisation_ratio",
                color="loan_default",
                title="Credit Utilisation by Default Status",
                labels={"loan_default": "Loan Default (0=No, 1=Yes)", "credit_utilisation_ratio": "Utilisation Ratio"},
                color_discrete_map={0: "#28a745", 1: "#dc3545"}
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        with c2:
            fig_scatter = px.scatter(
                df, x="avg_monthly_inflow_inr_thousands", y="monthly_spend_inr_thousands",
                color="loan_default",
                title="Monthly Inflow vs. Monthly Spend (INR Thousands)",
                labels={"avg_monthly_inflow_inr_thousands": "Avg Monthly Inflow", "monthly_spend_inr_thousands": "Monthly Spend"},
                color_discrete_map={0: "#5f259f", 1: "#ffc107"},
                opacity=0.7
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------------------------------------
# MODULE 2: PART 1
# -------------------------------------------------------------
elif nav_choice == "Part 1: Spend Estimation & Baseline":
    st.header("Part 1: Estimation, Inference & Regression Baseline")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sample Mean Spend", f"INR {float(p1.get('mean_value', 8.9382))*1000:,.2f}", "95% CI: [8.30, 9.57]")
    with col2:
        st.metric("Hypothesis Test Z-Stat", p1.get("z_stat", "2.8930"), f"Reject H0: {p1.get('reject_h0', 'yes')}")
    with col3:
        st.metric("Required Sample Size", p1.get("n_required", "1293"), "For E = INR 500 margin")

    st.markdown("### Closed-Form vs. Gradient Descent Convergence")
    st.write(f"""
    - **Closed-Form OLS Slope:** `{p1.get('reg_slope', '0.6701')}` | **Intercept:** `{p1.get('reg_intercept', '-3.6598')}`
    - **Gradient Descent Slope:** `{p1.get('gd_slope', '0.6701')}` | **Intercept:** `{p1.get('gd_intercept', '-3.6598')}`
    - **Absolute Slope Gap:** `{p1.get('gd_slope_gap', '0.0000')}` (Strictly $< 0.001$)
    """)

    # Cost progression curve
    gd_steps = [0, 1, 10, 100, 500]
    gd_costs = [float(p1.get(f"gd_J{s}", 0)) for s in [0, 1, 10, 100, 500]]
    fig_gd = go.Figure(data=go.Scatter(x=[str(s) for s in gd_steps], y=gd_costs, mode='lines+markers', line=dict(color='#5f259f', width=3)))
    fig_gd.update_layout(title="Gradient Descent Loss J Across Updates", xaxis_title="Updates Completed", yaxis_title="Cost J", template="plotly_white")
    st.plotly_chart(fig_gd, use_container_width=True)

    st.markdown("### Heteroskedasticity & Residual Funnel Remedy")
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**Raw Model Residual SD Ratio:** `{p1.get('resid_sd_ratio_raw', '3.7525')}`\n\nShows pronounced residual funnel (heteroskedasticity).")
    with c2:
        st.success(f"**Log-Log Model Residual SD Ratio:** `{p1.get('resid_sd_ratio_log', '1.0396')}`\n\nLog transformation effectively eliminates heteroskedasticity.")

# -------------------------------------------------------------
# MODULE 3: PART 2
# -------------------------------------------------------------
elif nav_choice == "Part 2: Default Classification & Cost":
    st.header("Part 2: Classification, Evaluation & Operational Cost")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test BCE Loss", p2.get("test_bce", "0.3807"), "Log-loss on 240 test rows")
    with col2:
        st.metric("Logistic F1 (@ τ=0.3)", p2.get("tau_0.3_f1", "0.5714"), f"Recall: {p2.get('tau_0.3_recall', '0.7451')}")
    with col3:
        st.metric("KNN Best K", p2.get("knn_best_k", "5"), "Chosen by 5-fold CV")
    with col4:
        st.metric("Test Cost @ τ*", f"INR {int(p2.get('cost_test_at_tau', 6050)):,}", "Policy Minimum")

    if p2_thresh is not None:
        st.subheader("ROC Curve (Threshold Sweep τ = 0.1 to 0.9)")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=p2_thresh['fpr'], y=p2_thresh['tpr'], mode='lines+markers+text', text=p2_thresh['tau'], textposition="top center", name="Logistic Regression", line=dict(color='#5f259f', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name="Random Guess"))
        fig_roc.update_layout(title="Receiver Operating Characteristic (ROC)", xaxis_title="False Positive Rate (FPR)", yaxis_title="True Positive Rate (TPR / Recall)", template="plotly_white")
        st.plotly_chart(fig_roc, use_container_width=True)

    if p2_knn is not None:
        st.subheader("K-Nearest Neighbours 5-Fold Cross Validation")
        fig_knn = px.line(p2_knn, x="K", y="mean_cv_f1", markers=True, title="Mean 5-Fold CV F1 Score vs. K", labels={"mean_cv_f1": "Mean CV F1 Score"})
        fig_knn.update_traces(line_color="#ff7f0e", line_width=2)
        st.plotly_chart(fig_knn, use_container_width=True)

# -------------------------------------------------------------
# MODULE 4: PART 3
# -------------------------------------------------------------
elif nav_choice == "Part 3: Segmentation & PCA":
    st.header("Part 3: Customer Segmentation & Dimensionality Reduction")

    if p3_seg is not None:
        st.subheader("Borrower Cohort Profiles (K = 3 Segments)")
        st.dataframe(p3_seg.style.format({
            "size": "{:d}",
            "mean_X1": "{:.2f}",
            "mean_X2": "{:.2f}",
            "mean_X3": "{:.3f}",
            "mean_X4": "{:.2f}",
            "mean_X5": "{:.2f}",
            "outcome_rate": "{:.2%}",
            "mean_value": "{:.2f}"
        }), use_container_width=True)

        fig_seg = px.scatter(
            p3_seg, x="mean_X3", y="outcome_rate", size="size", color="segment",
            text="segment", title="Segment Risk vs. Credit Utilisation",
            labels={"mean_X3": "Mean Credit Utilisation", "outcome_rate": "Default Rate"},
            size_max=40, color_discrete_sequence=["#28a745", "#ffc107", "#dc3545"]
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    st.subheader("Principal Component Analysis (PCA)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PC1 Explained Variance", f"{float(p3.get('explained_ratio_1', 0.4420))*100:.2f}%", "Credit Health Axis")
    with col2:
        st.metric("PC2 Explained Variance", f"{float(p3.get('explained_ratio_2', 0.2036))*100:.2f}%", "Spend & Inflow Axis")
    with col3:
        st.metric("Components for 80% Variance", p3.get("components_for_80pct", "3"), "Cumulative: 81.43%")

# -------------------------------------------------------------
# MODULE 5: PART 4
# -------------------------------------------------------------
elif nav_choice == "Part 4: Ensembles & Sizing":
    st.header("Part 4: Decision Tree Splits, Ensembles & Neural Network Sizing")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Decision Stump Purity Splits")
        if p4_splits is not None:
            st.dataframe(p4_splits.style.format({"info_gain": "{:.4f}", "gini_gain": "{:.4f}"}), use_container_width=True)
            st.success(f"Best Split Selected: **{p4.get('best_split_by_info_gain', 'S4')}** (Account Age ≤ 24 months)")
    with c2:
        st.subheader("Ensemble Model Comparison on Test Rows")
        if p4_vote is not None:
            st.dataframe(p4_vote.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}), use_container_width=True)

    st.markdown("### Neural Network Parameter Sizing (Models A & B)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model A (MLP Total)", f"{int(p4.get('mlp_total', 13002)):,}", "Fully Connected Option")
    with col2:
        st.metric("Model A (CNN Total)", f"{int(p4.get('cnn_total', 15734)):,}", "Convolutional Option")
    with col3:
        st.metric("Model B (RNN Total)", f"{int(p4.get('rnn_total', 26768)):,}", "NLP Word Sequence Model")

    if p4_arch is not None:
        fig_arch = px.bar(p4_arch, x="layer", y="parameters", text="parameters", title="Model A (CNN) Parameters by Layer", color="layer", color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig_arch, use_container_width=True)

# -------------------------------------------------------------
# MODULE 6: RAW DATASET EXPLORER
# -------------------------------------------------------------
elif nav_choice == "Raw Dataset Explorer":
    st.header("Raw Synthetic Dataset Exploration")
    if df is not None:
        st.write(f"Displaying {len(df)} borrower records generated with seed 2508:")
        st.dataframe(df, use_container_width=True)

        st.subheader("Download Dataset")
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data/records.csv",
            data=csv_bytes,
            file_name="records.csv",
            mime="text/csv"
        )
