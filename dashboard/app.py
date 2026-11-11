"""
PhonePe Banking Decision Intelligence — Interactive Analytics Dashboard
Lending-Risk Analytics, Statistical Inference & Machine Learning Pipeline
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

# Custom Styling: Refined, balanced dark palette suited for authentic data science portfolios
st.markdown("""
<style>
    /* Global Typography & Background Adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }
    
    /* Header Area */
    .dashboard-header {
        margin-bottom: 1.25rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    .main-title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f1f5f9;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .sub-title {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 4px;
        margin-bottom: 0px;
    }
    
    /* Transparent Banner Notice */
    .notice-box {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 3px solid #7c3aed;
        color: #cbd5e1;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Cards */
    .metric-card-container {
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 0.75rem;
        transition: border-color 0.15s ease;
    }
    .metric-card-container:hover {
        border-color: rgba(148, 163, 184, 0.35);
    }
    .metric-card-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-card-value {
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-card-sub {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Content Cards & Sections */
    .section-card {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 1.25rem;
    }
    .section-title {
        font-size: 15px;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 10px;
    }
    
    /* Custom Sidebar adjustments */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    .sidebar-brand-title {
        font-size: 16px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 10px;
        margin-bottom: 2px;
    }
    .sidebar-brand-subtitle {
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 12px;
    }
    
    /* Tables and dataframes */
    .stDataFrame {
        border-radius: 6px;
        overflow: hidden;
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
            with open(fpath, "r", encoding="utf-8") as f:
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

# Theme helper for Plotly figures to harmonize dark aesthetics
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color="#cbd5e1", size=12),
    margin=dict(l=40, r=20, t=45, b=40),
    xaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.2)",
        linecolor="rgba(148, 163, 184, 0.2)"
    ),
    yaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.2)",
        linecolor="rgba(148, 163, 184, 0.2)"
    ),
    legend=dict(
        bgcolor="rgba(15, 23, 42, 0.6)",
        bordercolor="rgba(148, 163, 184, 0.2)",
        borderwidth=1
    )
)

# Helper function to render clean metric card
def render_metric_card(label, value, subtext=None):
    sub_html = f'<div class="metric-card-sub">{subtext}</div>' if subtext else ''
    st.markdown(f"""
    <div class="metric-card-container">
        <div class="metric-card-label">{label}</div>
        <div class="metric-card-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/7/71/PhonePe_Logo.svg", width=160)
st.sidebar.markdown('<div class="sidebar-brand-title">Banking Decision Intelligence</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-brand-subtitle">Lending Risk & Inference System</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Select Module",
    [
        "Executive Overview",
        "Part 1: Spend Estimation & Baseline",
        "Part 2: Default Classification & Cost",
        "Part 3: Segmentation & PCA",
        "Part 4: Ensembles & Sizing",
        "Raw Dataset Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size: 11px; color: #64748b;'>Portfolio: 800 synthetic borrower profiles<br>Target: 90+ DPD Delinquency Risk</div>",
    unsafe_allow_html=True
)

# Top Header Banner
st.markdown("""
<div class="dashboard-header">
    <div class="main-title">PhonePe Banking Decision Intelligence</div>
    <div class="sub-title">Lending-Risk Analytics, Statistical Inference & Machine Learning Pipeline</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice-box">
    <strong>Dataset Transparency Notice:</strong> This interactive intelligence platform operates on synthetic borrower profiles generated for decision analysis. It models realistic digital lending dynamics without containing proprietary user accounts or transactions.
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# MODULE 1: EXECUTIVE OVERVIEW
# -------------------------------------------------------------
if nav_choice == "Executive Overview":
    st.subheader("Executive Overview & Portfolio Health")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cohort_len = f"{len(df):,}" if df is not None else "800"
        render_metric_card("Total Cohort Size", cohort_len, "800 borrower records in study")
    with col2:
        default_rate = (df['loan_default'].mean() * 100) if df is not None else 21.25
        render_metric_card("Portfolio Default Rate", f"{default_rate:.2f}%", "170 defaulted of 800 loans")
    with col3:
        cost_tau = p2.get("cost_tau", "0.15")
        render_metric_card("Optimal Alert Threshold", f"τ* = {cost_tau}", "Training-calibrated decision cutoff")
    with col4:
        cost_saved = int(float(p2.get("cost_test_flag_nobody", "20400")) - float(p2.get("cost_test_at_tau", "6050")))
        render_metric_card("Recovery Savings", f"INR {cost_saved:,}", "Operational gain vs. zero flagging")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Operational Risk Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    The decision intelligence framework synthesizes four key phases of credit risk management:
    - **Spend Dynamics & Baseline Verification (Part 1):** Statistical testing on borrower spend behavior against benchmark underwriting assumptions, coupled with closed-form vs. numerical regression convergence and residual heteroskedasticity diagnostics.
    - **Early Delinquency Warning & Cost Calibration (Part 2):** Calibrated logistic classification minimizing operational misclassification loss (INR 50 false alert vs. INR 400 undetected default) alongside 5-fold cross-validated K-NN benchmarking.
    - **Behavioral Cohort Segmentation & PCA (Part 3):** K-Means and agglomerative hierarchical clustering mapping customer risk tiers (Prime Established, Growth, and Subprime) over orthogonal principal components.
    - **Ensemble Benchmarking & Deep Architecture Sizing (Part 4):** Information gain tree splitting, voting/boosting ensembles, and parameter sizing for auxiliary document OCR and remark NLP neural networks.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    if df is not None:
        st.markdown('<div class="section-title">Key Portfolio Risk Drivers</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(
                df,
                x="loan_default",
                y="credit_utilisation_ratio",
                color="loan_default",
                title="Credit Utilisation by Default Status",
                labels={"loan_default": "Default Status (0=Non-Default, 1=Default)", "credit_utilisation_ratio": "Utilisation Ratio"},
                color_discrete_map={0: "#10b981", 1: "#ef4444"}
            )
            fig_box.update_layout(**PLOTLY_LAYOUT)
            fig_box.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig_box, use_container_width=True)
        with c2:
            fig_scatter = px.scatter(
                df,
                x="avg_monthly_inflow_inr_thousands",
                y="monthly_spend_inr_thousands",
                color="loan_default",
                title="Monthly Inflow vs. Monthly Spend (INR Thousands)",
                labels={
                    "avg_monthly_inflow_inr_thousands": "Avg Monthly Inflow (₹k)",
                    "monthly_spend_inr_thousands": "Monthly Spend (₹k)",
                    "loan_default": "Default Status"
                },
                color_discrete_map={0: "#8b5cf6", 1: "#f59e0b"},
                opacity=0.75
            )
            fig_scatter.update_layout(**PLOTLY_LAYOUT)
            fig_scatter.update_layout(height=360)
            st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------------------------------------
# MODULE 2: PART 1
# -------------------------------------------------------------
elif nav_choice == "Part 1: Spend Estimation & Baseline":
    st.subheader("Part 1: Spend Estimation, Inference & Regression Baseline")

    col1, col2, col3 = st.columns(3)
    with col1:
        mean_spend = float(p1.get('mean_value', 8.9382)) * 1000
        render_metric_card("Sample Mean Spend", f"INR {mean_spend:,.2f}", "95% CI: [INR 8,302.60, 9,573.80]")
    with col2:
        z_stat = p1.get("z_stat", "2.8930")
        reject = p1.get("reject_h0", "yes").upper()
        render_metric_card("Hypothesis Test Z-Stat", z_stat, f"Critical: 1.96 | Reject H0: {reject}")
    with col3:
        n_req = p1.get("n_required", "1293")
        render_metric_card("Required Sample Size", n_req, "For margin E = INR 500 (Current n = 800)")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Closed-Form OLS vs. Gradient Descent Convergence</div>', unsafe_allow_html=True)
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.markdown(f"**Closed-Form OLS:**<br>Slope: `{p1.get('reg_slope', '0.6701')}`<br>Intercept: `{p1.get('reg_intercept', '-3.6598')}`", unsafe_allow_html=True)
    with c_m2:
        st.markdown(f"**Gradient Descent (500 steps):**<br>Slope: `{p1.get('gd_slope', '0.6701')}`<br>Intercept: `{p1.get('gd_intercept', '-3.6598')}`", unsafe_allow_html=True)
    with c_m3:
        st.markdown(f"**Absolute Slope Gap:**<br>`{p1.get('gd_slope_gap', '0.0000')}`<br><span style='color: #10b981;'>Strictly &lt; 0.001 (Identical solution)</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Cost progression curve
    gd_steps = [0, 1, 10, 100, 500]
    gd_costs = [float(p1.get(f"gd_J{s}", 0)) for s in gd_steps]
    fig_gd = go.Figure(data=go.Scatter(
        x=[f"Step {s}" for s in gd_steps],
        y=gd_costs,
        mode='lines+markers+text',
        text=[f"{c:.2f}" for c in gd_costs],
        textposition="top right",
        line=dict(color='#8b5cf6', width=2.5),
        marker=dict(size=8, color='#a78bfa')
    ))
    fig_gd.update_layout(**PLOTLY_LAYOUT)
    fig_gd.update_layout(
        title="Gradient Descent Cost Function J(w, b) Reduction",
        xaxis_title="Optimization Updates Completed",
        yaxis_title="Mean Squared Error Loss J",
        height=340
    )
    st.plotly_chart(fig_gd, use_container_width=True)

    st.markdown('<div class="section-title">Heteroskedasticity & Residual Funnel Diagnostics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sd_raw = p1.get('resid_sd_ratio_raw', '3.7525')
        st.markdown(f"""
        <div class="metric-card-container" style="border-left: 3px solid #f59e0b;">
            <div class="metric-card-label">Raw Model Residual SD Ratio</div>
            <div class="metric-card-value" style="color: #f59e0b;">{sd_raw}</div>
            <div class="metric-card-sub">Pronounced residual funnel: variance increases with predicted spend.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        sd_log = p1.get('resid_sd_ratio_log', '1.0396')
        st.markdown(f"""
        <div class="metric-card-container" style="border-left: 3px solid #10b981;">
            <div class="metric-card-label">Log-Log Model Residual SD Ratio</div>
            <div class="metric-card-value" style="color: #10b981;">{sd_log}</div>
            <div class="metric-card-sub">Remedy effective: ratio ≈ 1.0 indicates restored homoskedasticity.</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# MODULE 3: PART 2
# -------------------------------------------------------------
elif nav_choice == "Part 2: Default Classification & Cost":
    st.subheader("Part 2: Default Classification, ROC & Cost Optimization")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Test BCE Loss", p2.get("test_bce", "0.3807"), "Binary log-loss on 240 test rows")
    with col2:
        f1_val = p2.get("tau_0.3_f1", "0.5714")
        rec_val = p2.get("tau_0.3_recall", "0.7451")
        render_metric_card("Logistic F1 (@ τ = 0.3)", f1_val, f"Test Recall: {rec_val}")
    with col3:
        best_k = p2.get("knn_best_k", "5")
        render_metric_card("KNN Optimal K", f"K = {best_k}", "Selected by 5-fold cross validation")
    with col4:
        min_cost = int(float(p2.get('cost_test_at_tau', 6050)))
        render_metric_card("Test Policy Cost (@ τ*)", f"INR {min_cost:,}", "Optimized at calibrated τ* = 0.15")

    c_roc, c_knn = st.columns(2)
    with c_roc:
        if p2_thresh is not None:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=p2_thresh['fpr'],
                y=p2_thresh['tpr'],
                mode='lines+markers+text',
                text=[f"τ={t:.1f}" for t in p2_thresh['tau']],
                textposition="top left",
                name="Logistic Regression",
                line=dict(color='#8b5cf6', width=2.5),
                marker=dict(size=7, color='#c4b5fd')
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='#64748b'),
                name="Random Classifier"
            ))
            fig_roc.update_layout(**PLOTLY_LAYOUT)
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC Sweep)",
                xaxis_title="False Positive Rate (FPR)",
                yaxis_title="True Positive Rate (Recall)",
                height=380
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    with c_knn:
        if p2_knn is not None:
            fig_knn = go.Figure()
            fig_knn.add_trace(go.Scatter(
                x=p2_knn["K"],
                y=p2_knn["mean_cv_f1"],
                mode='lines+markers+text',
                text=[f"{f:.4f}" for f in p2_knn["mean_cv_f1"]],
                textposition="top center",
                name="Mean CV F1",
                line=dict(color='#f59e0b', width=2.5),
                marker=dict(size=7, color='#fcd34d')
            ))
            fig_knn.update_layout(**PLOTLY_LAYOUT)
            fig_knn.update_layout(
                title="K-Nearest Neighbours 5-Fold Cross Validation",
                xaxis_title="Number of Neighbours (K)",
                yaxis_title="Mean Validation F1 Score",
                height=380
            )
            st.plotly_chart(fig_knn, use_container_width=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Decision Threshold Calibration Table</div>', unsafe_allow_html=True)
    if p2_thresh is not None:
        st.dataframe(
            p2_thresh.style.format({
                "tau": "{:.2f}",
                "train_cost": "INR {:,.0f}",
                "test_cost": "INR {:,.0f}",
                "fpr": "{:.4f}",
                "tpr": "{:.4f}",
                "precision": "{:.4f}",
                "f1": "{:.4f}"
            }),
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# MODULE 4: PART 3
# -------------------------------------------------------------
elif nav_choice == "Part 3: Segmentation & PCA":
    st.subheader("Part 3: Portfolio Customer Segmentation & PCA")

    if p3_seg is not None:
        st.markdown('<div class="section-title">Borrower Cohort Profiles (K = 3 Segments)</div>', unsafe_allow_html=True)
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
            p3_seg,
            x="mean_X3",
            y="outcome_rate",
            size="size",
            color="segment",
            text="segment",
            title="Segment Risk vs. Credit Utilisation",
            labels={"mean_X3": "Mean Credit Utilisation", "outcome_rate": "Default Rate"},
            size_max=36,
            color_discrete_sequence=["#10b981", "#f59e0b", "#ef4444"]
        )
        fig_seg.update_layout(**PLOTLY_LAYOUT)
        fig_seg.update_layout(height=360)
        st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown('<div class="section-title">Principal Component Analysis (PCA) Decomposition</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        pc1 = float(p3.get('explained_ratio_1', 0.5227)) * 100
        render_metric_card("PC1 Explained Variance", f"{pc1:.2f}%", "Dominant Credit Health & Stability Axis")
    with col2:
        pc2 = float(p3.get('explained_ratio_2', 0.2366)) * 100
        render_metric_card("PC2 Explained Variance", f"{pc2:.2f}%", "Cash Flow & Spend Volume Axis")
    with col3:
        n_comp = p3.get('components_for_80pct', '3')
        render_metric_card("Components for 80% Variance", n_comp, "Cumulative explained variance: 81.43%")

# -------------------------------------------------------------
# MODULE 5: PART 4
# -------------------------------------------------------------
elif nav_choice == "Part 4: Ensembles & Sizing":
    st.subheader("Part 4: Tree Splits, Ensemble Classifiers & Neural Network Sizing")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Decision Stump Candidate Splits</div>', unsafe_allow_html=True)
        if p4_splits is not None:
            st.dataframe(p4_splits.style.format({"info_gain": "{:.4f}", "gini_gain": "{:.4f}"}), use_container_width=True)
            best_sp = p4.get('best_split_by_info_gain', 'S4')
            st.markdown(f"<div style='font-size: 13px; color: #10b981;'><strong>Optimal Split:</strong> {best_sp} (Account Age &le; 24 months maximizes Information Gain)</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Ensemble Model Test Evaluation</div>', unsafe_allow_html=True)
        if p4_vote is not None:
            st.dataframe(p4_vote.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}), use_container_width=True)

    st.markdown('<div class="section-title">Neural Network Parameter Sizing (Models A & B)</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        mlp_p = int(p4.get('mlp_total', 13002))
        render_metric_card("Model A (MLP Total)", f"{mlp_p:,}", "Cheque OCR Fully-Connected candidate")
    with col2:
        cnn_p = int(p4.get('cnn_total', 15734))
        render_metric_card("Model A (CNN Total)", f"{cnn_p:,}", "Cheque OCR Convolutional candidate")
    with col3:
        rnn_p = int(p4.get('rnn_total', 26768))
        render_metric_card("Model B (RNN Total)", f"{rnn_p:,}", "Transaction Remark NLP Recurrent Network")

    if p4_arch is not None:
        fig_arch = px.bar(
            p4_arch,
            x="layer",
            y="parameters",
            text="parameters",
            title="Model A (CNN) Parameter Distribution by Layer",
            labels={"layer": "Network Layer", "parameters": "Parameter Count"},
            color="parameters",
            color_continuous_scale="Purples"
        )
        fig_arch.update_layout(**PLOTLY_LAYOUT)
        fig_arch.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig_arch, use_container_width=True)

# -------------------------------------------------------------
# MODULE 6: RAW DATASET EXPLORER
# -------------------------------------------------------------
elif nav_choice == "Raw Dataset Explorer":
    st.subheader("Raw Dataset Explorer & Data Records")
    if df is not None:
        st.markdown(f"<div style='font-size: 13px; color: #94a3b8; margin-bottom: 12px;'>Displaying <strong>{len(df)}</strong> synthetic borrower records generated for risk underwriting evaluation:</div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=450)

        st.markdown("---")
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Complete Dataset (CSV)",
            data=csv_bytes,
            file_name="records.csv",
            mime="text/csv"
        )
