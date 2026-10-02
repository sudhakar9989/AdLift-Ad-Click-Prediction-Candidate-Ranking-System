"""Interactive Streamlit dashboard for AdLift: Ad Click Prediction & Candidate Ranking.

Demonstrates model evaluation benchmarks, probability calibration analysis,
interactive single-impression scoring, and candidate ad ranking.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from adlift.features import (
    APPROVED_CATEGORICAL_FEATURES,
    APPROVED_INPUT_FEATURES,
    APPROVED_NUMERIC_FEATURES,
)
from adlift.ranking import OFFLINE_RANKING_DISCLAIMER, rank_candidates

# Model paths
LIGHTGBM_PATH = os.path.join("models", "lightgbm_model.joblib")
LOGISTIC_PATH = os.path.join("models", "logistic_regression.joblib")
BASELINE_PATH = os.path.join("models", "constant_baseline.joblib")
PYTORCH_PATH = os.path.join("models", "pytorch_click_model.pt")

CANDIDATES_JSON_PATH = os.path.join("data", "demo", "synthetic_candidates.json")
COMPARISON_JSON_PATH = os.path.join("reports", "model_comparison.json")
CALIBRATION_JSON_PATH = os.path.join("reports", "calibration_and_error_analysis.json")

FIGURE_CALIBRATION_PATH = os.path.join("reports", "figures", "calibration_curve.png")
FIGURE_DISTRIBUTION_PATH = os.path.join("reports", "figures", "probability_distribution.png")
FIGURE_BUCKETS_PATH = os.path.join("reports", "figures", "score_buckets.png")


@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> Any:
    """Load a model artifact with caching."""
    if model_name == "LightGBM":
        if not os.path.exists(LIGHTGBM_PATH):
            st.error(f"LightGBM artifact not found at {LIGHTGBM_PATH}")
            return None
        return joblib.load(LIGHTGBM_PATH)
    elif model_name == "Logistic Regression":
        if not os.path.exists(LOGISTIC_PATH):
            st.error(f"Logistic Regression artifact not found at {LOGISTIC_PATH}")
            return None
        return joblib.load(LOGISTIC_PATH)
    elif model_name == "Constant Baseline":
        if not os.path.exists(BASELINE_PATH):
            st.error(f"Baseline artifact not found at {BASELINE_PATH}")
            return None
        return joblib.load(BASELINE_PATH)
    elif model_name == "PyTorch Embedding-MLP":
        if not os.path.exists(PYTORCH_PATH):
            st.error(f"PyTorch artifact not found at {PYTORCH_PATH}")
            return None
        from adlift.models.pytorch_model import PyTorchClickModel
        return PyTorchClickModel.load(PYTORCH_PATH)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


@st.cache_data(show_spinner=False)
def load_candidates() -> List[Dict[str, Any]]:
    """Load synthetic candidates from demo json file."""
    if os.path.exists(CANDIDATES_JSON_PATH):
        with open(CANDIDATES_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data(show_spinner=False)
def load_comparison_data() -> Optional[Dict[str, Any]]:
    """Load model comparison metrics report."""
    if os.path.exists(COMPARISON_JSON_PATH):
        with open(COMPARISON_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_calibration_data() -> Optional[Dict[str, Any]]:
    """Load calibration and failure analysis report."""
    if os.path.exists(CALIBRATION_JSON_PATH):
        with open(CALIBRATION_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def format_candidate_for_scoring(
    cpo: float,
    time_since_last_click: int,
    campaign: int,
    cat_values: Dict[str, int],
) -> Dict[str, Any]:
    """Helper to assemble a single candidate dictionary for model inference."""
    data = {
        "cpo": float(cpo),
        "time_since_last_click": int(time_since_last_click),
        "campaign": int(campaign),
    }
    for col in APPROVED_CATEGORICAL_FEATURES:
        if col != "campaign":
            data[col] = int(cat_values.get(col, 29196072))
    return data


def render_disclaimer_banner() -> None:
    """Render the observational ranking disclaimer banner prominently."""
    st.warning(
        f"⚠️ **DEMO NOTICE & OBSERVATIONAL CAUSALITY DISCLAIMER**\n\n"
        f"{OFFLINE_RANKING_DISCLAIMER}\n\n"
        f"*(All candidate items in this ranking demonstration are marked `is_synthetic = True`)*",
        icon="ℹ️",
    )


def render_ranking_tab(model: Any, model_name: str) -> None:
    """Render the candidate ad ranking demonstration tab."""
    st.subheader("🎯 Live Candidate Ad Ranking Demonstration")
    st.markdown(
        "Sort a set of ad impression candidates in descending order of predicted "
        "click probability (**pCTR**) with deterministic tie-breaking."
    )

    candidates = load_candidates()
    if not candidates:
        st.error("No synthetic candidates found in `data/demo/synthetic_candidates.json`.")
        return

    st.markdown(f"**Loaded Candidate Pool:** `{len(candidates)} candidate ads` | **Active Model:** `{model_name}`")

    with st.expander("🔍 View & Edit Raw Candidate Impressions", expanded=False):
        table_rows = []
        for c in candidates:
            recency_str = "First Visit (-1)" if c["time_since_last_click"] == -1 else f"{c['time_since_last_click']}s ({c['time_since_last_click'] // 60}m)"
            table_rows.append({
                "Candidate ID": c["candidate_id"],
                "Creative Theme": c["creative_theme"],
                "Bid (CPO)": f"${c['cpo']:.2f}",
                "User Recency": recency_str,
                "Campaign": c["campaign"],
                "Is Synthetic": c.get("is_synthetic", True),
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        rank_button = st.button("🚀 Rank Candidate Ads", type="primary", use_container_width=True)

    if rank_button or True:  # Default to display ranked view on load
        if model is None:
            st.error("Selected model artifact is not available. Please train models first.")
            return

        ranked_result = rank_candidates(candidates, model=model)
        ranked_list = ranked_result["ranked_candidates"]

        st.markdown("### 🏆 Deterministically Ranked Results")

        display_data = []
        for item in ranked_list:
            pctr = item["predicted_click_probability"]
            bid = item["features"]["cpo"]
            ecpm = pctr * bid * 1000.0  # standard eCPM formula: pCTR * Bid * 1000

            rec = item["features"]["time_since_last_click"]
            rec_display = "First Visit (-1)" if rec == -1 else f"{rec}s ({rec // 60} min ago)"

            display_data.append({
                "Rank": f"#{item['rank']}",
                "Candidate ID": item["candidate_id"],
                "Creative Theme": item["creative_theme"],
                "pCTR (Click Prob)": f"{pctr * 100:.2f}%",
                "Bid CPO": f"${bid:.2f}",
                "Implied eCPM": f"${ecpm:.2f}",
                "User Recency": rec_display,
                "Synthetic Flag": "SYNTHETIC" if item.get("is_synthetic", True) else "OBSERVED",
            })

        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Visual chart of pCTR distribution across ranked candidates
        chart_df = pd.DataFrame({
            "Candidate": [f"#{item['rank']} {item['candidate_id']}" for item in ranked_list],
            "pCTR (%)": [item["predicted_click_probability"] * 100 for item in ranked_list],
            "Bid CPO ($)": [item["features"]["cpo"] for item in ranked_list],
        }).set_index("Candidate")

        st.bar_chart(chart_df["pCTR (%)"], color="#1E88E5")


def render_scorer_tab(model: Any, model_name: str) -> None:
    """Render interactive single-impression probability scorer tab."""
    st.subheader("⚡ Real-Time Single-Impression Probability Scorer")
    st.markdown(
        "Configure ad impression attributes and compute the calibrated click probability "
        "(**pCTR**) and real-time bidding auction metrics."
    )

    if model is None:
        st.error("Model artifact is not available.")
        return

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Impression & Context Features")
        cpo = st.slider(
            "Bid / Cost Per Order (CPO in $)",
            min_value=0.01,
            max_value=3.00,
            value=0.25,
            step=0.01,
            help="CPO proxy value associated with the advertiser's bid.",
        )

        recency_mode = st.radio(
            "User Click Recency",
            options=["First-Time Visitor (-1)", "Recent (< 1 hour)", "Active (1 to 24 hours)", "Dormant (> 24 hours)"],
            index=1,
            help="Time since the user previously clicked an ad in seconds.",
        )

        if recency_mode == "First-Time Visitor (-1)":
            time_since_last_click = -1
        elif recency_mode == "Recent (< 1 hour)":
            time_since_last_click = 900  # 15 minutes
        elif recency_mode == "Active (1 to 24 hours)":
            time_since_last_click = 14400  # 4 hours
        else:
            time_since_last_click = 172800  # 48 hours

        campaign_presets = {
            "Campaign 22589171 (High Volume Retail)": 22589171,
            "Campaign 884761 (Tech / SaaS)": 884761,
            "Campaign 18975823 (Automotive Brand)": 18975823,
            "Campaign 29427842 (Entertainment Media)": 29427842,
        }
        selected_campaign_name = st.selectbox("Campaign ID", list(campaign_presets.keys()))
        campaign_id = campaign_presets[selected_campaign_name]

    with col_right:
        st.markdown("#### Categorical Context Hashes")
        st.caption("Standard hashed contextual dimensions (cat1 - cat9)")
        cat_values = {
            "cat1": 5824233,
            "cat2": 9312274,
            "cat3": 3490278,
            "cat4": 29196072,
            "cat5": 11409686,
            "cat6": 1973606,
            "cat7": 25162884,
            "cat8": 29196072,
            "cat9": 29196072,
        }

        with st.expander("Customize Context Hashes", expanded=False):
            for col in APPROVED_CATEGORICAL_FEATURES:
                if col != "campaign":
                    cat_values[col] = st.number_input(f"{col}", value=cat_values[col], step=1)

    record = format_candidate_for_scoring(cpo, time_since_last_click, campaign_id, cat_values)
    df_single = pd.DataFrame([record])

    probs = model.predict_proba(df_single)[0]
    pctr = float(probs[1])
    ecpm = pctr * cpo * 1000.0

    st.markdown("---")
    st.markdown("### 📊 Prediction & Real-Time Auction Outcome")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Predicted pCTR", value=f"{pctr * 100:.2f}%")
    with m_col2:
        st.metric(label="Expected eCPM", value=f"${ecpm:.2f}")
    with m_col3:
        st.metric(label="Bid CPO", value=f"${cpo:.2f}")
    with m_col4:
        intent_label = "🔥 High Intent" if pctr > 0.45 else ("⚡ Medium Intent" if pctr > 0.30 else "❄️ Low Intent")
        st.metric(label="User Intent Tier", value=intent_label)

    # Explanation text
    if time_since_last_click == -1:
        st.info("ℹ️ User is a first-time visitor (`time_since_last_click = -1`). The model assigns lower baseline probability due to lack of immediate browsing momentum.")
    elif time_since_last_click < 3600:
        st.success("🔥 User engaged within the past hour (`time_since_last_click < 3600s`). Historical click rate for this cohort is ~66%, driving high predicted pCTR.")
    else:
        st.info("⏳ User has a recency gap exceeding 24 hours. The model dials back intent probability closer to the dormant baseline rate.")


def render_benchmark_tab() -> None:
    """Render model evaluation and benchmark comparison tab."""
    st.subheader("📈 Offline Validation Model Comparison")
    st.markdown(
        "Strict chronological validation evaluation on **15,000 holdout impressions** "
        "(empirical click rate = 34.29%). All models were evaluated under zero target leakage."
    )

    data = load_comparison_data()
    if not data or "models" not in data:
        st.error("Model comparison report not found at `reports/model_comparison.json`.")
        return

    models_dict = data["models"]
    rows = []
    for name, m in models_dict.items():
        rows.append({
            "Model Name": name,
            "Log Loss (↓)": round(m["log_loss"], 4),
            "ROC AUC (↑)": round(m["roc_auc"], 4),
            "PR AUC (↑)": round(m["pr_auc"], 4),
            "Brier Score (↓)": round(m["brier_score"], 4),
            "Train Latency (s)": round(m.get("train_time_seconds", 0.0), 3),
        })

    df_models = pd.DataFrame(rows).sort_values("Log Loss (↓)")
    st.dataframe(df_models, use_container_width=True, hide_index=True)

    st.markdown("### 🏆 Metric Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ROC AUC (Ranking Discrimination — Higher is Better)**")
        st.bar_chart(df_models.set_index("Model Name")["ROC AUC (↑)"], color="#2E7D32")

    with col2:
        st.markdown("**Log Loss (Cross-Entropy Calibration — Lower is Better)**")
        st.bar_chart(df_models.set_index("Model Name")["Log Loss (↓)"], color="#C62828")

    st.markdown("### 💡 Key Recommender Architecture Takeaways")
    st.markdown(
        """
        1. **LightGBM Champion**: LightGBM achieved the best overall performance (**ROC AUC: 0.6689, Log Loss: 0.6001**), beating the linear baseline by +0.0045 AUC points and the constant baseline by +0.1689 AUC points in just **1.08 seconds** of training.
        2. **User Cold Start (92.93%)**: Because user identity exhibits very low repeat overlap (only 7.07% user repeat across splits), memorizing user IDs is futile. Successful models must rely on campaign context and recency momentum.
        3. **PyTorch Embedding-MLP**: Achieved competitive ranking performance (**ROC AUC: 0.6573, Log Loss: 0.6070**) with dense entity embeddings, demonstrating how deep architectures handle high-cardinality categorical features.
        """
    )


def render_calibration_tab() -> None:
    """Render probability calibration and failure analysis tab."""
    st.subheader("🎯 Probability Calibration & Auction Error Analysis")
    st.markdown(
        "Why calibration matters: In Real-Time Bidding (RTB), expected payout is computed as "
        r"$$\text{eCPM} = \text{pCTR} \times \text{Bid} \times 1000$$. "
        "A model can rank ads well (high ROC AUC) but still bankrupt an advertiser if its probabilities are miscalibrated."
    )

    data = load_calibration_data()
    if not data:
        st.error("Calibration data not found at `reports/calibration_and_error_analysis.json`.")
        return

    c_summary = data.get("calibration_summary", {})
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Expected Calibration Error (ECE)", f"{c_summary.get('expected_calibration_error', 0.0):.4f}")
    with m2:
        st.metric("Brier Score", f"{c_summary.get('brier_score', 0.0):.4f}")
    with m3:
        st.metric("Validation Log Loss", f"{c_summary.get('log_loss', 0.0):.4f}")
    with m4:
        st.metric("Empirical CTR", f"{data.get('empirical_click_rate', 0.0) * 100:.2f}%")

    st.markdown("### 📊 Diagnostic Figures")
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        if os.path.exists(FIGURE_CALIBRATION_PATH):
            st.image(FIGURE_CALIBRATION_PATH, caption="Reliability Curve (Predicted vs Observed Click Rate)")
        else:
            st.info("Calibration curve figure not found.")
    with fig_col2:
        if os.path.exists(FIGURE_BUCKETS_PATH):
            st.image(FIGURE_BUCKETS_PATH, caption="Calibration Gap across Score Buckets")
        else:
            st.info("Score buckets figure not found.")

    if os.path.exists(FIGURE_DISTRIBUTION_PATH):
        st.image(FIGURE_DISTRIBUTION_PATH, caption="Predicted Click Probability Density (Non-Clicks vs Clicks)", use_container_width=True)

    st.markdown("### 🔬 Performance Slices: User Recency & Intent")
    slices = data.get("performance_slices", {}).get("user_recency_slices", [])
    if slices:
        df_slices = pd.DataFrame(slices)
        display_slices = df_slices[[
            "slice_name", "sample_count", "percentage_of_data",
            "actual_click_rate", "mean_predicted_prob", "roc_auc", "log_loss"
        ]].rename(columns={
            "slice_name": "Recency Cohort",
            "sample_count": "Samples",
            "percentage_of_data": "% of Data",
            "actual_click_rate": "Actual CTR",
            "mean_predicted_prob": "Mean pCTR",
            "roc_auc": "ROC AUC",
            "log_loss": "Log Loss",
        })
        st.dataframe(display_slices, use_container_width=True, hide_index=True)


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="AdLift — Ad Click Prediction & Ranking",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("⚡ AdLift: Ad Click Prediction & Candidate Ranking")
    st.caption("A portfolio-grade recommender systems showcase inspired by NVIDIA Merlin's click prediction and ranking architecture.")

    render_disclaimer_banner()

    # Sidebar model configuration
    st.sidebar.title("🛠️ Configuration")
    model_choice = st.sidebar.selectbox(
        "Active Prediction Model",
        options=["LightGBM", "PyTorch Embedding-MLP", "Logistic Regression", "Constant Baseline"],
        index=0,
        help="Select which trained model artifact will power the interactive demo.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ Guardrails Enforced")
    st.sidebar.markdown("- **Zero Target Leakage**: Post-impression conversions, attribution, and outcome counters strictly prohibited.")
    st.sidebar.markdown("- **Chronological Splits**: No future information leaked into training.")
    st.sidebar.markdown("- **Deterministic Ties**: Secondary sort by `candidate_id`.")
    st.sidebar.markdown("- **Synthetic Ad Demos**: Observational data disclaimed.")

    model = load_model(model_choice)

    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Candidate Ranking Demo",
        "⚡ Real-Time Scorer",
        "📈 Model Benchmarks",
        "🔬 Calibration & Error Analysis",
    ])

    with tab1:
        render_ranking_tab(model, model_choice)

    with tab2:
        render_scorer_tab(model, model_choice)

    with tab3:
        render_benchmark_tab()

    with tab4:
        render_calibration_tab()


if __name__ == "__main__":
    main()
