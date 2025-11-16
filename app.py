"""
Streamlit App for Sales Opportunity Prediction Explainability
Schneider Electric - GTM Machine Learning Project
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

# Page configuration
st.set_page_config(
    page_title="Sales Opportunity Explainability",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #3CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E7D32;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3CAF50;
    }
    .prompt-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #3CAF50;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #1a1a1a;
    }
    .feature-value {
        font-weight: bold;
        color: #1976D2;
    }
    .shap-positive {
        color: #2E7D32;
        font-weight: bold;
    }
    .shap-negative {
        color: #C62828;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title
st.markdown("## 🎯 Sales Opportunity Prediction Explainability")
st.markdown("**Schneider Electric - Go to Market Analytics**")
st.markdown("---")


# Load data
@st.cache_data
def load_data():
    """Load predictions data and prompt template"""
    try:
        # Load CSV with predictions
        predictions_df = pd.read_csv("data/predictions_with_shap.csv")

        # Load detailed JSON
        with open("predictions_detailed.json", "r") as f:
            predictions_detailed = json.load(f)

        # Load prompt template
        with open("templates/prompt_template.txt", "r") as f:
            prompt_template = f.read()

        # Load feature descriptions from PROJECT_INFO.md
        feature_descriptions = {
            "product_A_sold_in_the_past": "Historical sales of Product A with this customer",
            "product_B_sold_in_the_past": "Historical sales of Product B with this customer",
            "product_A_recommended": "Whether Product A was recommended in the past to this customer",
            "product_A": "Amount of Product A in this opportunity",
            "product_C": "Amount of Product C in this opportunity",
            "product_D": "Amount of Product D in this opportunity",
            "cust_hitrate": "Customer success rate in previous interactions",
            "cust_interactions": "Number of interactions with the customer",
            "cust_contracts": "Number of contracts signed with the customer",
            "opp_month": "Month when the opportunity was created",
            "opp_old": "Whether the opportunity has been open for a long time",
            "competitor_Z": "Presence of competitor Z in the opportunity",
            "competitor_X": "Presence of competitor X in the opportunity",
            "competitor_Y": "Presence of competitor Y in the opportunity",
            "cust_in_iberia": "Whether the customer is located in Iberia",
        }

        return (
            predictions_df,
            predictions_detailed,
            prompt_template,
            feature_descriptions,
        )
    except FileNotFoundError as e:
        st.error(
            f"⚠️ Required files not found. Please run temp.py first to generate predictions."
        )
        st.error(f"Error: {e}")
        return None, None, None, None


predictions_df, predictions_detailed, prompt_template, feature_descriptions = (
    load_data()
)

if predictions_df is None:
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filter Options")

# Prediction filter
prediction_filter = st.sidebar.multiselect(
    "Prediction", options=["WON", "LOST"], default=["WON", "LOST"]
)

# Correctness filter
correctness_filter = st.sidebar.multiselect(
    "Prediction Correctness",
    options=["Correct", "Incorrect"],
    default=["Correct", "Incorrect"],
)

# Confidence range
confidence_range = st.sidebar.slider(
    "Prediction Confidence (%)", min_value=0, max_value=100, value=(0, 100), step=5
)

# Apply filters
filtered_df = predictions_df.copy()

# Filter by prediction
pred_map = {"WON": 1, "LOST": 0}
if prediction_filter:
    filtered_df = filtered_df[
        filtered_df["prediction"].isin([pred_map[p] for p in prediction_filter])
    ]

# Filter by correctness
if "Correct" in correctness_filter and "Incorrect" not in correctness_filter:
    filtered_df = filtered_df[filtered_df["correct_prediction"] == 1]
elif "Incorrect" in correctness_filter and "Correct" not in correctness_filter:
    filtered_df = filtered_df[filtered_df["correct_prediction"] == 0]

# Filter by confidence
filtered_df = filtered_df[
    (filtered_df["prediction_probability"] * 100 >= confidence_range[0])
    & (filtered_df["prediction_probability"] * 100 <= confidence_range[1])
]

# Main content
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Dataset Overview",
        "🔎 Explore Predictions",
        "💬 Generate LLM Prompt",
        "📈 Model Performance",
    ]
)

# Tab 1: Dataset Overview
with tab1:
    st.markdown(
        '<div class="sub-header">Dataset Statistics</div>', unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Instances", len(predictions_df))

    with col2:
        won_count = (predictions_df["prediction"] == 1).sum()
        st.metric(
            "Predicted WON", f"{won_count} ({won_count/len(predictions_df)*100:.1f}%)"
        )

    with col3:
        accuracy = (
            (predictions_df["correct_prediction"] == 1).sum()
            / len(predictions_df)
            * 100
        )
        st.metric("Accuracy", f"{accuracy:.1f}%")

    with col4:
        avg_confidence = predictions_df["prediction_probability"].mean() * 100
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

    st.markdown("---")

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        # Prediction distribution
        pred_dist = predictions_df["prediction"].value_counts()
        fig = px.pie(
            values=pred_dist.values,
            names=["LOST", "WON"],
            title="Prediction Distribution",
            color_discrete_sequence=["#C62828", "#2E7D32"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Confidence distribution
        fig = px.histogram(
            predictions_df,
            x="prediction_probability",
            nbins=30,
            title="Prediction Confidence Distribution",
            labels={"prediction_probability": "Confidence"},
            color_discrete_sequence=["#1976D2"],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    st.markdown(
        '<div class="sub-header">Confusion Matrix</div>', unsafe_allow_html=True
    )

    confusion_data = pd.crosstab(
        predictions_df["actual_outcome"].map({0: "LOST", 1: "WON"}),
        predictions_df["prediction"].map({0: "LOST", 1: "WON"}),
        rownames=["Actual"],
        colnames=["Predicted"],
    )

    fig = px.imshow(
        confusion_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Greens",
        labels=dict(x="Predicted", y="Actual"),
        title="Model Performance",
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Explore Predictions
with tab2:
    st.markdown(
        '<div class="sub-header">Filtered Predictions</div>', unsafe_allow_html=True
    )
    st.write(f"Showing {len(filtered_df)} of {len(predictions_df)} instances")

    # Display filtered data
    display_cols = [
        "test_index",
        "prediction",
        "prediction_probability",
        "actual_outcome",
        "correct_prediction",
    ]

    # Add feature columns
    feature_cols = [
        col
        for col in predictions_df.columns
        if col not in display_cols and not col.startswith("shap_") and col != "shap_sum"
    ]

    display_data = filtered_df[display_cols + feature_cols].copy()
    display_data["prediction"] = display_data["prediction"].map({0: "LOST", 1: "WON"})
    display_data["actual_outcome"] = display_data["actual_outcome"].map(
        {0: "LOST", 1: "WON"}
    )
    display_data["correct_prediction"] = display_data["correct_prediction"].map(
        {0: "❌", 1: "✅"}
    )
    display_data["prediction_probability"] = (
        display_data["prediction_probability"] * 100
    ).round(1)

    st.dataframe(display_data, use_container_width=True, height=400)

    # Top features by SHAP importance
    st.markdown(
        '<div class="sub-header">Global Feature Importance (Mean |SHAP|)</div>',
        unsafe_allow_html=True,
    )

    shap_cols = [
        col
        for col in predictions_df.columns
        if col.startswith("shap_") and col != "shap_base_value" and col != "shap_sum"
    ]

    mean_shap = {}
    for col in shap_cols:
        feature_name = col.replace("shap_", "")
        mean_shap[feature_name] = predictions_df[col].abs().mean()

    importance_df = pd.DataFrame(
        list(mean_shap.items()), columns=["Feature", "Mean |SHAP|"]
    )
    importance_df = importance_df.sort_values("Mean |SHAP|", ascending=False).head(10)

    fig = px.bar(
        importance_df,
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        title="Top 10 Most Important Features",
        color="Mean |SHAP|",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    # Generate Global Importance Prompt
    st.markdown("---")
    st.markdown(
        '<div class="sub-header">📊 Global Feature Importance Explanation</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Generate an LLM prompt to explain the global feature importance across all predictions."
    )

    if st.button("🤖 Generate Global Importance Prompt"):
        # Load global importance prompt template
        try:
            with open("templates/global_importance_prompt.txt", "r") as f:
                global_prompt_template = f.read()

            # Create full importance table (not just top 10)
            full_importance_df = pd.DataFrame(
                list(mean_shap.items()), columns=["Feature", "Mean |SHAP|"]
            )
            full_importance_df = full_importance_df.sort_values(
                "Mean |SHAP|", ascending=False
            )

            # Format as a readable table with feature descriptions
            importance_table = "| Rank | Feature | Importance Score | Description |\n"
            importance_table += "|------|---------|------------------|-------------|\n"

            for idx, row in full_importance_df.iterrows():
                rank = full_importance_df.index.tolist().index(idx) + 1
                feature = row["Feature"]
                score = row["Mean |SHAP|"]
                description = feature_descriptions.get(feature, feature)
                importance_table += (
                    f"| {rank} | {feature} | {score:.4f} | {description} |\n"
                )

            # Fill in the template
            filled_global_prompt = global_prompt_template.format(
                feature_importance_table=importance_table
            )

            # Display the prompt
            st.markdown(
                '<div class="prompt-box">'
                + filled_global_prompt.replace("\n", "<br>")
                + "</div>",
                unsafe_allow_html=True,
            )

            # Copy to clipboard functionality
            import streamlit.components.v1 as components

            escaped_global_prompt = (
                filled_global_prompt.replace("\\", "\\\\")
                .replace("`", "\\`")
                .replace("$", "\\$")
            )

            copy_button_html_global = f"""
            <button onclick="copyGlobalToClipboard()" style="
                background-color: #3CAF50;
                color: white;
                padding: 0.5rem 1rem;
                border: none;
                border-radius: 0.25rem;
                cursor: pointer;
                font-size: 1rem;
                margin-top: 10px;
            ">
                📋 Copy Global Importance Prompt to Clipboard
            </button>
            <script>
            function copyGlobalToClipboard() {{
                const text = `{escaped_global_prompt}`;
                navigator.clipboard.writeText(text).then(function() {{
                    alert('✅ Global importance prompt copied to clipboard!');
                }}, function(err) {{
                    alert('❌ Failed to copy.');
                }});
            }}
            </script>
            """
            components.html(copy_button_html_global, height=60)

            st.info(
                "💡 **Tip**: Copy this prompt and paste it into your preferred LLM to get strategic insights about which features drive sales success globally."
            )

        except FileNotFoundError:
            st.error(
                "⚠️ Global importance prompt template not found. Please ensure templates/global_importance_prompt.txt exists."
            )

# Tab 3: Generate LLM Prompt
with tab3:
    st.markdown(
        '<div class="sub-header">Select Instance for Explanation</div>',
        unsafe_allow_html=True,
    )

    # Instance selector
    col1, col2 = st.columns([3, 1])

    with col1:
        # Initialize session state for random selection
        if "random_index" not in st.session_state:
            st.session_state.random_index = None

        # Determine the index for the selectbox
        if (
            st.session_state.random_index is not None
            and st.session_state.random_index in filtered_df["test_index"].tolist()
        ):
            default_idx = (
                filtered_df["test_index"].tolist().index(st.session_state.random_index)
            )
        else:
            default_idx = 0

        selected_index = st.selectbox(
            "Choose an instance to explain:",
            options=filtered_df["test_index"].tolist(),
            index=default_idx,
            format_func=lambda x: f"Instance #{x} - Predicted: {predictions_detailed[x]['prediction_label']}, Actual: {predictions_detailed[x]['actual_label']}, Confidence: {predictions_detailed[x]['prediction_probability']*100:.1f}%",
        )

    with col2:
        # Quick select buttons
        st.write("Quick Select:")
        if st.button("Random Instance"):
            st.session_state.random_index = np.random.choice(
                filtered_df["test_index"].tolist()
            )
            st.rerun()

    if selected_index is not None:
        instance = predictions_detailed[selected_index]

        # Display instance details
        st.markdown("---")
        st.markdown(
            '<div class="sub-header">Instance Details</div>', unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pred_color = "🟢" if instance["prediction"] == 1 else "🔴"
            st.metric("Prediction", f"{pred_color} {instance['prediction_label']}")

        with col2:
            actual_color = "🟢" if instance["actual_outcome"] == 1 else "🔴"
            st.metric("Actual", f"{actual_color} {instance['actual_label']}")

        with col3:
            st.metric("Confidence", f"{instance['prediction_probability']*100:.1f}%")

        with col4:
            correct = "✅" if instance["correct_prediction"] else "❌"
            st.metric("Correct", correct)

        # Feature values
        st.markdown(
            '<div class="sub-header">Feature Values</div>', unsafe_allow_html=True
        )

        feature_df = pd.DataFrame(
            [
                {
                    "Feature": feature_descriptions.get(k, k),
                    "Value": f"{v:.4f}",
                    "SHAP Impact": f"{instance['shap_values'][k]:+.4f}",
                }
                for k, v in instance["feature_values"].items()
            ]
        )

        st.dataframe(feature_df, use_container_width=True, height=300)

        # SHAP visualization
        st.markdown(
            '<div class="sub-header">SHAP Value Analysis</div>', unsafe_allow_html=True
        )

        shap_data = pd.DataFrame(
            [
                {"Feature": k, "SHAP Value": v}
                for k, v in instance["shap_values"].items()
            ]
        ).sort_values("SHAP Value", ascending=True)

        fig = go.Figure()

        colors = ["#C62828" if x < 0 else "#2E7D32" for x in shap_data["SHAP Value"]]

        fig.add_trace(
            go.Bar(
                y=shap_data["Feature"],
                x=shap_data["SHAP Value"],
                orientation="h",
                marker=dict(color=colors),
                text=[f"{x:+.4f}" for x in shap_data["SHAP Value"]],
                textposition="outside",
            )
        )

        fig.update_layout(
            title="SHAP Values: How Each Feature Influenced the Prediction",
            xaxis_title="SHAP Value (← LOST | WON →)",
            yaxis_title="Feature",
            height=500,
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Generate prompt
        st.markdown("---")
        st.markdown(
            '<div class="sub-header">📝 Generated LLM Prompt</div>',
            unsafe_allow_html=True,
        )

        # Format feature values
        feature_values_text = "\n".join(
            [
                f"- **{feature_descriptions.get(k, k)}**: {v:.4f}"
                for k, v in instance["feature_values"].items()
            ]
        )

        # Format SHAP explanation
        shap_explanation_text = "\n".join(
            [
                f"- **{feature_descriptions.get(feat, feat)}**: {val:+.4f} {'(pushes toward WON)' if val > 0 else '(pushes toward LOST)'}"
                for feat, val in sorted(
                    instance["shap_values"].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
            ]
        )

        # Format top factors
        top_factors_text = ""
        for i, (feat, val) in enumerate(instance["top_absolute_features"][:5], 1):
            direction = "WON" if val > 0 else "LOST"
            top_factors_text += f"{i}. **{feature_descriptions.get(feat, feat)}** (SHAP: {val:+.4f})\n   - This feature strongly pushes the prediction toward {direction}\n\n"

        # Fill in the template
        filled_prompt = prompt_template.format(
            opportunity_id=f"Test Instance #{selected_index}",
            prediction=instance["prediction"],
            prediction_label=instance["prediction_label"],
            prediction_probability=instance["prediction_probability"],
            actual_outcome=instance["actual_label"],
            feature_values=feature_values_text,
            shap_explanation=shap_explanation_text,
            base_value=instance["shap_base_value"],
            top_factors=top_factors_text,
        )

        # Display prompt
        st.markdown(
            '<div class="prompt-box">' + filled_prompt.replace("\n", "<br>") + "</div>",
            unsafe_allow_html=True,
        )

        # Copy to clipboard functionality
        import streamlit.components.v1 as components

        # Escape the prompt for JavaScript
        escaped_prompt = (
            filled_prompt.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        )

        copy_button_html = f"""
        <button onclick="copyToClipboard()" style="
            background-color: #3CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 0.25rem;
            cursor: pointer;
            font-size: 1rem;
        ">
            📋 Copy to Clipboard
        </button>
        <script>
        function copyToClipboard() {{
            const text = `{escaped_prompt}`;
            navigator.clipboard.writeText(text).then(function() {{
                alert('✅ Prompt copied to clipboard!');
            }}, function(err) {{
                alert('❌ Failed to copy.');
            }});
        }}
        </script>
        """
        components.html(copy_button_html, height=50)

        # Additional info
        st.info(
            "💡 **Tip**: Copy this prompt and paste it into your preferred LLM (ChatGPT, Claude, etc.) to get a human-readable explanation of this prediction."
        )

# Tab 4: Model Performance Deep Dive
with tab4:
    st.markdown(
        '<div class="sub-header">📈 Model Performance Deep Dive</div>',
        unsafe_allow_html=True,
    )

    # Calculate comprehensive metrics
    y_true = predictions_df["actual_outcome"].values
    y_pred = predictions_df["prediction"].values
    y_proba = predictions_df["prediction_probability"].values

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    accuracy = (y_true == y_pred).mean()

    # Display key metrics
    st.markdown("### 🎯 Classification Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        metric_color = "🟢" if f1 >= 0.7 else "🟡" if f1 >= 0.6 else "🔴"
        st.metric("F1 Score", f"{f1:.3f} {metric_color}")
    with col3:
        st.metric("Precision", f"{precision:.3f}")
    with col4:
        st.metric("Recall", f"{recall:.3f}")
    with col5:
        st.metric("ROC-AUC", f"{roc_auc:.3f}")
    with col6:
        st.metric("Avg Precision", f"{avg_precision:.3f}")

    # Performance benchmark info
    if f1 >= 0.7:
        st.success(
            f"✅ **Model exceeds minimum requirement**: F1 Score {f1:.3f} ≥ 0.70 (project requirement)"
        )
    else:
        st.warning(
            f"⚠️ **Performance below target**: F1 Score {f1:.3f} < 0.70 (project requirement)"
        )

    st.markdown("---")

    # Detailed confusion matrix with metrics
    st.markdown("### 🔢 Confusion Matrix Analysis")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Enhanced confusion matrix
        confusion_data = pd.crosstab(
            predictions_df["actual_outcome"].map({0: "LOST", 1: "WON"}),
            predictions_df["prediction"].map({0: "LOST", 1: "WON"}),
            rownames=["Actual"],
            colnames=["Predicted"],
        )

        # Calculate percentages
        confusion_pct = confusion_data.div(confusion_data.sum(axis=1), axis=0) * 100

        # Create annotated confusion matrix
        annotations = []
        for i, row in enumerate(confusion_data.values):
            for j, val in enumerate(row):
                pct = confusion_pct.values[i, j]
                annotations.append(f"{val}<br>({pct:.1f}%)")

        fig = go.Figure(
            data=go.Heatmap(
                z=confusion_data.values,
                x=confusion_data.columns,
                y=confusion_data.index,
                text=np.array(annotations).reshape(confusion_data.shape),
                texttemplate="%{text}",
                textfont={"size": 14},
                colorscale="Greens",
                showscale=True,
            )
        )

        fig.update_layout(
            title="Confusion Matrix (Count & %)",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Performance metrics breakdown
        tn = confusion_data.loc["LOST", "LOST"]
        fp = confusion_data.loc["LOST", "WON"]
        fn = confusion_data.loc["WON", "LOST"]
        tp = confusion_data.loc["WON", "WON"]

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        metrics_data = {
            "Metric": [
                "True Positives (TP)",
                "True Negatives (TN)",
                "False Positives (FP)",
                "False Negatives (FN)",
                "Sensitivity (Recall)",
                "Specificity",
                "Precision (PPV)",
                "Negative Pred. Value",
            ],
            "Value": [
                f"{tp}",
                f"{tn}",
                f"{fp}",
                f"{fn}",
                f"{recall:.3f}",
                f"{specificity:.3f}",
                f"{precision:.3f}",
                f"{npv:.3f}",
            ],
            "Description": [
                "Correctly predicted WON",
                "Correctly predicted LOST",
                "Predicted WON but was LOST",
                "Predicted LOST but was WON",
                "% of actual WON identified",
                "% of actual LOST identified",
                "% of predicted WON correct",
                "% of predicted LOST correct",
            ],
        }

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, height=350)

    st.markdown("---")

    # ROC and Precision-Recall Curves
    st.markdown("### 📉 Model Discrimination Curves")

    col1, col2 = st.columns(2)

    with col1:
        # ROC Curve
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_proba)

        fig = go.Figure()

        # ROC curve
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC (AUC = {roc_auc:.3f})",
                line=dict(color="#2E7D32", width=3),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            )
        )

        # Diagonal reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="gray", width=2, dash="dash"),
                showlegend=True,
            )
        )

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            height=400,
            hovermode="closest",
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Precision-Recall Curve
        prec, rec, thresholds_pr = precision_recall_curve(y_true, y_proba)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=rec,
                y=prec,
                mode="lines",
                name=f"PR Curve (AP = {avg_precision:.3f})",
                line=dict(color="#1976D2", width=3),
                hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
            )
        )

        # Baseline
        baseline = y_true.sum() / len(y_true)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[baseline, baseline],
                mode="lines",
                name=f"Baseline ({baseline:.3f})",
                line=dict(color="gray", width=2, dash="dash"),
            )
        )

        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400,
            hovermode="closest",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Performance by Confidence Threshold
    st.markdown("### 🎚️ Performance by Prediction Threshold")

    thresholds = np.arange(0.1, 1.0, 0.05)
    threshold_metrics = []

    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        if len(np.unique(y_pred_thresh)) > 1:  # Avoid division by zero
            f1_thresh = f1_score(y_true, y_pred_thresh)
            prec_thresh = precision_score(y_true, y_pred_thresh, zero_division=0)
            rec_thresh = recall_score(y_true, y_pred_thresh, zero_division=0)
        else:
            f1_thresh = prec_thresh = rec_thresh = 0

        threshold_metrics.append(
            {
                "Threshold": thresh,
                "F1": f1_thresh,
                "Precision": prec_thresh,
                "Recall": rec_thresh,
            }
        )

    threshold_df = pd.DataFrame(threshold_metrics)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=threshold_df["Threshold"],
            y=threshold_df["F1"],
            mode="lines+markers",
            name="F1 Score",
            line=dict(color="#2E7D32", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=threshold_df["Threshold"],
            y=threshold_df["Precision"],
            mode="lines+markers",
            name="Precision",
            line=dict(color="#1976D2", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=threshold_df["Threshold"],
            y=threshold_df["Recall"],
            mode="lines+markers",
            name="Recall",
            line=dict(color="#F57C00", width=2),
        )
    )

    # Mark current threshold (0.5)
    current_thresh_idx = np.argmin(np.abs(threshold_df["Threshold"] - 0.5))
    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Current (0.5)",
        annotation_position="top",
    )

    fig.update_layout(
        title="Metrics vs. Classification Threshold",
        xaxis_title="Threshold",
        yaxis_title="Score",
        height=400,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **Insight**: Adjusting the classification threshold can optimize for precision (fewer false positives) or recall (fewer false negatives) based on business needs."
    )

    st.markdown("---")

    # Performance Segmentation Analysis
    st.markdown("### 🔍 Performance by Segments")

    segment_option = st.selectbox(
        "Analyze performance by:",
        [
            "Competitor Presence",
            "Customer History",
            "Opportunity Age",
            "Product Mix",
            "Confidence Level",
        ],
    )

    if segment_option == "Competitor Presence":
        # Analyze by competitor presence
        predictions_df["has_competitor"] = (
            (predictions_df["competitor_X"] > 0)
            | (predictions_df["competitor_Y"] > 0)
            | (predictions_df["competitor_Z"] > 0)
        ).astype(int)

        segments = predictions_df.groupby("has_competitor").apply(
            lambda x: pd.Series(
                {
                    "Count": len(x),
                    "Accuracy": (x["prediction"] == x["actual_outcome"]).mean(),
                    "F1": f1_score(x["actual_outcome"], x["prediction"]),
                    "Precision": precision_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Recall": recall_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Win_Rate": x["actual_outcome"].mean(),
                }
            )
        )
        segments.index = ["No Competitor", "Has Competitor"]

    elif segment_option == "Customer History":
        # Analyze by customer contracts
        predictions_df["customer_tier"] = pd.cut(
            predictions_df["cust_contracts"],
            bins=[-np.inf, 0, 2, np.inf],
            labels=["New", "Growing", "Established"],
        )

        segments = predictions_df.groupby("customer_tier").apply(
            lambda x: pd.Series(
                {
                    "Count": len(x),
                    "Accuracy": (x["prediction"] == x["actual_outcome"]).mean(),
                    "F1": f1_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Precision": precision_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Recall": recall_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Win_Rate": x["actual_outcome"].mean(),
                }
            )
        )

    elif segment_option == "Opportunity Age":
        # Analyze by opportunity age
        segments = predictions_df.groupby("opp_old").apply(
            lambda x: pd.Series(
                {
                    "Count": len(x),
                    "Accuracy": (x["prediction"] == x["actual_outcome"]).mean(),
                    "F1": f1_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Precision": precision_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Recall": recall_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Win_Rate": x["actual_outcome"].mean(),
                }
            )
        )
        segments.index = ["Recent", "Old"]

    elif segment_option == "Product Mix":
        # Analyze by product diversity
        predictions_df["product_diversity"] = (
            (predictions_df["product_A"] > 0).astype(int)
            + (predictions_df["product_C"] > 0).astype(int)
            + (predictions_df["product_D"] > 0).astype(int)
        )

        segments = predictions_df.groupby("product_diversity").apply(
            lambda x: pd.Series(
                {
                    "Count": len(x),
                    "Accuracy": (x["prediction"] == x["actual_outcome"]).mean(),
                    "F1": f1_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Precision": precision_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Recall": recall_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Win_Rate": x["actual_outcome"].mean(),
                }
            )
        )
        segments.index = [f"{int(i)} Products" for i in segments.index]

    else:  # Confidence Level
        predictions_df["confidence_tier"] = pd.cut(
            predictions_df["prediction_probability"],
            bins=[0, 0.6, 0.8, 1.0],
            labels=["Low (0-60%)", "Medium (60-80%)", "High (80-100%)"],
        )

        segments = predictions_df.groupby("confidence_tier").apply(
            lambda x: pd.Series(
                {
                    "Count": len(x),
                    "Accuracy": (x["prediction"] == x["actual_outcome"]).mean(),
                    "F1": f1_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Precision": precision_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Recall": recall_score(
                        x["actual_outcome"], x["prediction"], zero_division=0
                    ),
                    "Win_Rate": x["actual_outcome"].mean(),
                }
            )
        )

    # Display segment analysis
    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(
            segments.style.format(
                {
                    "Count": "{:.0f}",
                    "Accuracy": "{:.3f}",
                    "F1": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "Win_Rate": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

    with col2:
        # Visualize segment performance
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=segments.index,
                y=segments["F1"],
                name="F1 Score",
                marker_color="#2E7D32",
            )
        )

        fig.add_trace(
            go.Bar(
                x=segments.index,
                y=segments["Accuracy"],
                name="Accuracy",
                marker_color="#1976D2",
            )
        )

        fig.update_layout(
            title=f"Performance by {segment_option}",
            xaxis_title="Segment",
            yaxis_title="Score",
            barmode="group",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    # Key insights
    best_segment = segments["F1"].idxmax()
    worst_segment = segments["F1"].idxmin()

    st.success(
        f"✅ **Best Performance**: {best_segment} segment (F1 = {segments.loc[best_segment, 'F1']:.3f})"
    )
    st.warning(
        f"⚠️ **Needs Improvement**: {worst_segment} segment (F1 = {segments.loc[worst_segment, 'F1']:.3f})"
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Schneider Electric - GTM Machine Learning Explainability | "
    "Built with Streamlit 🚀"
    "</div>",
    unsafe_allow_html=True,
)
