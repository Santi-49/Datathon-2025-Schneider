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
import streamlit.components.v1 as components
import dotenv
import os
import base64
import llm

# Load environment variables
dotenv.load_dotenv()

env_api_key = os.getenv("OPEN_AI_API_KEY", "")
if env_api_key:
    st.session_state["openai_api_key"] = env_api_key

# Page configuration
st.set_page_config(
    page_title="Schneider Electric | Sales Opportunity Explainability",
    page_icon="media/Schneider-Electric-logo-jpg-.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Helper function to load and encode image
def get_base64_image(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None


# Load logo
logo_base64 = get_base64_image("media/Schneider-Electric-logo-jpg-.png")

# Custom CSS - Schneider Electric branded
st.markdown(
    f"""
<style>
    /* Schneider Electric Brand Colors */
    :root {{
        --se-green: #3DCD58;
        --se-dark-green: #009530;
        --se-light-green: #7FE89A;
        --se-gray: #4A4A4A;
        --se-light-gray: #F0F2F6;
    }}
    
    /* Main headers - White for dark mode */
    .main-header {{
        font-size: 2.8rem;
        font-weight: 700;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .sub-title {{
        font-size: 1.2rem;
        font-weight: 400;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 2rem;
        opacity: 0.9;
    }}
    
    /* Sub-headers - White for dark mode */
    .sub-header {{
        font-size: 1.8rem;
        font-weight: 600;
        color: #FFFFFF !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #3DCD58;
        padding-left: 15px;
    }}
    
    .section-header {{
        font-size: 1.4rem;
        font-weight: 600;
        color: #FFFFFF !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }}
    
    /* Metric cards with Schneider branding */
    .metric-card {{
        background: linear-gradient(135deg, rgba(61, 205, 88, 0.1) 0%, rgba(0, 149, 48, 0.1) 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #3DCD58;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    /* Prompt box styling */
    .prompt-box {{
        background-color: rgba(42, 42, 42, 0.8);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #3DCD58;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(61, 205, 88, 0.2);
    }}
    
    /* Feature values */
    .feature-value {{
        font-weight: 600;
        color: #3DCD58;
    }}
    
    /* SHAP value colors */
    .shap-positive {{
        color: #3DCD58;
        font-weight: 700;
    }}
    
    .shap-negative {{
        color: #FF5252;
        font-weight: 700;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(61, 205, 88, 0.05) 0%, rgba(0, 149, 48, 0.05) 100%);
    }}
    
    /* Button styling */
    .stButton>button {{
        background-color: #3DCD58;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        background-color: #009530;
        box-shadow: 0 4px 12px rgba(61, 205, 88, 0.3);
        transform: translateY(-2px);
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        color: #FFFFFF;
        font-weight: 600;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: rgba(61, 205, 88, 0.2);
        border-bottom: 3px solid #3DCD58;
    }}
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {{
        font-size: 1.8rem;
        font-weight: 700;
        color: #3DCD58;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: #FFFFFF !important;
        font-weight: 600;
    }}
    
    /* Dataframe styling */
    .dataframe {{
        border-radius: 10px;
        overflow: hidden;
    }}
    
    /* Header container */
    .header-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem 0 1.5rem 0;
        background: linear-gradient(135deg, rgba(61, 205, 88, 0.1) 0%, rgba(0, 149, 48, 0.1) 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    
    .logo-container {{
        text-align: center;
        margin-bottom: 1rem;
    }}
    
    /* Info boxes */
    .stAlert {{
        border-radius: 10px;
        border-left: 5px solid #3DCD58;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# Header with logo
st.markdown('<div class="header-container">', unsafe_allow_html=True)
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if logo_base64:
        st.markdown(
            f'<div class="logo-container"><img src="data:image/png;base64,{logo_base64}" width="120"></div>',
            unsafe_allow_html=True,
        )
with col_title:
    st.markdown(
        '<h1 class="main-header">Sales Opportunity Prediction</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-title">AI-Powered Explainability Dashboard | Go-to-Market Analytics</p>',
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


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
            f" Required files not found. Please run temp.py first to generate predictions."
        )
        st.error(f"Error: {e}")
        return None, None, None, None


predictions_df, predictions_detailed, prompt_template, feature_descriptions = (
    load_data()
)

if predictions_df is None:
    st.stop()

# Sidebar filters with enhanced branding
st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 1rem 0; margin-bottom: 1rem;'>
        <h1 style='color: #3DCD58; font-size: 2.5rem; margin: 0;'>⚡</h1>
        <h3 style='color: #3DCD58; margin: 0.5rem 0;'>AI Control Panel</h3>
        <p style='color: #AAAAAA; font-size: 0.9rem; margin: 0;'>Schneider Electric</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Filter Options")

# Prediction filter
prediction_filter = st.sidebar.multiselect(
    " Prediction Outcome",
    options=["WON", "LOST"],
    default=["WON", "LOST"],
    help="Filter by predicted opportunity outcome",
)

# Correctness filter
correctness_filter = st.sidebar.multiselect(
    "✓ Prediction Accuracy",
    options=["Correct", "Incorrect"],
    default=["Correct", "Incorrect"],
    help="Show only correct or incorrect predictions",
)

# Confidence range
confidence_range = st.sidebar.slider(
    " Confidence Level (%)",
    min_value=0,
    max_value=100,
    value=(0, 100),
    step=5,
    help="Filter by model prediction confidence",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center; color: #AAAAAA; font-size: 0.85rem; padding: 1rem;'>
        <p><strong style='color: #3DCD58;'>Life Is On</strong></p>
        <p>🌍 Sustainability in Action</p>
        <p style='font-size: 0.75rem; margin-top: 0.5rem;'>
            Driving efficiency through<br>intelligent analytics
        </p>
    </div>
    """,
    unsafe_allow_html=True,
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

# Main content with enhanced tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Dataset Overview",
        "Explore Predictions",
        "Generate LLM Prompt",
        "Model Performance",
        "Settings",
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
            color_discrete_sequence=["#FF5252", "#3DCD58"],
        )
        fig.update_layout(
            title_font_color="#FFFFFF", title_font_size=18, legend_font_color="#FFFFFF"
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
            color_discrete_sequence=["#3DCD58"],
        )
        fig.update_layout(
            title_font_color="#FFFFFF",
            title_font_size=18,
            xaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
            yaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
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
        color_continuous_scale=[[0, "#1a472a"], [0.5, "#3DCD58"], [1, "#7FE89A"]],
        labels=dict(x="Predicted", y="Actual"),
        title="Model Performance",
    )
    fig.update_layout(
        title_font_color="#FFFFFF",
        title_font_size=18,
        xaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
        yaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
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
        color_continuous_scale=[[0, "#009530"], [0.5, "#3DCD58"], [1, "#7FE89A"]],
    )
    fig.update_layout(
        title_font_color="#FFFFFF",
        title_font_size=18,
        xaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
        yaxis=dict(
            categoryorder="total ascending",
            title_font_color="#FFFFFF",
            tickfont_color="#FFFFFF",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Global Feature Importance Explanation
    st.markdown("---")
    st.markdown(
        '<div class="sub-header"> Global Feature Importance Explanation</div>',
        unsafe_allow_html=True,
    )

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
        ">
             Copy to Clipboard
        </button>
        <script>
        function copyGlobalToClipboard() {{
            const text = `{escaped_global_prompt}`;
            navigator.clipboard.writeText(text).then(function() {{
                alert(' Prompt copied to clipboard!');
            }}, function(err) {{
                alert(' Failed to copy.');
            }});
        }}
        </script>
        """
        components.html(copy_button_html_global, height=50)

        st.info(
            " **Tip**: Copy this prompt and paste it into your preferred LLM (ChatGPT, Claude, etc.) to get a human-readable explanation of the global feature importance."
        )

        # Generate Explanation with OpenAI for Global Importance
        st.markdown("---")
        st.markdown(
            '<div class="sub-header">🤖 AI-Generated Explanation</div>',
            unsafe_allow_html=True,
        )

        if "openai_api_key" in st.session_state and st.session_state["openai_api_key"]:
            if st.button(
                "🚀 Generate Explanation with AI", type="primary", key="global_ai_btn"
            ):
                with st.spinner("Generating explanation..."):
                    try:
                        explanation = llm.generate_explanation(
                            prompt=filled_global_prompt,
                            api_key=st.session_state["openai_api_key"],
                        )
                        st.markdown("### Generated Explanation")
                        st.markdown(explanation)
                        st.success("✅ Explanation generated successfully!")
                    except ValueError as e:
                        st.error(f"❌ API Key Error: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Error generating explanation: {str(e)}")
        else:
            st.warning(
                "⚠️ OpenAI API key required. Please configure your API key in the Settings tab to use this feature."
            )

    except FileNotFoundError:
        st.error(
            " Global importance prompt template not found. Please ensure global_importance_prompt.txt exists."
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

        colors = ["#FF5252" if x < 0 else "#3DCD58" for x in shap_data["SHAP Value"]]

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
            title_font_color="#FFFFFF",
            title_font_size=18,
            xaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
            yaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Generate prompt
        st.markdown("---")
        st.markdown(
            '<div class="sub-header"> Generated LLM Prompt</div>',
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
            background: linear-gradient(135deg, #3DCD58 0%, #009530 100%);
            color: white;
            padding: 0.75rem 2rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.1rem;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(61, 205, 88, 0.3);
            transition: all 0.3s ease;
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(61, 205, 88, 0.4)'"
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(61, 205, 88, 0.3)'">
             Copy Prompt to Clipboard
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
        components.html(copy_button_html, height=60)

        # Generate Explanation with OpenAI
        st.markdown("---")
        st.markdown(
            '<div class="sub-header">🤖 AI-Generated Explanation</div>',
            unsafe_allow_html=True,
        )

        if "openai_api_key" in st.session_state and st.session_state["openai_api_key"]:
            if st.button(
                "🚀 Generate Explanation with AI", type="primary", key="instance_ai_btn"
            ):
                with st.spinner("Generating explanation..."):
                    try:
                        explanation = llm.generate_explanation(
                            prompt=filled_prompt,
                            api_key=st.session_state["openai_api_key"],
                        )
                        st.markdown("### Generated Explanation")
                        st.markdown(explanation)
                        st.success("✅ Explanation generated successfully!")
                    except ValueError as e:
                        st.error(f"❌ API Key Error: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Error generating explanation: {str(e)}")
        else:
            st.warning(
                "⚠️ OpenAI API key required. Please configure your API key in the Settings tab to use this feature."
            )

        st.info(
            " **Tip**: Copy this prompt and paste it into your preferred LLM (ChatGPT, Claude, etc.) to get a human-readable explanation of this prediction."
        )


# Tab 4: Model Performance Deep Dive
with tab4:
    st.markdown(
        '<div class="sub-header"> Model Performance Deep Dive</div>',
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
    st.markdown("###  Classification Metrics")
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
            f" **Model exceeds minimum requirement**: F1 Score {f1:.3f} ≥ 0.70 (project requirement)"
        )
    else:
        st.warning(
            f" **Performance below target**: F1 Score {f1:.3f} < 0.70 (project requirement)"
        )

    st.markdown("---")

    # Detailed confusion matrix with metrics
    st.markdown("###  Confusion Matrix Analysis")

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
                textfont={"size": 14, "color": "white"},
                colorscale=[[0, "#1a472a"], [0.5, "#3DCD58"], [1, "#7FE89A"]],
                showscale=True,
            )
        )

        fig.update_layout(
            title="Confusion Matrix (Count & %)",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
            title_font_color="#FFFFFF",
            title_font_size=18,
            xaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
            yaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
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
    st.markdown("###  Model Discrimination Curves")

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
                line=dict(color="#3DCD58", width=3),
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
            title_font_color="#FFFFFF",
            title_font_size=18,
            xaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
            yaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
            legend=dict(font_color="#FFFFFF"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
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
                line=dict(color="#3DCD58", width=3),
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
            title_font_color="#FFFFFF",
            title_font_size=18,
            xaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
            yaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
            legend=dict(font_color="#FFFFFF"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Performance by Confidence Threshold
    st.markdown("###  Performance by Prediction Threshold")

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
            line=dict(color="#3DCD58", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=threshold_df["Threshold"],
            y=threshold_df["Precision"],
            mode="lines+markers",
            name="Precision",
            line=dict(color="#009530", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=threshold_df["Threshold"],
            y=threshold_df["Recall"],
            mode="lines+markers",
            name="Recall",
            line=dict(color="#7FE89A", width=2),
        )
    )

    # Mark current threshold (0.5)
    current_thresh_idx = np.argmin(np.abs(threshold_df["Threshold"] - 0.5))
    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="#FF5252",
        annotation_text="Current (0.5)",
        annotation_position="top",
        annotation_font_color="#FFFFFF",
    )

    fig.update_layout(
        title="Metrics vs. Classification Threshold",
        xaxis_title="Threshold",
        yaxis_title="Score",
        height=400,
        hovermode="x unified",
        title_font_color="#FFFFFF",
        title_font_size=18,
        xaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
        yaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
        legend=dict(font_color="#FFFFFF"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        " **Insight**: Adjusting the classification threshold can optimize for precision (fewer false positives) or recall (fewer false negatives) based on business needs."
    )

    st.markdown("---")

    # Performance Segmentation Analysis
    st.markdown("###  Performance by Segments")

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
                marker_color="#3DCD58",
            )
        )

        fig.add_trace(
            go.Bar(
                x=segments.index,
                y=segments["Accuracy"],
                name="Accuracy",
                marker_color="#009530",
            )
        )

        fig.update_layout(
            title=f"Performance by {segment_option}",
            xaxis_title="Segment",
            yaxis_title="Score",
            barmode="group",
            height=400,
            title_font_color="#FFFFFF",
            title_font_size=18,
            xaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
            yaxis=dict(title_font_color="#FFFFFF", tickfont_color="#FFFFFF"),
            legend=dict(font_color="#FFFFFF"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Key insights
    best_segment = segments["F1"].idxmax()
    worst_segment = segments["F1"].idxmin()

    st.success(
        f" **Best Performance**: {best_segment} segment (F1 = {segments.loc[best_segment, 'F1']:.3f})"
    )
    st.warning(
        f" **Needs Improvement**: {worst_segment} segment (F1 = {segments.loc[worst_segment, 'F1']:.3f})"
    )

# Tab 5: Settings
with tab5:
    st.markdown(
        '<div class="sub-header"> Application Settings</div>', unsafe_allow_html=True
    )

    st.markdown("### API Configuration")

    if "openai_api_key" in st.session_state and st.session_state["openai_api_key"]:
        st.info("ℹ️ OpenAI API key loaded from environment variable.")
    else:
        st.warning(
            " No OpenAI API key found in environment variables. Please enter your API key below to enable LLM features."
        )

        # OpenAI API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key to enable LLM features. Your key is not stored permanently.",
        )

        if api_key:
            st.success(" API Key provided")
            # Store in session state for potential use
            st.session_state["openai_api_key"] = api_key
        else:
            st.info("ℹ️ No API key provided. Some features may be limited.")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.write(
        "This application provides explainability tools for sales opportunity predictions."
    )
    st.write("**Version**: 1.0.0")
    st.write("**Model**: CatBoost Classifier")
    st.write("**Framework**: Streamlit + SHAP")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Schneider Electric - GTM Machine Learning Explainability | "
    "Built with Streamlit 🚀"
    "</div>",
    unsafe_allow_html=True,
)
