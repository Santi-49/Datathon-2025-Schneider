# Sales Opportunity Prediction - Explainability Dashboard

![Schneider Electric](media/Schneider-Electric-logo-jpg-.png)

**AI-Powered Sales Intelligence Platform for Schneider Electric**

An interactive Streamlit dashboard providing explainable AI insights for sales opportunity predictions using CatBoost and SHAP values.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Technology Stack](#technology-stack)
- [License](#license)

---

##  Overview

This project delivers an explainability dashboard for a machine learning model that predicts sales opportunity outcomes (WON/LOST) for Schneider Electric's Go-to-Market strategy. The dashboard provides:

- **Interactive visualizations** of model predictions and performance
- **SHAP-based explainability** showing which features influence each prediction
- **AI-generated natural language explanations** using OpenAI's GPT models
- **Comprehensive performance metrics** and analysis tools

---

##  Features

###  Dataset Overview
- Visual statistics and distribution of predictions
- Confusion matrix with performance metrics
- Confidence distribution analysis

###  Explore Predictions
- Filter predictions by outcome, correctness, and confidence
- Global feature importance analysis (SHAP values)
- Interactive data exploration

###  Generate AI Prompts
- Instance-level prediction explanations
- SHAP value visualizations
- LLM-ready prompts for detailed explanations
- One-click AI explanation generation with OpenAI

###  Model Performance
- Comprehensive classification metrics (F1, Precision, Recall, ROC-AUC)
- ROC and Precision-Recall curves
- Performance segmentation analysis
- Threshold optimization tools

###  Settings
- OpenAI API key configuration
- Application information

---

##  Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/Santi-49/Datathon-2025-Schneider.git
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Configuration

### 1. Environment Variables

Create a `.env` file in the project root directory:

```bash
# .env
OPEN_AI_API_KEY=sk-your-openai-api-key-here
```

**Note**: The OpenAI API key is optional. The dashboard works without it, but AI-generated explanations will be disabled.

### 2. Streamlit Theme Configuration

The project includes a dark theme configuration in `.streamlit/config.toml`. The theme is automatically applied when you run the app.

### 3. Generate Predictions (First Time Setup)

Before running the dashboard, you need to generate predictions with SHAP values:

```bash
python model.py
```

This will:
- Train a CatBoost classifier on `data/train.csv`
- Generate predictions for `data/X_test.csv`
- Calculate SHAP values for explainability
- Save outputs to:
  - `data/predictions_with_shap.csv`
  - `predictions_detailed.json`
  - `model/catboost_model.joblib`

---

##  Usage

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

### Alternative: Using Python Directly

```bash
python -m streamlit run app.py
```

### Dashboard Navigation

1. **Dataset Overview Tab**: View overall statistics and model performance
2. **Explore Predictions Tab**: Filter and analyze predictions, view global feature importance
3. **Generate LLM Prompt Tab**: Select specific instances for detailed explanations
4. **Model Performance Tab**: Deep dive into performance metrics and segmentation
5. **Settings Tab**: Configure OpenAI API key and view application info

---

##  File Descriptions

### Core Application Files

#### `app.py`
The main Streamlit application that provides the interactive dashboard.

**Key Features**:
- Multi-tab interface for different analysis views
- Interactive visualizations using Plotly
- SHAP value displays for prediction explainability
- Integration with OpenAI for AI-generated explanations
- Schneider Electric branded UI with dark theme

**Main Sections**:
- Data loading and caching
- Sidebar filters for prediction exploration
- Tab 1: Dataset statistics and confusion matrix
- Tab 2: Prediction explorer with global feature importance
- Tab 3: Instance-level explanations with SHAP analysis
- Tab 4: Comprehensive performance metrics
- Tab 5: Settings and configuration

#### `model.py`
Handles model training, prediction generation, and SHAP value calculation.

**Functionality**:
- Loads and preprocesses training data (`data/train.csv`)
- Trains a CatBoost classifier with optimized hyperparameters
- Generates predictions for test data (`data/X_test.csv`)
- Calculates SHAP values for each prediction
- Exports results to CSV and JSON formats
- Saves trained model for future use

**Outputs**:
- `data/predictions_with_shap.csv`: Predictions with SHAP values
- `predictions_detailed.json`: Detailed prediction information
- `model/catboost_model.joblib`: Trained model artifact

**To Run**:
```bash
python model.py
```

#### `llm.py`
Provides OpenAI integration for generating natural language explanations.

**Functionality**:
- Interfaces with OpenAI's GPT models (GPT-4)
- Converts technical SHAP explanations into business-friendly language
- Handles API key validation and error management
- Configurable temperature and token limits

**Key Function**:
```python
generate_explanation(prompt: str, api_key: str) -> str
```

**Usage**:
- Called from `app.py` when user requests AI explanations
- Requires valid OpenAI API key in environment or session state
- Returns markdown-formatted explanations

### Configuration Files

#### `requirements.txt`
Python package dependencies required to run the project.

**Key Dependencies**:
- `streamlit`: Web application framework
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning utilities
- `catboost`: Gradient boosting classifier
- `shap`: Model explainability
- `plotly`: Interactive visualizations
- `openai`: OpenAI API client
- `python-dotenv`: Environment variable management

#### `.streamlit/config.toml`
Streamlit configuration for theme and app settings.

**Configuration**:
- Sets dark theme as default
- Customizes Schneider Electric brand colors
- Configures layout and display options

### Data Files

#### `data/train.csv`
Training dataset containing historical sales opportunities with features and outcomes.

#### `data/X_test.csv`
Test dataset containing features for prediction (no outcomes included).

#### `data/predictions_with_shap.csv`
Generated by `model.py` - contains predictions and SHAP values for each test instance.

#### `predictions_detailed.json`
Generated by `model.py` - detailed prediction information in JSON format for the dashboard.

### Template Files

#### `templates/prompt_template.txt`
Template for generating instance-level explanation prompts for LLMs.

**Variables**:
- Opportunity details (ID, prediction, confidence)
- Feature values
- SHAP explanations
- Top influential factors

#### `templates/global_importance_prompt.txt`
Template for generating global feature importance explanation prompts.

**Variables**:
- Feature importance rankings
- Aggregated SHAP values
- Feature descriptions

---

##  Technology Stack

### Machine Learning
- **CatBoost**: Gradient boosting classifier for predictions
- **SHAP**: Model-agnostic explainability framework
- **Scikit-learn**: Data preprocessing and metrics

### Visualization & UI
- **Streamlit**: Interactive web dashboard framework
- **Plotly**: Interactive charts and visualizations
- **HTML/CSS**: Custom styling for Schneider Electric branding

### AI Integration
- **OpenAI GPT-4**: Natural language explanation generation
- **Python-dotenv**: Environment variable management

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations

---

##  Customization

### Changing Brand Colors

Edit the CSS in `app.py` to change the color scheme:

```python
st.markdown(
    f"""
<style>
    :root {{
        --se-green: #3DCD58;        /* Primary green */
        --se-dark-green: #009530;   /* Dark green */
        --se-light-green: #7FE89A;  /* Light green */
    }}
</style>
""",
    unsafe_allow_html=True,
)
```

### Adding New Features

1. **New visualization**: Add in the appropriate tab in `app.py`
2. **New metric**: Calculate in the performance section
3. **New filter**: Add to sidebar in `app.py`

---

##  Model Performance

The CatBoost model is trained to predict sales opportunity outcomes with:
- **Target**: Binary classification (WON vs LOST)
- **Features**: 15 business-relevant features including customer history, product mix, and competitive landscape
- **Minimum F1 Score Requirement**: 0.70

Performance is continuously monitored through the dashboard's Performance tab.

---

##  Contributing

This project was developed for the Schneider Electric Challenge in the 2025 Datathon.

### How to Contribute

1.  **Fork the repository**
2.  **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3.  **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4.  **Push to the branch** (`git push origin feature/AmazingFeature`)
5.  **Open a Pull Request**

---

**Built with ⚡ by Team Quick2 Datathon 2025**

*Life Is On - Driving efficiency and sustainability through intelligent analytics *

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

