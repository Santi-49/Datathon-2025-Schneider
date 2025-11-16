# Sales Opportunity Explainability System

## Overview
This system provides comprehensive explainability for sales opportunity predictions at Schneider Electric. It includes:
- A trained CatBoost model for predicting WON/LOST opportunities
- SHAP-based explainability for both global and local interpretations
- An interactive Streamlit dashboard for exploring predictions
- LLM prompt generation for human-readable explanations

## Project Structure

```
├── temp.py                          # Main training script with SHAP analysis
├── app.py                           # Streamlit dashboard application
├── prompt_template.txt              # LLM prompt template for explanations
├── requirements.txt                 # Python dependencies
├── train.csv                        # Training dataset
├── predictions_with_shap.csv        # Generated predictions with SHAP values
├── predictions_detailed.json        # Detailed predictions in JSON format
└── catboost_model.joblib           # Trained model
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Training Script
This will train the model and generate predictions with SHAP values:
```bash
python temp.py
```

**Output files:**
- `predictions_with_shap.csv` - CSV with all predictions and SHAP values
- `predictions_detailed.json` - Detailed JSON with structured prediction data
- `catboost_model.joblib` - Trained model
- Various PNG files with visualizations

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Using the Streamlit Dashboard

### Tab 1: Dataset Overview
- View overall statistics about predictions
- Explore prediction distribution and confidence levels
- Analyze confusion matrix

### Tab 2: Explore Predictions
- Filter predictions by:
  - Prediction outcome (WON/LOST)
  - Correctness (Correct/Incorrect)
  - Confidence level
- View global feature importance
- Browse filtered dataset

### Tab 3: Generate LLM Prompt
1. **Select an instance** from the filtered dataset
2. **View instance details**:
   - Prediction vs actual outcome
   - Feature values
   - SHAP value analysis with visualization
3. **Generated LLM Prompt**:
   - Automatically fills the prompt template with instance data
   - Includes all feature values and SHAP explanations
   - Ready to copy and paste into ChatGPT, Claude, or any LLM
4. **Download** the prompt as a text file

## LLM Prompt Template

The prompt template (`prompt_template.txt`) is designed to generate business-focused explanations that include:
- **Decision Summary**: Why the model made this prediction
- **Key Drivers**: Top 3 factors influencing the decision
- **Risk Factors/Opportunities**: Patterns to watch
- **Actionable Recommendations**: Specific actions for sales teams
- **Confidence Assessment**: Model certainty analysis

### Variables in Template:
- `{opportunity_id}` - Instance identifier
- `{prediction}` - Predicted class (0/1)
- `{prediction_label}` - Predicted label (WON/LOST)
- `{prediction_probability}` - Confidence percentage
- `{actual_outcome}` - Actual result
- `{feature_values}` - All feature values formatted
- `{shap_explanation}` - SHAP values with interpretations
- `{base_value}` - Model baseline probability
- `{top_factors}` - Top 3 most influential features

## Model Performance

The CatBoost model achieves:
- **Minimum F1 Score**: 0.7 (project requirement)
- **Cross-validation**: 5-fold stratified CV
- **Evaluation metrics**: F1, ROC-AUC, Precision, Recall

## Explainability Techniques Used

1. **SHAP (SHapley Additive exPlanations)**
   - Global feature importance
   - Local instance-level explanations
   - Force plots and summary plots

2. **LIME (Local Interpretable Model-agnostic Explanations)**
   - Alternative local explanations
   - HTML report generation

3. **Partial Dependence Plots (PDP)**
   - Feature effect visualization
   - Generated for top features

4. **LLM-Enhanced Interpretation**
   - Converts SHAP values to natural language
   - Business-focused insights
   - Non-technical stakeholder friendly

## Feature Descriptions

- **product_A_sold_in_the_past**: Historical sales of Product A with this customer
- **product_B_sold_in_the_past**: Historical sales of Product B with this customer
- **product_A_recommended**: Whether Product A was recommended in the past
- **product_A**: Amount of Product A in this opportunity
- **product_C**: Amount of Product C in this opportunity
- **product_D**: Amount of Product D in this opportunity
- **cust_hitrate**: Customer success rate in previous interactions
- **cust_interactions**: Number of interactions with the customer
- **cust_contracts**: Number of contracts signed with the customer
- **opp_month**: Month when the opportunity was created
- **opp_old**: Whether the opportunity has been open for a long time
- **competitor_Z**: Presence of competitor Z
- **competitor_X**: Presence of competitor X
- **competitor_Y**: Presence of competitor Y
- **cust_in_iberia**: Whether the customer is located in Iberia

## Workflow

1. **Train Model** (`temp.py`)
   - Loads training data
   - Trains CatBoost classifier
   - Evaluates performance
   - Generates SHAP values
   - Exports predictions with explanations

2. **Explore Data** (Streamlit Tab 1 & 2)
   - View dataset statistics
   - Filter predictions
   - Analyze feature importance

3. **Generate Explanations** (Streamlit Tab 3)
   - Select instance
   - View SHAP analysis
   - Generate LLM prompt
   - Copy to LLM for human explanation

4. **Get LLM Interpretation**
   - Paste prompt into ChatGPT/Claude/etc.
   - Receive business-focused explanation
   - Share with sales team

## Tips for Best Results

1. **Filter wisely**: Focus on high-confidence predictions or misclassifications
2. **Compare instances**: Look at both WON and LOST predictions
3. **Use SHAP charts**: Visual representation helps understand feature impacts
4. **Iterate prompts**: Modify the template for specific use cases
5. **Validate with domain experts**: Ensure LLM explanations align with business logic

## Customization

### Modify the Prompt Template
Edit `prompt_template.txt` to:
- Change the tone or style
- Add specific business context
- Focus on different aspects
- Include additional instructions

### Adjust Feature Descriptions
In `app.py`, update the `feature_descriptions` dictionary to:
- Use business-specific terminology
- Add more context
- Simplify technical terms

### Filter Options
Modify the sidebar in `app.py` to add:
- Additional filter criteria
- Different groupings
- Custom visualizations

## Troubleshooting

**Issue**: Files not found when running `app.py`
- **Solution**: Run `temp.py` first to generate prediction files

**Issue**: Memory error during SHAP computation
- **Solution**: Reduce test set size or use SHAP's sampling options

**Issue**: Streamlit won't start
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## Contact

For questions about this project, refer to PROJECT_INFO.md for mentor contacts.

---

**Built for the Schneider Electric Datathon Challenge** 🚀
