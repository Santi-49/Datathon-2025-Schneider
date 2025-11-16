# Example LLM Prompt Output

This document shows an example of what the generated prompt looks like for a specific instance.

---

## Example Instance: Test Instance #42

### Input Data
- **Prediction**: WON (1)
- **Confidence**: 87.3%
- **Actual Outcome**: WON
- **Prediction Status**: ✅ Correct

### Feature Values
- **Historical sales of Product A with this customer**: 0.6595
- **Historical sales of Product B with this customer**: 3.3531
- **Whether Product A was recommended in the past to this customer**: -0.1097
- **Amount of Product A in this opportunity**: -0.0891
- **Amount of Product C in this opportunity**: -0.0237
- **Amount of Product D in this opportunity**: -0.0425
- **Customer success rate in previous interactions**: 1.5509
- **Number of interactions with the customer**: -0.6817
- **Number of contracts signed with the customer**: 3.4654
- **Month when the opportunity was created**: 1.2440
- **Whether the opportunity has been open for a long time**: -0.2819
- **Presence of competitor Z in the opportunity**: 0
- **Presence of competitor X in the opportunity**: 0
- **Presence of competitor Y in the opportunity**: 0
- **Whether the customer is located in Iberia**: 1

---

## Generated Prompt (Sample)

```
You are an expert data scientist specializing in explainable AI for sales opportunity prediction at Schneider Electric.

## CONTEXT
Schneider Electric uses a machine learning model to predict whether sales opportunities will be WON or LOST. The model analyzes historical CRM data to help sales teams prioritize their efforts and understand key success factors.

## YOUR TASK
Explain why the model made a specific prediction for a sales opportunity. Your explanation should be clear enough for non-technical stakeholders (sales managers, account executives) to understand and act upon.

## OPPORTUNITY DETAILS
- **Opportunity ID**: Test Instance #42
- **Predicted Outcome**: 1 (WON)
- **Prediction Confidence**: 87.3%
- **Actual Outcome**: WON

## FEATURE VALUES
- **Historical sales of Product A with this customer**: 0.6595
- **Historical sales of Product B with this customer**: 3.3531
- **Whether Product A was recommended in the past to this customer**: -0.1097
- **Amount of Product A in this opportunity**: -0.0891
- **Amount of Product C in this opportunity**: -0.0237
- **Amount of Product D in this opportunity**: -0.0425
- **Customer success rate in previous interactions**: 1.5509
- **Number of interactions with the customer**: -0.6817
- **Number of contracts signed with the customer**: 3.4654
- **Month when the opportunity was created**: 1.2440
- **Whether the opportunity has been open for a long time**: -0.2819
- **Presence of competitor Z in the opportunity**: 0
- **Presence of competitor X in the opportunity**: 0
- **Presence of competitor Y in the opportunity**: 0
- **Whether the customer is located in Iberia**: 1

## MODEL EXPLANATION (SHAP VALUES)
SHAP values indicate how much each feature pushed the prediction toward WON (positive values) or LOST (negative values):

- **Number of contracts signed with the customer**: +0.2347 (pushes toward WON)
- **Customer success rate in previous interactions**: +0.1852 (pushes toward WON)
- **Historical sales of Product B with this customer**: +0.1234 (pushes toward WON)
- **Whether the customer is located in Iberia**: +0.0892 (pushes toward WON)
- **Historical sales of Product A with this customer**: +0.0456 (pushes toward WON)
- **Month when the opportunity was created**: -0.0123 (pushes toward LOST)
- **Number of interactions with the customer**: -0.0234 (pushes toward LOST)
- **Amount of Product A in this opportunity**: -0.0156 (pushes toward LOST)
- **Whether the opportunity has been open for a long time**: -0.0098 (pushes toward LOST)
- **Presence of competitor Z in the opportunity**: -0.0045 (pushes toward LOST)
- **Product A recommended in the past**: -0.0034 (pushes toward LOST)
- **Amount of Product C in this opportunity**: -0.0012 (pushes toward LOST)
- **Amount of Product D in this opportunity**: -0.0009 (pushes toward LOST)
- **Presence of competitor X in the opportunity**: -0.0003 (pushes toward LOST)
- **Presence of competitor Y in the opportunity**: -0.0001 (pushes toward LOST)

**Base Prediction Rate**: 45.2% (average probability across all opportunities)
**Final Prediction**: 87.3%

## TOP 3 MOST INFLUENTIAL FACTORS
1. **Number of contracts signed with the customer** (SHAP: +0.2347)
   - This feature strongly pushes the prediction toward WON

2. **Customer success rate in previous interactions** (SHAP: +0.1852)
   - This feature strongly pushes the prediction toward WON

3. **Historical sales of Product B with this customer** (SHAP: +0.1234)
   - This feature strongly pushes the prediction toward WON

## INSTRUCTIONS
Provide a clear, business-focused explanation that addresses:

1. **Decision Summary**: In 2-3 sentences, explain why the model predicted this opportunity as WON.

2. **Key Drivers**: Identify and explain the 3 most important factors that influenced this prediction. For each factor:
   - Explain what the feature means in business terms
   - Describe how its value (high/low/present) affected the prediction
   - Suggest what this means for sales strategy

3. **Risk Factors or Opportunities**: Highlight any concerning patterns (for WON predictions) or potential bright spots (for LOST predictions) that sales teams should be aware of.

4. **Actionable Recommendations**: Provide 2-3 specific actions the sales team could take based on this prediction and its explanation.

5. **Confidence Assessment**: Comment on the model's confidence level and whether there are any contradictory signals in the data.

Keep your language clear, avoid technical jargon, and focus on actionable business insights.
```

---

## How to Use This Prompt

1. **Copy the entire prompt** from the Streamlit app
2. **Paste into an LLM** (ChatGPT, Claude, Gemini, etc.)
3. **Receive a human-readable explanation** that includes:
   - Why the model made this decision
   - What factors were most important
   - What sales teams should do next
   - Assessment of prediction reliability

---

## Expected LLM Response (Example)

When you paste the above prompt into an LLM, you might receive something like:

> ### Decision Summary
> The model predicted this opportunity as WON with high confidence (87.3%) primarily because this customer has an excellent track record with Schneider Electric. They have signed multiple contracts in the past (3.46 standard deviations above average) and maintain a strong success rate in their interactions (1.55 standard deviations above average). Additionally, there is substantial historical engagement with Product B, indicating a proven relationship.
>
> ### Key Drivers
>
> **1. Number of Contracts Signed (SHAP: +0.2347)**
> This customer has a significantly higher number of signed contracts compared to typical customers. This is the single strongest predictor of success in this opportunity. Past contract history demonstrates trust, satisfaction, and a proven ability to close deals with this customer.
> 
> *Sales Strategy Implication*: Leverage the existing relationship. Reference past successful projects and emphasize continuity and proven results.
>
> **2. Customer Success Rate (SHAP: +0.1852)**
> The customer's hit rate in previous interactions is well above average, meaning they typically follow through on opportunities and convert at a high rate. This behavioral pattern strongly suggests they are a serious buyer.
>
> *Sales Strategy Implication*: Prioritize this opportunity. The customer's history shows they don't waste time on opportunities they won't pursue. Allocate senior resources to this deal.
>
> **3. Historical Sales of Product B (SHAP: +0.1234)**
> Strong historical sales of Product B indicate familiarity with Schneider Electric's solutions and satisfaction with previous purchases. This builds credibility and reduces perceived risk for new purchases.
>
> *Sales Strategy Implication*: Use Product B success stories and testimonials from this same customer. Consider upselling or cross-selling opportunities.
>
> ### Risk Factors
> While the prediction is highly confident, there are minor negative signals:
> - The opportunity has been open for a relatively short time (not flagged as "old"), which is actually positive
> - No competitor presence detected, which eliminates one common risk factor
> - The slight decrease in recent interactions could indicate cooling interest, though this is minimal
>
> ### Actionable Recommendations
> 1. **Accelerate the sales cycle**: Given the high confidence and strong customer relationship, move quickly to proposal and negotiation stages
> 2. **Leverage past successes**: Prepare case studies and ROI analyses from previous contracts with this customer
> 3. **Assign a senior account manager**: This is a high-value opportunity that deserves top-tier attention
>
> ### Confidence Assessment
> The model's 87.3% confidence is well-justified. All major indicators align positively, with no significant contradictory signals. The customer's proven track record makes this a reliable prediction. Sales teams can confidently invest resources in closing this deal.

---

## Benefits of This Approach

1. **Non-technical explanations**: Sales teams understand WHY without needing data science knowledge
2. **Actionable insights**: Clear recommendations on what to do next
3. **Transparency**: Shows model reasoning, building trust
4. **Scalability**: Can generate explanations for thousands of opportunities
5. **Customizable**: Adjust the prompt template for different audiences or use cases

---

## Tips for Best Results

- **Select interesting cases**: High confidence with unusual feature patterns
- **Compare WON vs LOST**: Understand what differentiates successful opportunities
- **Focus on actionable features**: Some features can be influenced by sales actions
- **Validate with experts**: Ensure LLM interpretations align with business reality
- **Iterate the template**: Refine the prompt based on user feedback

