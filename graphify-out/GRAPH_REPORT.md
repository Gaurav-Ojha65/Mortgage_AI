# Graph Report - .  (2026-04-18)

## Corpus Check
- 56 files · ~64,465 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 376 nodes · 481 edges · 43 communities detected
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_API Logging|API Logging]]
- [[_COMMUNITY_Risk Decision Engine|Risk Decision Engine]]
- [[_COMMUNITY_Model Evaluation|Model Evaluation]]
- [[_COMMUNITY_Model Routing|Model Routing]]
- [[_COMMUNITY_Credit Risk Production|Credit Risk Production]]
- [[_COMMUNITY_Credit Risk v2|Credit Risk v2]]
- [[_COMMUNITY_Frontend API Client|Frontend API Client]]
- [[_COMMUNITY_Fair Credit Risk|Fair Credit Risk]]
- [[_COMMUNITY_Dashboard UI|Dashboard UI]]
- [[_COMMUNITY_Home Credit Pipeline|Home Credit Pipeline]]
- [[_COMMUNITY_SHAP Explainability|SHAP Explainability]]
- [[_COMMUNITY_Credit Risk Fairness|Credit Risk Fairness]]
- [[_COMMUNITY_Monte Carlo Simulation|Monte Carlo Simulation]]
- [[_COMMUNITY_ML Artifacts|ML Artifacts]]
- [[_COMMUNITY_Model Metrics|Model Metrics]]
- [[_COMMUNITY_Model Fixed|Model Fixed]]
- [[_COMMUNITY_Risk Assessment|Risk Assessment]]
- [[_COMMUNITY_EMI Calculator|EMI Calculator]]
- [[_COMMUNITY_Features Fixed|Features Fixed]]
- [[_COMMUNITY_Loan Data Real|Loan Data Real]]
- [[_COMMUNITY_Monte Carlo Fixed|Monte Carlo Fixed]]
- [[_COMMUNITY_Risk Fixed|Risk Fixed]]
- [[_COMMUNITY_Credit Risk Production Data|Credit Risk Production Data]]
- [[_COMMUNITY_Evaluation Fixed|Evaluation Fixed]]
- [[_COMMUNITY_Best Model|Best Model]]
- [[_COMMUNITY_Advisor System|Advisor System]]
- [[_COMMUNITY_Model Training|Model Training]]
- [[_COMMUNITY_Risk Analysis|Risk Analysis]]
- [[_COMMUNITY_Features Engineering|Features Engineering]]
- [[_COMMUNITY_Database Models|Database Models]]
- [[_COMMUNITY_Constants|Constants]]
- [[_COMMUNITY_App Entry|App Entry]]
- [[_COMMUNITY_Index|Index]]
- [[_COMMUNITY_Report Web Vitals|Report Web Vitals]]
- [[_COMMUNITY_Setup Tests|Setup Tests]]
- [[_COMMUNITY_API Tests|API Tests]]
- [[_COMMUNITY_Tailwind Config|Tailwind Config]]
- [[_COMMUNITY_Design System|Design System]]
- [[_COMMUNITY_Docker Backend|Docker Backend]]
- [[_COMMUNITY_Docker Frontend|Docker Frontend]]
- [[_COMMUNITY_Docker Compose|Docker Compose]]
- [[_COMMUNITY_Nginx Config|Nginx Config]]
- [[_COMMUNITY_Environment Config|Environment Config]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 14 edges
2. `main()` - 12 edges
3. `main()` - 12 edges
4. `main()` - 12 edges
5. `main()` - 10 edges
6. `main()` - 9 edges
7. `create_response()` - 8 edges
8. `apply_tiers()` - 8 edges
9. `get_ollama_advice()` - 7 edges
10. `analyze_loan()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `LoanApplication` --inherits--> `BaseModel`  [EXTRACTED]
  api.py →   _Bridges community 0 → community 10_
- `ApplicantInput` --inherits--> `BaseModel`  [EXTRACTED]
  model_router.py →   _Bridges community 10 → community 3_

## Communities

### Community 0 - "API Logging"
Cohesion: 0.06
Nodes (39): analyze_application(), compare_loan_amounts(), create_response(), get_decisions_history(), get_errors(), get_history(), get_memory_usage(), get_uptime_seconds() (+31 more)

### Community 1 - "Risk Decision Engine"
Cohesion: 0.14
Nodes (24): apply_tiers(), business_cost_at_tiers(), calibrate_oof(), compute_shap(), engineer_features(), evaluate_tiered(), fairness_audit_tiered(), load_home_credit() (+16 more)

### Community 2 - "Model Evaluation"
Cohesion: 0.1
Nodes (19): load_model_and_data(), plot_confusion_matrix(), plot_feature_importance(), plot_precision_recall_curve(), plot_roc_curve(), print_classification_report(), print_model_card(), Professional Model Evaluation Report for Mortgage Approval Model Generates metri (+11 more)

### Community 3 - "Model Routing"
Cohesion: 0.16
Nodes (17): _active_model_name(), analyze(), ApplicantInput, compare_all_models(), feature_importance(), get_comparison(), list_models(), _load() (+9 more)

### Community 4 - "Credit Risk Production"
Cohesion: 0.19
Nodes (18): calibrate_probabilities(), compute_shap(), engineer_features(), evaluate(), fairness_audit(), find_optimal_thresholds(), load_data(), main() (+10 more)

### Community 5 - "Credit Risk v2"
Cohesion: 0.22
Nodes (16): calibrate_probabilities(), compute_shap(), engineer_features(), evaluate(), fairness_audit(), find_optimal_threshold(), load_data(), main() (+8 more)

### Community 6 - "Frontend API Client"
Cohesion: 0.15
Nodes (5): compareLoans(), normalizeCompareResponse(), normalizeDecision(), normalizeRiskLevel(), normalizeScenarioItem()

### Community 7 - "Fair Credit Risk"
Cohesion: 0.24
Nodes (15): calibrate_probabilities(), compute_shap(), engineer_features(), evaluate(), fairness_audit(), find_optimal_thresholds(), load_data(), main() (+7 more)

### Community 8 - "Dashboard UI"
Cohesion: 0.18
Nodes (14): analyze_loan(), create_ai_advice(), create_comparison_chart(), create_decision_banner(), create_history_table(), create_input_with_slider(), create_metric_cards(), create_monte_carlo_chart() (+6 more)

### Community 9 - "Home Credit Pipeline"
Cohesion: 0.23
Nodes (15): build_features(), cross_validate(), evaluate(), fairness_audit(), feature_importance(), find_optimal_thresholds(), handle_imbalance(), load_data() (+7 more)

### Community 10 - "SHAP Explainability"
Cohesion: 0.2
Nodes (13): ErrorReport, Client-side error report., BaseModel, _active(), ApplicantInput, explain(), explain_compare(), _load() (+5 more)

### Community 11 - "Credit Risk Fairness"
Cohesion: 0.18
Nodes (13): engineer_features(), load_and_preprocess_data(), predict(), prepare_training_data(), Mortgage Loan Approval ML Model - Real World Data Training Trains on real loan a, Prepare features and target from preprocessed dataframe., Train multiple models and return best one., Save model and test set for reproducible evaluation. (+5 more)

### Community 12 - "Monte Carlo Simulation"
Cohesion: 0.23
Nodes (12): analyze(), build_prompt(), get_ollama_advice(), parse_ollama_response(), Parse JSON from Ollama response, handling markdown code fences., Call Ollama and return structured advice dict., Full mortgage decision pipeline.     Returns structured JSON with decision, metr, Validate and normalise loan application input. Raises ValueError on bad data. (+4 more)

### Community 13 - "ML Artifacts"
Cohesion: 0.27
Nodes (12): build_features(), evaluate(), fairness_audit(), feature_importance(), find_optimal_threshold(), handle_imbalance(), load_data(), main() (+4 more)

### Community 14 - "Model Metrics"
Cohesion: 0.18
Nodes (11): generate_synthetic_data(), predict(), print_comparison_table(), Mortgage Loan Approval ML Model Comparison Compares Logistic Regression, Random, Train all three models and return evaluation results., Print a clean comparison table of all models., Save the best performing model to disk., Load best_model.pkl and make a prediction.      Args:         input_dict: Dictio (+3 more)

### Community 15 - "Model Fixed"
Cohesion: 0.2
Nodes (3): load_model_and_test_set(), Model Evaluation - Fixed version using saved test set., Load trained model and saved test set (not regenerated data!).

### Community 16 - "Risk Assessment"
Cohesion: 0.39
Nodes (7): clamp(), DecisionExplainer(), FactorRow(), pct(), signed(), WaterfallChart(), WhatIfSimulator()

### Community 17 - "EMI Calculator"
Cohesion: 0.32
Nodes (7): explain_decision(), _get_explainer(), _plain_english(), SHAP Explainability Engine Computes feature contributions for XGBoost, LightGBM,, Turn top SHAP factors into human-readable sentences., Build or return cached SHAP explainer for the given model., Given one applicant's features and a trained model, return:       - shap_values:

### Community 18 - "Features Fixed"
Cohesion: 0.33
Nodes (5): engineer_features(), get_feature_names(), Advanced feature engineering for mortgage loan applications. Transforms raw inpu, Engineer features from raw loan application data.      Args:         data: Dicti, Returns all feature column names in order for model training/inference.      Ret

### Community 19 - "Loan Data Real"
Cohesion: 0.33
Nodes (5): engineer_features(), get_feature_names(), Feature engineering - Fixed version with validation and consistent schema., Engineer features from raw loan application data.     Raises ValueError for inva, Returns feature columns in exact order for model inference.

### Community 20 - "Monte Carlo Fixed"
Cohesion: 0.33
Nodes (5): plot_simulation(), Monte Carlo Simulation for Mortgage Loan Default Risk Vectorized simulation of 1, Run Monte Carlo simulation to estimate loan default probability.      Args:, Create 2x2 subplot visualization of Monte Carlo simulation results.      Subplot, simulate()

### Community 21 - "Risk Fixed"
Cohesion: 0.33
Nodes (1): ErrorBoundary

### Community 22 - "Credit Risk Production Data"
Cohesion: 0.5
Nodes (3): calculate_emi(), EMI (Equated Monthly Installment) calculator., Calculate EMI for a loan.      Args:         P: Principal loan amount (must be p

### Community 23 - "Evaluation Fixed"
Cohesion: 0.5
Nodes (3): Monte Carlo Simulation - Fixed version with proper seeding., Run Monte Carlo simulation to estimate loan default probability.      Fixed: see, simulate()

### Community 24 - "Best Model"
Cohesion: 0.5
Nodes (3): calculate_risk(), Risk Assessment - Fixed version using constants and loan_term factor., Calculate risk level using constants from constants.py.     Now includes loan_te

### Community 25 - "Advisor System"
Cohesion: 1.0
Nodes (1): Application constants - centralized configuration.

### Community 26 - "Model Training"
Cohesion: 1.0
Nodes (0): 

### Community 27 - "Risk Analysis"
Cohesion: 1.0
Nodes (1): Multi-Model Credit Scoring Trainer Trains LogisticRegression (baseline), XGBoost

### Community 28 - "Features Engineering"
Cohesion: 1.0
Nodes (0): 

### Community 29 - "Database Models"
Cohesion: 1.0
Nodes (0): 

### Community 30 - "Constants"
Cohesion: 1.0
Nodes (0): 

### Community 31 - "App Entry"
Cohesion: 1.0
Nodes (0): 

### Community 32 - "Index"
Cohesion: 1.0
Nodes (0): 

### Community 33 - "Report Web Vitals"
Cohesion: 1.0
Nodes (0): 

### Community 34 - "Setup Tests"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "API Tests"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "Tailwind Config"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "Design System"
Cohesion: 1.0
Nodes (0): 

### Community 38 - "Docker Backend"
Cohesion: 1.0
Nodes (0): 

### Community 39 - "Docker Frontend"
Cohesion: 1.0
Nodes (0): 

### Community 40 - "Docker Compose"
Cohesion: 1.0
Nodes (0): 

### Community 41 - "Nginx Config"
Cohesion: 1.0
Nodes (0): 

### Community 42 - "Environment Config"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **112 isolated node(s):** `Validate and normalise loan application input. Raises ValueError on bad data.`, `Build the full prompt sent to Ollama.`, `Parse JSON from Ollama response, handling markdown code fences.`, `Call Ollama and return structured advice dict.`, `Full mortgage decision pipeline.     Returns structured JSON with decision, metr` (+107 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Advisor System`** (2 nodes): `constants.py`, `Application constants - centralized configuration.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Model Training`** (2 nodes): `risk.py`, `calculate_risk()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Risk Analysis`** (2 nodes): `train_models.py`, `Multi-Model Credit Scoring Trainer Trains LogisticRegression (baseline), XGBoost`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Features Engineering`** (2 nodes): `App()`, `App.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Database Models`** (2 nodes): `Compare()`, `Compare.jsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Constants`** (2 nodes): `Dashboard()`, `Dashboard.jsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `App Entry`** (2 nodes): `DecisionResult.jsx`, `DecisionResult()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Index`** (2 nodes): `GaugeChart.jsx`, `GaugeChart()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Report Web Vitals`** (2 nodes): `History()`, `History.jsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Setup Tests`** (2 nodes): `LoanForm.jsx`, `LoanForm()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `API Tests`** (2 nodes): `MetricCard.jsx`, `MetricCard()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tailwind Config`** (2 nodes): `ModelComparison()`, `ModelComparison.jsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Design System`** (2 nodes): `MonteCarlo3D.jsx`, `MonteCarlo3D()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Docker Backend`** (2 nodes): `Navigation.jsx`, `Navigation()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Docker Frontend`** (2 nodes): `useApi.js`, `useApi()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Docker Compose`** (1 nodes): `postcss.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Nginx Config`** (1 nodes): `tailwind.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Environment Config`** (1 nodes): `index.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ApplicantInput` connect `Model Routing` to `SHAP Explainability`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **Why does `LoanApplication` connect `API Logging` to `SHAP Explainability`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **Why does `ErrorReport` connect `SHAP Explainability` to `API Logging`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **What connects `Validate and normalise loan application input. Raises ValueError on bad data.`, `Build the full prompt sent to Ollama.`, `Parse JSON from Ollama response, handling markdown code fences.` to the rest of the system?**
  _112 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `API Logging` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._
- **Should `Risk Decision Engine` be split into smaller, more focused modules?**
  _Cohesion score 0.14 - nodes in this community are weakly interconnected._
- **Should `Model Evaluation` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._