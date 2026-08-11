# Credit Score Classification

**R | Machine Learning | Credit Analytics | Model Validation**

[View Recruiter Summary PDF](docs/credit_score_classification_recruiter_summary.pdf) | [View Original Presentation](docs/credit_score_classification_original_presentation_ko.pdf) | [View R Modelling Code](src/credit_score_classification.R)

## Project Snapshot

Machine-learning project developed in **R** to predict customers' **next-month credit-score classes** using historical financial and behavioural data.

The project analysed **100K+ credit records across 12,500 customers** and benchmarked **329 individual, blending and ensemble model configurations** to identify the strongest credit-scoring approach.

### Key Results

| Metric | Result |
|---|---:|
| Dataset size | 100K+ credit records |
| Customer base | 12,500 customers |
| Best next-month classification accuracy | ~82.5% |
| Model configurations benchmarked | 329 |
| Validation approach | Customer-level 5-fold cross-validation |
| Main tools | R, Random Forest, XGBoost, LightGBM, Elastic Net |

![Credit Score Key Charts](images/credit_score_key_charts_summary.png)

## Business Problem

Credit scores are important for financial institutions when assessing customer creditworthiness. The goal of this project was to use historical customer and credit information to predict future credit-score classes and support more data-driven credit assessment.

The project explored two prediction scenarios:

1. **Existing customers:** predict the customer's next-month credit-score class.
2. **New customers:** sequentially predict future credit-score classes over an eight-month horizon.

## Dataset

The dataset contains monthly customer-level credit observations, including historical financial and behavioural variables such as payment behaviour, credit limits, credit inquiries and previous credit-score information.

> The raw dataset is not included in this repository. Add the original data source only if redistribution is permitted.

## Methodology

The project followed a structured machine-learning workflow:

1. Data preparation and feature engineering
2. Customer-level train/test splitting
3. Feature selection
4. Hyperparameter tuning
5. Customer-level 5-fold cross-validation
6. Individual model comparison
7. Blending and weighted ensemble modelling
8. Evaluation using accuracy and a cost-sensitive credit-scoring metric

## Models Evaluated

The following model families were evaluated:

- Elastic Net
- Random Forest
- XGBoost
- LightGBM
- Blending models
- Weighted ensemble models

A total of **329 model configurations** were benchmarked:

- 4 individual model families
- 2 blending approaches
- 323 weighted ensemble configurations

## Validation Strategy

Instead of randomly splitting rows, the project used **customer-level validation**. This prevents the same customer's records from appearing in both training and validation folds, making the evaluation more realistic for credit-scoring use cases.

## Results

### Next-Month Credit-Score Classification

The best model achieved approximately **82.5% classification accuracy** for next-month credit-score class prediction.

Feature selection improved overall classification performance, and Random Forest / ensemble-based approaches showed strong predictive performance.

### Sequential Prediction for New Customers

The project also evaluated sequential prediction for customers without previous database records:

- Month 1 prediction accuracy: **75.1%**
- Month 8 cumulative prediction accuracy: **68.2%**

## Cost-Sensitive Evaluation

In credit scoring, not all classification errors have the same business impact. Misclassifying a high-risk customer as a low-risk customer can be more costly than the reverse case.

The project therefore evaluated models not only with overall accuracy but also with a cost-sensitive metric that assigns greater importance to high-risk misclassification.

## Business Relevance

This project demonstrates how machine learning can support financial institutions by:

- Identifying changes in customer credit profiles
- Supporting data-driven credit assessment
- Comparing alternative predictive models
- Considering the business cost of classification errors
- Providing forward-looking information for credit-risk monitoring

## Project Files

| File | Description |
|---|---|
| [Recruiter Summary PDF](docs/credit_score_classification_recruiter_summary.pdf) | Visual portfolio summary for recruiters and hiring managers |
| [Original Presentation](docs/credit_score_classification_original_presentation_ko.pdf) | Full project presentation in Korean |
| [R Modelling Code](src/credit_score_classification.R) | Final R modelling script |
| [Key Charts Summary](images/credit_score_key_charts_summary.png) | One-page visual summary of the most important charts |

## CV Bullet Points

**Credit Score Classification | R, Machine Learning**  
- Analysed 100K+ credit records across 12,500 customers and developed machine-learning models to predict next-month credit-score classes, achieving ~82.5% classification accuracy.  
- Benchmarked 329 individual, blending and ensemble model configurations using Elastic Net, Random Forest, XGBoost and LightGBM, with customer-level 5-fold cross-validation and feature selection to identify the best-performing credit-scoring approach.

## Tools & Technologies

**R, glmnet, randomForest, xgboost, lightgbm, dplyr, tidyr, ggplot2, machine learning, classification modelling, cross-validation, feature selection, ensemble modelling**

## Project Context

Academic team project completed as part of a Data Mining course. This repository is structured as a professional portfolio version of the project for data analytics, BI analytics and financial analytics applications.
