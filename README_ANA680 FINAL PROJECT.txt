# ANA680 FINAL PROJECT - CUSTOMER CHURN PREDICTION

## Problem

Predict which telecom subscribers are likely to cancel their service, so a provider can target retention offers before the customers leave.

## Dataset

Telco Customer Churn (IBM sample dataset, distributed via Kaggle). There are 7,043 customer sevice records across 21 columns. The target is 'Churn', a binary label. This is an imbalanced classification problem because 26.54% of customers churned.

## Key Findings

month-to month contract | 42.7%
one year contract | 11.3%
two year contract | 2.8%
Fiber optic internet | 41.9%
DSL internet | 19.0%

Churned customers averaged 18 months tenure versus 37.6 months for those who stayed.

## Models Compared

Data was split 60/20/20 into training, validation, and test sets, stratified on the target. Three models were compared on the validation set:

Logistic Regression| ACC: 0.746| Recall 0.79| Precision 0.51| F1 0.62
Random Forest| ACC: 0.769| Recall 0.63| Precision 0.56| F1 0.59
Gradient Boosting| ACC: 0.803| Recall 0.49| Precision 0.68| F1 0.57

Accuracy and recall run in opposite direction across the three models. Gradient Boosting was the most accurate and the least useful because it missed 189 out of 374 churners. Logistic Regression was select as the model to deploy based on the recall and F1 rather than accuracy.

The final test set performance was: 74.0% accuracy, 0.78 recall, 0.51 precision, 0.61 F1. Validation and test results were nearly identical. This indicates the model generalizes and was not overfit during selection.

