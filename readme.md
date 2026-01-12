🍄 Mushroom Classification using Machine Learning
📌 Project Overview
This project implements a complete end-to-end machine learning pipeline to classify mushrooms as edible or poisonous using the Mushroom Classification Dataset from Kaggle. The dataset consists entirely of categorical features describing physical mushroom characteristics such as odor, cap shape, gill size, stalk surface, and spore print color.
The pipeline covers:
Data preprocessing and exploratory analysis
Feature engineering and statistical testing
Model training and hyperparameter tuning
Evaluation using multiple performance metrics
Model interpretability using feature importance, SHAP, and LIME
The primary goal was to identify the best-performing model and understand which features most strongly predict mushroom toxicity.
📊 Dataset Summary
Instances: 8,124
Features: 23 categorical attributes
Target Variable:
e → Edible
p → Poisonous
Class Distribution:
Edible: ~52%
Poisonous: ~48%
The near-balanced classes eliminate the need for resampling and reduce bias during training.
🧪 Methodology
1. Data Preprocessing
Checked for missing values and inconsistencies.
The stalk-root feature contained missing entries, which were treated as a separate "missing" category to preserve data.
Removed irrelevant features such as veil-type, which was constant across all samples.
Performed descriptive statistics and visualizations (count plots, frequency analysis).
Observed that odor had the strongest correlation with mushroom toxicity.
2. Feature Engineering
All categorical features were transformed using One-Hot Encoding (OHE).
Feature space expanded from 22 features to 114 encoded columns.
Dataset split into:
80% training
20% testing
Stratified splitting ensured class balance.
3. Statistical Feature Selection
Applied Chi-Square tests to evaluate feature independence with the target.
Features with low p-values (e.g., odor, gill-size) were retained.
Features with high p-values (e.g., gill-attachment) were removed.
4. Models Implemented
Six supervised classification models were trained and evaluated:
Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Decision Tree
Random Forest
Gradient Boosting
Each model was tuned using GridSearchCV with 5-fold cross-validation.
📈 Evaluation Metrics
Models were evaluated using multiple metrics to ensure robustness:
Accuracy
Precision
Recall
F1-Score
Cohen’s Kappa
ROC Curve and AUC
Precision–Recall Curve
Both 70–30 and 80–20 train-test splits were tested. Performance consistently improved with larger training sets.
🏆 Results
Model Performance
Logistic Regression, KNN, SVM, Decision Tree, Random Forest:
100% Accuracy
Gradient Boosting:
99.88% Accuracy
Most models achieved:
Precision, Recall, F1-Score, Kappa, and ROC AUC = 1.0
ROC curves largely overlapped, indicating near-perfect separability between edible and poisonous mushrooms.
After Hyperparameter Tuning
All models improved or maintained perfect performance.
Hyperparameter tuning consistently outperformed base models.
🔍 Model Interpretability
Feature Importance (Random Forest)
Top predictive features:
Odor
Gill size
Stalk surface (above & below ring)
Spore print color
SHAP Analysis
Confirmed that odor is by far the most influential feature.
Certain odors (e.g., foul, pungent) strongly indicate poisonous mushrooms.
Results align with biological knowledge and expert expectations.
LIME Explanations
Individual predictions were explained using LIME.
Influential features included:
Gill size
Gill color
Spore print color
Stalk surface above the ring
Odor
Other features had minimal impact on predictions.
⚖️ Imbalanced Data Experiment
When experimenting with artificially imbalanced data:
Accuracy remained high.
Precision, Recall, and F1-score for the minority class showed slight declines.
Demonstrates why accuracy alone is insufficient for imbalanced datasets.
💬 Discussion
The dataset is extremely clean and well-structured, leading to near-perfect classification.
High-cardinality categorical features significantly increased dimensionality.
Careful handling was required to align OHE features with SHAP explanations.
GridSearchCV became computationally expensive due to high dimensionality.
Perfect accuracy suggests strong feature separability, which may not fully generalize to real-world mushroom data containing noise and ambiguity.
✅ Conclusion
This project successfully demonstrates a robust machine learning pipeline for mushroom classification. Multiple models achieved perfect or near-perfect performance, confirming that mushroom toxicity is highly predictable using categorical attributes.
Future Improvements
Experiment with alternative encoders to reduce dimensionality.
Test robustness on noisier or real-world datasets.
Evaluate generalization on unseen mushroom species.
Overall, this pipeline is both highly accurate and interpretable, making it a strong example of applied machine learning.
