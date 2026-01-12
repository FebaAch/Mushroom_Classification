Mushroom Classification (Edible vs Poisonous)
End-to-end machine learning pipeline for classifying mushrooms as edible or poisonous using categorical features from the Kaggle Mushroom Dataset.
Dataset
Samples: 8,124
Features: 23 categorical attributes
Target:
e → Edible
p → Poisonous
Class Balance: ~52% edible / ~48% poisonous
Features describe physical characteristics such as odor, gill size, cap shape, stalk surface, and spore print color.
Pipeline
Data cleaning and exploratory analysis
Missing value handling (stalk-root treated as "missing")
Removal of non-informative features (veil-type)
One-Hot Encoding (22 → 114 features)
Chi-Square feature selection
Stratified train–test split (80/20, 70/30)
Model training and hyperparameter tuning (GridSearchCV)
Evaluation and interpretability analysis
Models
Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Decision Tree
Random Forest
Gradient Boosting
Evaluation Metrics
Accuracy
Precision
Recall
F1-Score
Cohen’s Kappa
ROC Curve & AUC
Precision–Recall Curve
Results
Model	Accuracy
Logistic Regression	100%
KNN	100%
SVM	100%
Decision Tree	100%
Random Forest	100%
Gradient Boosting	99.88%
Most models achieved perfect or near-perfect performance
ROC curves largely overlap due to strong class separability
Performance improved with larger training splits
Interpretability
Random Forest Feature Importance
Odor
Gill size
Stalk surface (above & below ring)
Spore print color
SHAP
Odor is the most decisive feature
Certain odors strongly indicate poisonous mushrooms
LIME
Individual predictions influenced mainly by odor, gill size, and stalk features
Notes
Dataset is clean and highly separable
High dimensionality due to One-Hot Encoding
Perfect accuracy may not generalize to real-world mushroom data
