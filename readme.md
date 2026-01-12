#Machine Learning Pipeline for Mushroom Classification

## 1. Introduction

This project implements a complete machine learning pipeline for classifying mushrooms as **edible** or **poisonous** using the **Mushroom Classification Dataset** from Kaggle. The dataset contains categorical features describing physical characteristics such as cap shape, odor, gill color, stalk surface, and spore print color.

The pipeline includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training and hyperparameter tuning, evaluation using multiple metrics, and interpretability techniques such as feature importance, SHAP, and LIME. The goal is to identify the best-performing model and understand which features most strongly predict mushroom toxicity.

---

## 2. Data Summary

- **Total instances:** 8,124  
- **Total features:** 23 categorical features  
- **Target variable:**  
  - Edible (`e`)  
  - Poisonous (`p`)  

The dataset is nearly balanced, with approximately **52% edible** and **48% poisonous** mushrooms. This balance reduces the risk of bias during model training and eliminates the need for resampling techniques.

---

## 3. Methodology

### Data Preprocessing

- Checked for missing values and inconsistencies.
- The `stalk-root` feature contained missing values, which were treated as a separate `"missing"` category instead of removing rows.
- Constant features such as `veil-type` were removed due to lack of predictive value.
- Count plots and frequency analysis revealed that **odor** had the strongest correlation with mushroom toxicity.
- All categorical features were transformed using **One-Hot Encoding (OHE)**, expanding the feature space from 22 features to **114 encoded features**.
- The dataset was split into **80% training** and **20% testing**, stratified by the target variable.

### Feature Selection

- Chi-square tests were conducted to assess independence between each feature and the target.
- Features with low p-values (e.g., `odor`, `gill-size`) were retained.
- Features with high p-values (e.g., `gill-attachment`) were removed.

---

## 4. Models and Evaluation

### Models Implemented

Six supervised classification models were trained and evaluated:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  
- Gradient Boosting  

All models were tuned using **GridSearchCV** with **5-fold cross-validation**.

### Evaluation Metrics

To ensure robust evaluation, multiple metrics were used:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Cohen’s Kappa  
- ROC Curve and AUC  
- Precision–Recall Curve  

---

## 5. Results

- Logistic Regression, KNN, SVM, Decision Tree, and Random Forest achieved **100% accuracy**.
- Gradient Boosting achieved **99.88% accuracy**.
- Precision, Recall, F1-Score, Kappa, and ROC-AUC were perfect (1.0) for most models.
- Performance improved with higher training data proportions (80/20 split compared to 70/30).

After hyperparameter tuning, all models showed improved or stable performance. **Random Forest** was selected as the most robust model due to:
- Strong generalization
- Reduced overfitting compared to single decision trees
- Built-in feature importance for interpretability

---

## 6. Model Interpretability

- **Random Forest feature importance** identified:
  - Odor
  - Gill size
  - Stalk surface (above/below ring)
  - Spore print color  
- **SHAP analysis** confirmed odor as the most influential feature, with foul and pungent odors strongly indicating poisonous mushrooms.
- **LIME explanations** showed that gill size, gill color, spore print color, stalk surface, and odor had the greatest influence on individual predictions.

When tested on imbalanced versions of the dataset, accuracy remained high, but **precision, recall, and F1-score** for the minority class showed slight declines, emphasizing the importance of using multiple metrics.

---

## 7. Discussion

The pipeline demonstrates that mushroom toxicity is highly separable using categorical features alone. While perfect accuracy is impressive, it also suggests potential feature redundancy and limited real-world variability. The dataset is clean and lacks noise typically found in real-world data, which may limit generalizability.

High-dimensional feature space resulting from one-hot encoding increased computational cost, especially during GridSearch. Careful feature alignment was required when applying SHAP explanations.

---

## 8. Conclusion

This project successfully implemented an end-to-end machine learning pipeline for mushroom classification. Multiple models achieved perfect or near-perfect performance, with **Random Forest** offering the best balance of accuracy, robustness, and interpretability.

Future improvements may include testing alternative encoding methods to reduce dimensionality and evaluating performance on more realistic, noisy datasets.

---

## 9. Dataset Reference

Mushroom Classification Dataset  
Kaggle: https://www.kaggle.com/datasets/uciml/mushroom-classification
