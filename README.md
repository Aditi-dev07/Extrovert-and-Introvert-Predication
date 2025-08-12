# 🧠 Personality Prediction (Introvert vs Extrovert) using Ensemble Machine Learning

This project predicts whether a person is an **Introvert** or **Extrovert** based on behavioral and social interaction features.  
It combines **feature engineering**, **missing value imputation**, **hyperparameter tuning**, and **ensemble learning** techniques like **Voting** and **Stacking Classifiers** for improved accuracy.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Preprocessing](#-preprocessing)
- [Modeling](#-modeling)
- [Ensembling](#-ensembling)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output](#-output)
- [License](#-license)

## 📖 Overview
The pipeline includes:
1. **Data Cleaning** – Standardizing column names and formats.  
2. **Missing Value Handling** – Using advanced imputers for better accuracy.  
3. **Feature Engineering** – Creating new variables to improve predictive power.  
4. **Model Training** – Using **Random Forest**, **Gradient Boosting**, and **XGBoost**.  
5. **Hyperparameter Tuning** – Using **GridSearchCV** with Stratified K-Folds.  
6. **Ensemble Learning** – Combining models using Voting and Stacking.  
7. **Final Prediction** – Selecting the best-performing ensemble for submission.

---

## 📂 Dataset
- **Train Data (`train.csv`)** – Contains features + target (`personality`)
- **Test Data (`test.csv`)** – Contains features only

**Target Variable Mapping**  
- `Introvert` → **0**  
- `Extrovert` → **1**

---

## 🧾 Features

### Original Features
- `time_spent_alone` – Hours spent alone daily  
- `social_event_attendance` – Event attendance frequency  
- `going_outside` – Frequency of going outdoors  
- `friends_circle_size` – Number of friends  
- `post_frequency` – Social media posting frequency  
- `stage_fear` – Yes/No  
- `drained_after_socializing` – Yes/No  

### Engineered Features
- `alone_x_post` – Alone time × Post frequency  
- `friends_per_outing` – Friends per outing  
- `activity_index` – Social activity score  
- `normalized_alone_time` – Alone time normalized by friends count  
- `social_anxiety_score` – Stage fear × Drained feeling  
- `log_friends` – Log transformation of friends count  
- `log_post_frequency` – Log transformation of post frequency  
- `social_engagement` – Combined social activity measure  
- `alone_vs_social` – Ratio of alone time to social engagement  
- `social_avoidance` – Combined avoidance score  

---

## 🛠 Preprocessing
- **Column Cleaning** → Lowercase names & underscores for spaces  
- **Missing Value Imputation**:
  - Numerical: `KNNImputer(n_neighbors=5)`  
  - Categorical: `SimpleImputer(strategy='most_frequent')`  
- **Encoding**:
  - `Yes`/`No` → **1/0**  
  - `Introvert`/`Extrovert` → **0/1**  
- **Scaling** → `StandardScaler` for numerical features

---

## 🤖 Modeling
Models tuned using **GridSearchCV** with 5-fold **StratifiedKFold**:
- 🌳 **Random Forest Classifier**
- 📈 **Gradient Boosting Classifier**
- ⚡ **XGBoost Classifier**

---

## 🔗 Ensembling
Two ensemble strategies:
1. **Voting Classifier (Soft Voting)** – Weighted by validation accuracy  
2. **Stacking Classifier** – Logistic Regression meta-learner with passthrough features  

✅ **Best model** is selected based on validation accuracy.

---

## 📊 Results
- Best hyperparameters for each model  
- Validation accuracy for **Voting** & **Stacking**  
- Confusion matrix for the final selected model  
