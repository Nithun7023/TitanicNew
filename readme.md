# Titanic Survival Prediction Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange) ![Status](https://img.shields.io/badge/Precision-78%25-success)

**Titanic Survival Prediction Engine** is a machine learning project that analyzes passenger demographics to predict survival probabilities. Built with **Python** and **Scikit-learn**, it features a robust **Decision Tree Classifier** optimized for high precision and interpretability.

---

## 📌 Resume Highlights & Project Goals

* **Robust Classification:** Engineered a Decision Tree model (max_depth=3) to prevent overfitting, achieving a validated **78% precision** on the test set.
* **Data Engineering:** Performed extensive **Exploratory Data Analysis (EDA)** to identify missing data patterns. Implemented median imputation for `Age` and mode imputation for `Embarked`.
* **Feature Engineering:** Normalized categorical variables (Sex, Embarked) using Label Encoding to create a machine-readable feature matrix.
* **Analysis:** Documented feature importance, identifying `Sex` and `Pclass` as the primary determinants of survival.

---

## 🛠️ Tech Stack

* **Core:** Python 3.x
* **Machine Learning:** Scikit-learn (DecisionTreeClassifier)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib

---

## 🚀 Installation & Usage

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Engine:**
    ```bash
    python titanic_engine.py
    ```

3.  **Data Source:**
    * This project utilizes the **inbuilt Titanic dataset** from the `seaborn` library. No manual CSV download is required.

---

## 📊 Evaluation Results

* **Precision:** ~78% (Weighted Avg)
* **Key Insight:** The visualization confirms that "Women and Children First" was a statistically significant factor, with `Sex_female` being the root node of the decision tree.

---
