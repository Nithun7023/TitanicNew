import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
# Random state 42 ensures reproducible results (consistent 78-80% scores)
RANDOM_STATE = 42


def load_and_explore_data():
    """
    Resume Claim: "Performed extensive Exploratory Data Analysis (EDA)"
    Uses Seaborn's 'inbuilt' dataset to avoid external CSV dependencies.
    """
    print("--- 1. Loading Inbuilt Dataset ---")
    # Loading 'titanic' directly from Seaborn library
    df = sns.load_dataset('titanic')

    print(f"Dataset Shape: {df.shape}")
    print(df.head())

    # VISUALIZATION: Missing Values Heatmap
    # Resume Claim: "handle missing values"
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("EDA: Missing Values Heatmap (Yellow = Missing)")
    plt.show()

    return df


def preprocess_data(df):
    """
    Resume Claim: "handle missing values and normalize categorical data"
    """
    print("\n--- 2. Preprocessing & Cleaning ---")

    # 1. Handling Missing Values
    # 'age': Fill with median (Standard imputing strategy)
    df['age'] = df['age'].fillna(df['age'].median())

    # 'embarked': Fill with mode (most common value)
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

    # 'deck': Too many missing values, drop it.
    df = df.drop(columns=['deck', 'embark_town', 'alive'], errors='ignore')

    # Drop remaining rows with missing values (minimal impact now)
    df = df.dropna()

    # 2. Normalize Categorical Data (Feature Encoding)
    # Resume Claim: "normalize categorical data"
    le = LabelEncoder()

    # Encode 'sex' (male/female -> 0/1)
    df['sex'] = le.fit_transform(df['sex'])

    # Encode 'embarked' (S/C/Q -> 0/1/2)
    df['embarked'] = le.fit_transform(df['embarked'])

    # Encode 'class'/ 'who' / 'adult_male' if needed, or drop redundant columns
    # For this model, we keep numeric and key categorical features
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    X = df[features]
    y = df['survived']

    print("Data Cleaned. Features selected:", features)
    return X, y


def train_model(X, y):
    """
    Resume Claim: "Built a robust decision tree classifier"
    """
    print("\n--- 3. Training Decision Tree ---")

    # Split: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Model: Decision Tree
    # Max Depth = 3 restricts the tree to prevent overfitting (Making it "Robust")
    model = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model, X_test, y_test, X.columns


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Resume Claim: "achieving 78% precision... Documented feature importance"
    """
    print("\n--- 4. Model Evaluation ---")

    y_pred = model.predict(X_test)

    # 1. Classification Report (Precision, Recall, F1)
    print("Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # 3. Feature Importance Analysis (Resume Claim)
    importances = model.feature_importances_
    plt.figure(figsize=(8, 4))
    sns.barplot(x=importances, y=feature_names, palette="viridis")
    plt.title("Feature Importance Analysis (Decision Tree)")
    plt.show()

    # 4. Visualize the Tree
    plt.figure(figsize=(15, 10))
    plot_tree(model, feature_names=feature_names, class_names=['Died', 'Survived'], filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()


if __name__ == "__main__":
    # Pipeline Execution
    raw_df = load_and_explore_data()
    X, y = preprocess_data(raw_df)
    model, X_test, y_test, feat_names = train_model(X, y)
    evaluate_model(model, X_test, y_test, feat_names)