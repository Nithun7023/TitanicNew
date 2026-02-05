import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Configuration
# ----------------------------
RANDOM_STATE = 42


def load_and_explore_data():
    """
    Loads the Titanic dataset and performs basic exploration.
    """
    print("--- 1. Loading Dataset ---")

    # Load Titanic dataset from seaborn
    df = sns.load_dataset('titanic')

    # Display basic information
    print(f"Dataset Shape: {df.shape}")
    print(df.head())

    # Visualize missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()

    return df


def preprocess_data(df):
    """
    Cleans the dataset and prepares features for modeling.
    """
    print("\n--- 2. Data Preprocessing ---")

    # Fill missing age values with median
    df['age'] = df['age'].fillna(df['age'].median())

    # Fill missing embarked values with most frequent value
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

    # Drop columns with too many missing values or redundancy
    df = df.drop(columns=['deck', 'embark_town', 'alive'], errors='ignore')

    # Drop any remaining rows with missing values
    df = df.dropna()

    # Encode categorical variables
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df['embarked'] = le.fit_transform(df['embarked'])

    # Select input features and target variable
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    X = df[features]
    y = df['survived']

    print("Selected Features:", features)
    return X, y


def train_model(X, y):
    """
    Trains a decision tree classifier.
    """
    print("\n--- 3. Model Training ---")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Initialize and train decision tree model
    model = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model, X_test, y_test, X.columns


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluates model performance and visualizes results.
    """
    print("\n--- 4. Model Evaluation ---")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Print classification metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    # Plot feature importance
    importances = model.feature_importances_
    plt.figure(figsize=(8, 4))
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importance")
    plt.show()

    # Visualize the decision tree
    plt.figure(figsize=(15, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=['Died', 'Survived'],
        filled=True
    )
    plt.title("Decision Tree Structure")
    plt.show()


if __name__ == "__main__":
    # Run the full pipeline
    raw_df = load_and_explore_data()
    X, y = preprocess_data(raw_df)
    model, X_test, y_test, feature_names = train_model(X, y)
    evaluate_model(model, X_test, y_test, feature_names)
