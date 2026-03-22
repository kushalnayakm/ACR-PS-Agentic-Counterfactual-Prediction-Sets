import pandas as pd
import dice_ml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import json
import os

def run_diabetes_baseline():
    print("Step 1: Generating Raw Counterfactuals for Diabetes...")
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    
    # Target is Outcome (0 or 1)
    target = 'Outcome'
    X = df.drop(target, axis=1)
    y = df[target]

    # 1. Train a simple model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # We use a pipeline for easier integration with DiCE
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    clf.fit(X_train, y_train)

    # 2. Setup DiCE
    d = dice_ml.Data(dataframe=df, continuous_features=list(X.columns), outcome_name=target)
    m = dice_ml.Model(model=clf, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")

    # 3. Generate CFs for a patient WITH diabetes (Outcome=1)
    test_instance = X_test[y_test == 1].iloc[:1]
    dice_exp = exp.generate_counterfactuals(test_instance, total_CFs=5, desired_class=0)
    
    # Save the raw output
    os.makedirs("acr/diabetes", exist_ok=True)
    with open("acr/diabetes/raw_cf.json", 'w') as f:
        f.write(dice_exp.to_json())
    
    print("Raw counterfactuals saved to acr/diabetes/raw_cf.json")

if __name__ == "__main__":
    run_diabetes_baseline()
