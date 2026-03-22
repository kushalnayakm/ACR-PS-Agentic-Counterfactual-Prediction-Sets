import dice_ml
from acr.data_loader import get_adult_data
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pandas as pd
import numpy as np
import json
import os

def run_baseline():
    print("Step 1: Running Baseline XAI on Adult Census Dataset...")
    
    # Load dataset using custom loader
    dataset = get_adult_data()

    target = 'income'
    
    # Define feature types
    numerical = ["age", "hours_per_week"]
    categorical = dataset.columns.difference(numerical + [target])
    
    # Split data
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Create DiCE data object
    d = dice_ml.Data(dataframe=train_dataset, 
                     continuous_features=numerical, 
                     outcome_name=target)
    
    # Train a model pipeline
    X_train = train_dataset.drop(target, axis=1)
    y_train = train_dataset[target]
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical),
            ('num', 'passthrough', numerical)])

    clf = Pipeline(steps=[('preprocessor', transformations),
                          ('classifier', RandomForestClassifier())])
    
    model = clf.fit(X_train, y_train)
    
    # Create DiCE model object
    m = dice_ml.Model(model=model, backend="sklearn")
    
    # Initialize DiCE
    exp = dice_ml.Dice(d, m, method="random")
    
    # Generate counterfactuals for 5 individuals who were predicted as <=50K
    query_instances = test_dataset[test_dataset[target] == 0].head(5).drop(target, axis=1)
    
    print(f"Generating counterfactuals for {len(query_instances)} instances...")
    
    dice_exp = exp.generate_counterfactuals(query_instances, total_CFs=3, desired_class="opposite")
    
    # Save results to JSON
    output_path = "acr/raw_counterfactuals.json"
    with open(output_path, 'w') as f:
        f.write(dice_exp.to_json())

    
    print(f"Baseline setup complete. Raw counterfactuals saved to {output_path}")

if __name__ == "__main__":
    run_baseline()

