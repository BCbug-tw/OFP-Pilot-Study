import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

def main():
    print("Loading data...")
    df = pd.read_csv("data/diabetes_binary_subset_20k.csv")
    
    y = df['Diabetes_binary'].values
    X = df.drop(columns=['Diabetes_binary'])
    
    numerical_features = ['BMI', 'MentHlth', 'PhysHlth']
    categorical_features = [col for col in X.columns if col not in numerical_features]
    
    # Split: 70% Train, 15% Val, 15% Test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp)
    
    # Preprocessing
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_features])
    X_val_num = scaler.transform(X_val[numerical_features])
    X_test_num = scaler.transform(X_test[numerical_features])
    
    ord_encoder = OrdinalEncoder()
    # LightGBM prefers integer-encoded categorical features
    X_train_cat = ord_encoder.fit_transform(X_train[categorical_features]).astype(int)
    X_val_cat = ord_encoder.transform(X_val[categorical_features]).astype(int)
    X_test_cat = ord_encoder.transform(X_test[categorical_features]).astype(int)
    
    # Recombine into DataFrame to preserve integer types for categorical features
    X_train_processed = pd.DataFrame(X_train_num, columns=numerical_features)
    X_train_processed[categorical_features] = X_train_cat
    
    X_val_processed = pd.DataFrame(X_val_num, columns=numerical_features)
    X_val_processed[categorical_features] = X_val_cat
    
    X_test_processed = pd.DataFrame(X_test_num, columns=numerical_features)
    X_test_processed[categorical_features] = X_test_cat
    
    cat_feature_indices = categorical_features
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='binary',
        early_stopping_rounds=50
    )
    
    print("Training LightGBM...")
    # fit method signature
    model.fit(
        X_train_processed, y_train,
        eval_set=[(X_val_processed, y_val)],
        eval_metric='binary_logloss',
        categorical_feature=cat_feature_indices
    )
    
    print("Evaluating LightGBM model...")
    test_probs = model.predict_proba(X_test_processed)[:, 1]
    test_preds = model.predict(X_test_processed)
    
    acc = accuracy_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_probs)
    ap = average_precision_score(y_test, test_probs)
    
    print(f"Test Accuracy: {acc:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
    
    os.makedirs("LightGBM", exist_ok=True)
    
    # Save model
    model.booster_.save_model("LightGBM/lightgbm_model.txt")
    print("Saved model to LightGBM/lightgbm_model.txt")
    
    results_df = pd.DataFrame({
        'GroundTruth_Diabetes_binary': y_test,
        'Prob_Diabetes_binary': test_probs,
        'Pred_Diabetes_binary': test_preds
    })
    out_path = "LightGBM/test_predictions.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
