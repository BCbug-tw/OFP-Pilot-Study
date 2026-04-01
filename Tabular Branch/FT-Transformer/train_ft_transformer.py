import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import rtdl
from tqdm import tqdm

def main():
    # 1. Load the dataset
    print("Loading data...")
    df = pd.read_csv("data/diabetes_binary_subset_20k.csv")
    
    # 2. Preprocess the data
    print("Preprocessing data...")
    # Target variable
    y = df['Diabetes_binary'].values.astype(np.float32)
    
    # Features
    X = df.drop(columns=['Diabetes_binary'])
    
    # Identify categorical and numerical features based on the dataset description
    # Most features in this dataset are binary or ordinal (categorical)
    # BMI, MentHlth, PhysHlth are continuous/numerical
    numerical_features = ['BMI', 'MentHlth', 'PhysHlth']
    categorical_features = [col for col in X.columns if col not in numerical_features]
    
    # We need to treat categorical features properly for FT-Transformer.
    # rtdl expects categorical features to be integers (indices)
    for col in categorical_features:
        X[col] = X[col].astype(int)
    
    # Split into train, val, test (70%, 15%, 15%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp)
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_features])
    X_val_num = scaler.transform(X_val[numerical_features])
    X_test_num = scaler.transform(X_test[numerical_features])
    
    # Extract categorical features
    X_train_cat = X_train[categorical_features].values
    X_val_cat = X_val[categorical_features].values
    X_test_cat = X_test[categorical_features].values
    
    # Get the number of categories for each categorical feature (required for FT-Transformer)
    cat_cardinalities = [len(X[col].unique()) for col in categorical_features]
    # Sometimes categorical values might not start at 0 continuously. Let's remap them to be safe.
    from sklearn.preprocessing import OrdinalEncoder
    ord_encoder = OrdinalEncoder()
    X_train_cat = ord_encoder.fit_transform(X_train_cat).astype(int)
    X_val_cat = ord_encoder.transform(X_val_cat).astype(int)
    X_test_cat = ord_encoder.transform(X_test_cat).astype(int)
    
    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset dictionaries
    data = {
        'train': {
            'x_num': torch.tensor(X_train_num, dtype=torch.float32).to(device),
            'x_cat': torch.tensor(X_train_cat, dtype=torch.long).to(device),
            'y': torch.tensor(y_train, dtype=torch.float32).to(device)
        },
        'val': {
            'x_num': torch.tensor(X_val_num, dtype=torch.float32).to(device),
            'x_cat': torch.tensor(X_val_cat, dtype=torch.long).to(device),
            'y': torch.tensor(y_val, dtype=torch.float32).to(device)
        },
        'test': {
            'x_num': torch.tensor(X_test_num, dtype=torch.float32).to(device),
            'x_cat': torch.tensor(X_test_cat, dtype=torch.long).to(device),
            'y': torch.tensor(y_test, dtype=torch.float32).to(device)
        }
    }
    
    # 3. Define the FT-Transformer Model
    print("Initializing FT-Transformer...")
    model = rtdl.FTTransformer.make_default(
        n_num_features=len(numerical_features),
        cat_cardinalities=cat_cardinalities,
        last_layer_query_idx=[-1],  # Query the [CLS] token
        d_out=1,  # Binary classification
    ).to(device)
    
    optimizer = (
        model.make_default_optimizer()
        if isinstance(model, rtdl.FTTransformer)
        else torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    )
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    # 4. Training Loop
    def apply_model(x_num, x_cat):
        return model(x_num, x_cat).squeeze()
    
    batch_size = 512
    epochs = 100
    
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Simple permissive mini-batching
        indices = torch.randperm(len(data['train']['x_num']))
        
        pbar = tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for i in pbar:
            batch_idx = indices[i:i+batch_size]
            
            optimizer.zero_grad()
            
            x_num_batch = data['train']['x_num'][batch_idx]
            x_cat_batch = data['train']['x_cat'][batch_idx]
            y_batch = data['train']['y'][batch_idx]
            
            logits = apply_model(x_num_batch, x_cat_batch)
            loss = loss_fn(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_idx)
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(data['train']['x_num'])
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = apply_model(data['val']['x_num'], data['val']['x_cat'])
            val_loss = loss_fn(val_logits, data['val']['y']).item()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'FT-Transformer/ft_transformer_model.pt')
                
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    print("Training finished.")
    
    # 5. Testing
    print("\nTesting best model...")
    # model.load_state_dict(torch.save)  # fix: torch.load
    model.load_state_dict(torch.load('FT-Transformer/ft_transformer_model.pt', map_location=device))
    model.eval()
    
    with torch.no_grad():
        test_logits = apply_model(data['test']['x_num'], data['test']['x_cat'])
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_probs > 0.5).astype(int)
        y_test_np = data['test']['y'].cpu().numpy()
        
    acc = accuracy_score(y_test_np, test_preds)
    auc = roc_auc_score(y_test_np, test_probs)
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")
    
    report = classification_report(y_test_np, test_preds, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test_np, test_preds))
    
    
    # Save test results
    results_df = pd.DataFrame({
        'True_Label': y_test_np,
        'Predicted_Label': test_preds,
        'Predicted_Probability': test_probs
    })
    results_df.to_csv('FT-Transformer/test_predictions.csv', index=False)
    print("Saved test predictions to FT-Transformer/test_predictions.csv")
    
    # Save evaluation metrics
    eval_metrics = {
        'Accuracy': [acc],
        'ROC_AUC': [auc],
        'Precision_0': [report['0.0']['precision']],
        'Recall_0': [report['0.0']['recall']],
        'F1_0': [report['0.0']['f1-score']],
        'Precision_1': [report['1.0']['precision']],
        'Recall_1': [report['1.0']['recall']],
        'F1_1': [report['1.0']['f1-score']],
        'Macro_F1': [report['macro avg']['f1-score']]
    }
    eval_df = pd.DataFrame(eval_metrics)
    eval_df.to_csv('FT-Transformer/test_eval.csv', index=False)
    print("Saved evaluation metrics to FT-Transformer/test_eval.csv")


if __name__ == "__main__":
    main()
