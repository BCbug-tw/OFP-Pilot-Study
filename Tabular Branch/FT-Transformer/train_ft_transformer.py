import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import rtdl
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logging, plot_loss_from_log

OUTPUT_DIR = 'FT-Transformer'

def main():
    logger = setup_logging(OUTPUT_DIR)
    
    # 1. Load the dataset
    logger.info("Loading data...")
    df = pd.read_csv("data/diabetes_binary_subset_20k.csv")
    
    # 2. Preprocess the data
    logger.info("Preprocessing data...")
    y = df['Diabetes_binary'].values.astype(np.float32)
    X = df.drop(columns=['Diabetes_binary'])
    
    numerical_features = ['BMI', 'MentHlth', 'PhysHlth']
    categorical_features = [col for col in X.columns if col not in numerical_features]
    
    for col in categorical_features:
        X[col] = X[col].astype(int)
    
    # Split into train, val, test (70%, 15%, 15%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp)
    
    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_features])
    X_val_num = scaler.transform(X_val[numerical_features])
    X_test_num = scaler.transform(X_test[numerical_features])
    
    # Categorical features
    ord_encoder = OrdinalEncoder()
    X_train_cat = ord_encoder.fit_transform(X_train[categorical_features].values).astype(int)
    X_val_cat = ord_encoder.transform(X_val[categorical_features].values).astype(int)
    X_test_cat = ord_encoder.transform(X_test[categorical_features].values).astype(int)
    
    cat_cardinalities = [len(X[col].unique()) for col in categorical_features]
    
    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
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
    logger.info("Initializing FT-Transformer...")
    model = rtdl.FTTransformer.make_default(
        n_num_features=len(numerical_features),
        cat_cardinalities=cat_cardinalities,
        last_layer_query_idx=[-1],
        d_out=1,
    ).to(device)
    
    optimizer = (
        model.make_default_optimizer()
        if isinstance(model, rtdl.FTTransformer)
        else torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    )
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Initialize Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 4. Training Loop
    def apply_model(x_num, x_cat):
        return model(x_num, x_cat).squeeze()
    
    batch_size = 512
    epochs = 100
    
    logger.info(f"Starting training for {epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
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
                torch.save(model.state_dict(), f'{OUTPUT_DIR}/ft_transformer_model.pt')
                logger.info(f"New best model saved at epoch {epoch+1} (Val Loss: {val_loss:.4f})")
        
        # Step the scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr < old_lr:
            logger.info(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
                
        logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
    logger.info("Training finished.")

    # Plot Loss Curve from log
    plot_loss_from_log(
        log_path=os.path.join(OUTPUT_DIR, 'logger.log'),
        save_dir=OUTPUT_DIR,
        title='FT-Transformer Training and Validation Loss'
    )
    
    # 5. Testing
    logger.info("Testing best model...")
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/ft_transformer_model.pt', map_location=device))
    model.eval()
    
    with torch.no_grad():
        test_logits = apply_model(data['test']['x_num'], data['test']['x_cat'])
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_probs > 0.5).astype(int)
        y_test_np = data['test']['y'].cpu().numpy()
        
    acc = accuracy_score(y_test_np, test_preds)
    auc = roc_auc_score(y_test_np, test_probs)
    
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test ROC-AUC: {auc:.4f}")
    
    report = classification_report(y_test_np, test_preds, output_dict=True)
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test_np, test_preds))
    
    # Save test results
    results_df = pd.DataFrame({
        'True_Label': y_test_np,
        'Predicted_Label': test_preds,
        'Predicted_Probability': test_probs
    })
    results_df.to_csv(f'{OUTPUT_DIR}/test_predictions.csv', index=False)
    logger.info(f"Saved test predictions to {OUTPUT_DIR}/test_predictions.csv")
    
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
    eval_df.to_csv(f'{OUTPUT_DIR}/test_eval.csv', index=False)
    logger.info(f"Saved evaluation metrics to {OUTPUT_DIR}/test_eval.csv")


if __name__ == "__main__":
    main()
