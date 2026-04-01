import pandas as pd
import numpy as np
import os

def main():
    input_file = "data/diabetes_binary_health_indicators_BRFSS2015.csv"
    output_file = "data/diabetes_binary_subset_20k.csv"
    
    print(f"Loading data from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return
        
    df = pd.read_csv(input_file)
    
    # Target ratio 4:6 (8k positive, 12k negative)
    pos_target = 8000
    neg_target = 12000
    
    pos_df = df[df['Diabetes_binary'] == 1.0]
    neg_df = df[df['Diabetes_binary'] == 0.0]
    
    print(f"Original positive samples: {len(pos_df)}")
    print(f"Original negative samples: {len(neg_df)}")
    
    # Sample without replacement
    if len(pos_df) >= pos_target:
        pos_sampled = pos_df.sample(n=pos_target, random_state=42)
    else:
        print(f"Warning: Not enough positive samples. Expected {pos_target}, got {len(pos_df)}. Using all.")
        pos_sampled = pos_df
        
    if len(neg_df) >= neg_target:
        neg_sampled = neg_df.sample(n=neg_target, random_state=42)
    else:
        print(f"Warning: Not enough negative samples. Expected {neg_target}, got {len(neg_df)}. Using all.")
        neg_sampled = neg_df
        
    # Concatenate and shuffle
    subset_df = pd.concat([pos_sampled, neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Created subset with shape: {subset_df.shape}")
    print(f"Positive samples in subset: {len(subset_df[subset_df['Diabetes_binary'] == 1.0])}")
    print(f"Negative samples in subset: {len(subset_df[subset_df['Diabetes_binary'] == 0.0])}")
    
    subset_df.to_csv(output_file, index=False)
    print(f"Saved subset to {output_file}")

if __name__ == "__main__":
    main()
