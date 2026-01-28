'''
Code for Estimating Treatment Effect using TarNet from GPI-Pack
Dataset: 6th & 10th Legislative Yuan Data
Aurhor: CHIEH YA-TSUN (113252017@nccu.edu.tw)
'''

from __future__ import annotations
import pandas as pd
import numpy as np
import torch
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import re

try:
    from gpi_pack.TarNet import TarNet
except ImportError:
    raise ImportError("gpi_pack not found. Please ensure it is installed and in the python path.")

# ===================================================================
# 0. Setting random seed
# ===================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================================================================
# 1. Configuration
# ===================================================================
AUGMENTATION_FACTOR = 1  

THEME_TREATMENT = "公務人員"
TARGET_THEMES = ["原住民", "身心障礙", "國軍", "教師", "農民", "勞工", "婦女", "兒少"]

# Path settings
BASE_DIR_ROOT = r"C:\Users\yachun\Desktop\RR_RR"
PATH_DATA_CSV = os.path.join(BASE_DIR_ROOT, "data", "6TH_LY.csv")
PATH_R_RAW = os.path.join(BASE_DIR_ROOT, "data", "True_Embedding", "R_embedding_combined_6th.pt")
SYNTHETIC_DIR = os.path.join(BASE_DIR_ROOT, "synthetic_data", "6th_LY") 

# Output paths
OUTPUT_ROOT = os.path.join(BASE_DIR_ROOT, "models", "Final_Analysis_6th")
OUTPUT_INDIVIDUAL = os.path.join(OUTPUT_ROOT, "Individual_Reports")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(OUTPUT_INDIVIDUAL, exist_ok=True)

print(f"=== TarNet Analysis of the 6th LY (Factor={AUGMENTATION_FACTOR}, With Clip) ===")
print(f"Input directory: {SYNTHETIC_DIR}")
print(f"Output directory: {OUTPUT_ROOT}\n")

summary_list = []

# ===================================================================
# 2. Load Raw Data and Enforce Alignment
# ===================================================================
print("Loading raw data...")
df_full = pd.read_csv(PATH_DATA_CSV)
R_raw_full = torch.load(PATH_R_RAW).float().cpu().numpy()

# Enforce alignment
min_len = min(len(df_full), len(R_raw_full))
if len(df_full) != len(R_raw_full):
    print(f"Warning: Inconsistent data lengths detected. CSV: {len(df_full)}, Embedding: {len(R_raw_full)}")
    print(f"-> Automatically truncating to {min_len} samples for alignment.")

df_full = df_full.iloc[:min_len].reset_index(drop=True)
R_raw_full = R_raw_full[:min_len]
Y_raw_full = df_full["三讀"].astype(float).values
print("Data loading complete.\n")

# ===================================================================
# 3. Main Loop
# ===================================================================
for theme in TARGET_THEMES:
    print(f"Processing: {theme}")
    
    # -------------------------------------------------------
    # A. Train TarNet (Real Data)
    # -------------------------------------------------------
    mask_t1 = df_full["關係文書"].fillna("").str.contains(THEME_TREATMENT)
    mask_t0 = df_full["關係文書"].fillna("").str.contains(theme)
    mask_y = ~np.isnan(Y_raw_full)
    
    train_indices = df_full.index[(mask_t1 | mask_t0) & mask_y].tolist()
    
    if len(train_indices) < 10:
        print(f"Extremely small sample size ({len(train_indices)}), skipping this theme.")
        continue

    R_train_data = R_raw_full[train_indices]
    Y_train_data = Y_raw_full[train_indices]
    T_train_data = np.where(mask_t1[train_indices], 1.0, 0.0)
    
    # Training set splitting
    if len(train_indices) > 50:
        R_train, _, Y_train, _, T_train, _ = train_test_split(
            R_train_data, Y_train_data, T_train_data, test_size=0.2, random_state=SEED
        )
    else:
        print("(Limited data detected; using full dataset for training)")
        R_train, Y_train, T_train = R_train_data, Y_train_data, T_train_data

    # Model Training
    tarnet = TarNet(
        epochs=200, 
        batch_size=32, 
        learning_rate=7e-3, 
        architecture_y=[128, 64, 1], 
        architecture_z=[128],
        dropout=0.3
    )
    tarnet.fit(R=R_train, Y=Y_train, T=T_train)
    
    # -------------------------------------------------------
    # B. Load Synthetic Data
    # -------------------------------------------------------
    # 1. Load synthetic text embeddings
    virtual_filename = f"hidden_states_combined_{theme}.pt"
    virtual_path = os.path.join(SYNTHETIC_DIR, virtual_filename)
    
    # Fault tolerance / Path verification
    if not os.path.exists(virtual_path):
        virtual_path = os.path.join(SYNTHETIC_DIR, f"hidden_states_combined_{theme}_10th_augmented.pt")
    
    if not os.path.exists(virtual_path):
        print(f"Synthetic file {virtual_filename} not found, skipping.")
        continue
        
    R_rewritten = torch.load(virtual_path).float().cpu().numpy()
    
    # 2. Load original text indices and outcomes
    indices_control = df_full.index[mask_t0 & mask_y].tolist()
    R_original_vec = R_raw_full[indices_control]
    Y_original_val = Y_raw_full[indices_control]
    n_original = len(R_original_vec)
    n_rewritten = len(R_rewritten)
    
    print(f"Original: {n_original} | Rewritten: {n_rewritten} (Analysis Ratio 1:{AUGMENTATION_FACTOR})")
    
    # Calculate reference length
    n_base = min(n_original, n_rewritten // AUGMENTATION_FACTOR)
    target_len = n_base * AUGMENTATION_FACTOR
    
    # Truncate synthetic data
    R_rewritten = R_rewritten[:target_len]
    
    # Expand original data (When Factor=1, this aligns length without duplication)
    R_original_expanded = np.repeat(R_original_vec[:n_base], AUGMENTATION_FACTOR, axis=0)
    Y_original_expanded = np.repeat(Y_original_val[:n_base], AUGMENTATION_FACTOR, axis=0)
    
    print(f"Final Analysis: {len(R_rewritten)} synthetic bills (derived from {n_base} original bills)")

    # -------------------------------------------------------
    # C. Prediction and Calculation
    # -------------------------------------------------------
    
    # 1. Predict
    _, y1_rewritten, _ = tarnet.predict(R_rewritten)
    _, y1_latent, _ = tarnet.predict(R_original_expanded)
    
    if isinstance(y1_rewritten, torch.Tensor): y1_rewritten = y1_rewritten.detach().cpu().numpy()
    if isinstance(y1_latent, torch.Tensor): y1_latent = y1_latent.detach().cpu().numpy()
    
    y1_rewritten = y1_rewritten.flatten()
    y1_latent = y1_latent.flatten()
    
    # Fix: Applied numerical clipping
    y1_rewritten = np.clip(y1_rewritten, 0.0, 1.0)
    y1_latent = np.clip(y1_latent, 0.0, 1.0)
    
    # 2. Calculate differences
    diff_rewritten_orig = y1_rewritten - Y_original_expanded
    diff_latent_orig = y1_latent - Y_original_expanded
    diff_final = diff_rewritten_orig - diff_latent_orig
    
    # 3. Generate individual reports
    version_labels = [f"Ver_{i%AUGMENTATION_FACTOR+1}" for i in range(len(y1_rewritten))]
    
    df_calc = pd.DataFrame({
        "Original_Index": np.repeat(range(n_base), AUGMENTATION_FACTOR),
        "Version": version_labels,
        "Original": Y_original_expanded,
        "Rewritten_pred": y1_rewritten,
        "Latent_pred": y1_latent,
        "Rewritten_minus_Original": diff_rewritten_orig,
        "Latent_minus_Original": diff_latent_orig,
        "Diff_of_Diffs": diff_final 
    }).round(2)
    
    save_path = os.path.join(OUTPUT_INDIVIDUAL, f"report_{theme}.csv")
    df_calc.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"   Report generated: {save_path}")
    
    # 4. Calculate Statistical Averages
    summary_list.append({
        "Theme": theme,
        "Avg_Original": np.mean(Y_original_expanded),
        "Avg_Rewritten_pred": np.mean(y1_rewritten),
        "Avg_Latent_pred": np.mean(y1_latent),
        "Avg_Diff_of_Diffs": np.mean(diff_final),
        "Sample_Base_Count": n_base,        
        "Total_Augmented_Count": target_len 
    })
    
    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(y1_latent, bins=30, alpha=0.5, color='blue', label='Latent (Machine Theory)')
    plt.hist(y1_rewritten, bins=30, alpha=0.5, color='red', label=f'Rewritten (AI x{AUGMENTATION_FACTOR})')
    plt.axvline(np.mean(y1_latent), color='blue', linestyle='dashed')
    plt.axvline(np.mean(y1_rewritten), color='red', linestyle='dashed')
    plt.title(f"[{theme}] 6th Term Analysis (x{AUGMENTATION_FACTOR})")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_ROOT, f"plot_{theme}.png"))
    plt.close()

# Output Summary Report
print("\n=== 6th Term Analysis Complete (Clipped) ===")
df_summary = pd.DataFrame(summary_list).round(2)
df_summary.to_csv(os.path.join(OUTPUT_ROOT, "summary_report_6th.csv"), index=False, encoding="utf-8-sig")
print(df_summary)