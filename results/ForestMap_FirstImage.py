'''
Code for Drawing Minimalist Forest Plot
Please confirm your input/output paths before running
The paths for the 6th and 10th need to be run separately and manually set.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# ===================================================================
# 1. Settings: 6th Term
# ===================================================================
# Keep these in Chinese to match your actual filenames (e.g., report_原住民.csv)
TARGET_THEMES = ["原住民", "身心障礙", "國軍", "教師", "農民", "勞工", "兒少", "婦女"]

# English Mapping for Plot Labels
THEME_MAPPING = {
    "原住民": "Indigenous",
    "身心障礙": "Disabled",
    "國軍": "Soldiers",
    "教師": "Teachers",
    "農民": "Farmers",
    "勞工": "Laborers",
    "兒少": "Children & Youth",
    "婦女": "Women"  
}

# Path Settings
# Please confirm this path matches your folder structure
INPUT_DIR = r"C:\Users\yachun\Desktop\RR_RR\models\Final_Analysis_6th\Individual_Reports"
OUTPUT_DIR = os.path.dirname(INPUT_DIR)

print(f"=== [6th Term] Generating Vertical Split Forest Plot (English Version) ===")

plot_data = []

# ===================================================================
# 2. Calculate Statistics
# ===================================================================
for theme in TARGET_THEMES:
    file_path = os.path.join(INPUT_DIR, f"report_{theme}.csv")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {theme}")
        continue
        
    df = pd.read_csv(file_path)
    n = len(df)

    if theme == TARGET_THEMES[0]:
        print(f"Checking columns for {THEME_MAPPING[theme]}: {df.columns.tolist()}")

    # 2. Define Column Names
    # If your CSV has different names, modify them here
    col_orig = 'Original'
    col_lat  = 'Latent_pred'
    col_rew  = 'Rewritten_pred'

    # Adjust column names if they differ
    if col_orig not in df.columns:
        col_orig = 'Original_Probability' if 'Original_Probability' in df.columns else col_orig
        col_lat = 'Latent_Probability' if 'Latent_Probability' in df.columns else col_lat
        col_rew = 'Rewritten_Probability' if 'Rewritten_Probability' in df.columns else col_rew

    try:
        eff_lat = df[col_orig] - df[col_lat]
        eff_rew = df[col_orig] - df[col_rew]
        
    except KeyError as e:
        print(f"❌ Error: Probability columns not found in {theme}.")
        print(f"Expected: {col_orig}, {col_lat}, {col_rew}")
        print(f"Actual: {df.columns.tolist()}")
        continue 

    # Calculate Statistics for Latent (Theoretical)
    mean_lat = np.mean(eff_lat)
    ci_lat = 1.96 * (np.std(eff_lat, ddof=1) / np.sqrt(n))
    
    # Calculate Statistics for Rewritten (Counterfactual)
    mean_rew = np.mean(eff_rew)
    ci_rew = 1.96 * (np.std(eff_rew, ddof=1) / np.sqrt(n))
    
    plot_data.append({
        "Theme": theme, 
        "Theme_Eng": THEME_MAPPING.get(theme, theme),
        "Lat_Mean": mean_lat, "Lat_CI": ci_lat,
        "Rew_Mean": mean_rew, "Rew_CI": ci_rew
    })

print("Data calculation complete. Preparing plot...")
df_plot = pd.DataFrame(plot_data)

# Sort by Latent Mean (Theoretical value)
if not df_plot.empty:
    df_plot = df_plot.sort_values(by="Lat_Mean", ascending=True).reset_index(drop=True)
else:
    print("❌ Failed to load any data. Please check your paths and file names.")

# ===================================================================
# 3. Plotting
# ===================================================================
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(df_plot))

# Set offset distance for the split
offset = 0.15 

# A. Plot Blue Dots (Theoretical / Latent) - Shifted Up
ax.errorbar(
    x=df_plot["Lat_Mean"], y=y_pos - offset, 
    xerr=df_plot["Lat_CI"],
    fmt='o', color='#1F77B4', ecolor='#1F77B4', 
    capsize=4, markersize=8, label='Theoretical Lift (Latent)'
)

# B. Plot Red Squares (Actual / Counterfactual) - Shifted Down
ax.errorbar(
    x=df_plot["Rew_Mean"], y=y_pos + offset, 
    xerr=df_plot["Rew_CI"],
    fmt='s', color='#D62728', ecolor='#D62728', 
    capsize=4, markersize=8, label='Actual Lift (Counterfactual)'
)

# C. Add Text Annotations
for i, row in df_plot.iterrows():
    # Blue text above blue dot
    ax.text(row["Lat_Mean"], i - offset - 0.15, f"{row['Lat_Mean']:.2f}", 
            color='#1F77B4', ha='center', fontsize=9, fontweight='bold')
    # Red text below red square
    ax.text(row["Rew_Mean"], i + offset + 0.25, f"{row['Rew_Mean']:.2f}", 
            color='#D62728', ha='center', fontsize=9, fontweight='bold')

# D. Add horizontal dotted lines for separation
for y in y_pos:
    ax.axhline(y=y, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

# E. Axis Settings
ax.set_yticks(y_pos)
ax.set_yticklabels(df_plot["Theme_Eng"], fontsize=12, fontweight='bold')

# Vertical line at 0
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.invert_yaxis() # Largest value at the top

ax.set_xlabel("Probability Advantage (Compared to Original Bill)", fontsize=11, fontweight='bold')
ax.set_title("6th Term Legislative Yuan: Theoretical vs. Counterfactual Effects", fontsize=14, fontweight='bold', pad=15)

# F. Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1F77B4', label='Theoretical (TarNet)', markersize=10),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#D62728', label='Counterfactual (LLM)', markersize=10)
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10, frameon=False)

# Aesthetics
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "Forest_Plot_Split_6th_English.png")
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✔ Plot Saved Successfully: {save_path}\n")