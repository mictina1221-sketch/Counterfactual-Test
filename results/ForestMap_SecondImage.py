'''
Code for Drawing Minimalist Forest Plot
Please confirm your input/output paths before running
The paths for the 6th and 10th need to be run separately and manually set.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===================================================================
# 1. Settings (Please confirm your paths)
# ===================================================================
# Set to 6th Term or 10th Term based on your folder
TARGET_THEMES = ["原住民", "身心障礙", "國軍", "教師", "農民", "勞工","婦女", "兒少"]

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
INPUT_DIR = r"C:\Users\yachun\Desktop\RR_RR\models\Final_Analysis_6th\Individual_Reports"
OUTPUT_DIR = os.path.dirname(INPUT_DIR)

print(f"=== Generating 6th Term Minimalist Forest Plot (Original - Counterfactual) ===")

plot_data = []

# ===================================================================
# 2. Data Loading and Calculation
# ===================================================================
for theme in TARGET_THEMES:
    file_path = os.path.join(INPUT_DIR, f"report_{theme}.csv")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {theme}")
        continue
        
    df = pd.read_csv(file_path)
    n = len(df)
    
    # Calculate Effect: Original - Rewritten (Counterfactual)
    eff = df["Original"].values - df["Rewritten_pred"].values
    
    mean_val = np.mean(eff)
    std_val = np.std(eff, ddof=1)
    ci_val = 1.96 * (std_val / np.sqrt(n))
    
    plot_data.append({
        "Theme": theme,
        "Theme_Eng": THEME_MAPPING.get(theme, theme), 
        "Mean": mean_val,
        "CI": ci_val
    })

# Convert to DataFrame and Sort
# Logic: Larger difference (Original Advantage) appears at the top
df_plot = pd.DataFrame(plot_data)
df_plot = df_plot.sort_values(by="Mean", ascending=True).reset_index(drop=True)

# ===================================================================
# 3. Plotting 
# ===================================================================
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

# Canvas Settings
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(df_plot))

# A. Baseline at 0 (Black Dotted Line)
ax.axvline(x=0, color='black', linestyle=':', linewidth=1.2, alpha=1.0)

# B. Plot Data Points (Red Squares)
ax.errorbar(
    x=df_plot["Mean"], y=y_pos, 
    xerr=df_plot["CI"], 
    fmt='s',                
    color='#D62728',        
    ecolor='#D62728',       
    capsize=5,              
    markersize=8, 
    linewidth=2,
    label='Effect Size (Original - Counterfactual)'
)

# C. Add Numerical Labels (Above the points)
for i, row in df_plot.iterrows():
    # Use Red for positive values, Gray for non-positive (optional styling)
    text_color = '#D62728' if row["Mean"] > 0 else 'gray'
    label_text = f"+{row['Mean']:.2f}" if row['Mean'] > 0 else f"{row['Mean']:.2f}"
    
    # Adjust Y position slightly up (i - 0.25) to avoid overlap
    ax.text(row["Mean"], i - 0.25, label_text, 
            color=text_color, ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

# D. Axis Settings
ax.set_yticks(y_pos)
ax.set_yticklabels(df_plot["Theme_Eng"], fontsize=12, fontweight='bold')
ax.invert_yaxis() # Largest value at the top

# X-axis Label
ax.set_xlabel("Probability Difference (Original - Rewritten)", fontsize=12)
ax.set_title("6th Term Legislative Yuan: Original Topic Advantage", fontsize=14, fontweight='bold', pad=15)

# E. Grid Lines
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.grid(axis='y', visible=False) 

# F. Remove Spines (Top & Right)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# G. Legend Settings
ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.5), frameon=False)

# H. Save and Show
plt.tight_layout()
plt.subplots_adjust(right=0.75) 

save_path = os.path.join(OUTPUT_DIR, "Forest_Plot_Clean_Effect_6th_Eng_O-C.png")
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✔ 6th Term Minimalist Forest Plot (Original - Counterfactual) saved to: {save_path}")