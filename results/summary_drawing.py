'''
Code for Drawing Summary Plots
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# ===================================================================
# 1. Configuration
# ===================================================================

# Paths to the report directories (Please verify these paths)
DIR_6TH = r"C:\Users\yachun\Desktop\RR_RR\models\Final_Analysis_6th\Individual_Reports"
DIR_10TH = r"C:\Users\yachun\Desktop\RR_RR\models\Final_Analysis_10th\Individual_Reports"

# Output directory (Parent folder of the 10th term reports)
OUTPUT_DIR = os.path.dirname(DIR_10TH)

# Theme Mapping (Chinese -> English)
THEME_MAPPING = {
    "農民": "Farmers",
    "原住民": "Indigenous",
    "勞工": "Laborers",
    "教師": "Teachers",
    "國軍": "Soldiers",
    "兒少": "Children & Youth",
    "身心障礙": "Disabled",
    "婦女": "Women"
}

# Ordered list of themes for plotting
ORDERED_THEMES = ["農民", "原住民", "勞工", "教師", "國軍", "兒少", "身心障礙", "婦女"]

# ===================================================================
# 2. Data Calculation Function 
# ===================================================================
def get_stats(folder_path, theme_chi):
    """
    Reads the report CSV and calculates Mean & 95% CI for (Original - Rewritten).
    Returns None, None if file is missing or columns are invalid.
    """
    file_path = os.path.join(folder_path, f"report_{theme_chi}.csv")
    
    if not os.path.exists(file_path):
        return None, None 
    
    try:
        df = pd.read_csv(file_path)
        n = len(df)
        
        # Logic: Calculate Original - Rewritten
        if 'Original' in df.columns and 'Rewritten_pred' in df.columns:
            eff = df['Original'] - df['Rewritten_pred']
        elif 'Original_minus_Rewritten' in df.columns:
            eff = df['Original_minus_Rewritten']
        elif 'Rewritten_minus_Original' in df.columns:
            eff = df['Rewritten_minus_Original'] * -1
        else:
            print(f"Warning: Probability columns missing in {theme_chi} ({folder_path}). Skipping.")
            return None, None

        # Statistics
        mean_val = np.mean(eff)
        std_val = np.std(eff, ddof=1)
        ci_val = 1.96 * (std_val / np.sqrt(n))
        
        return mean_val, ci_val
            
    except Exception as e:
        print(f"Error processing {theme_chi}: {e}")
        return None, None

# ===================================================================
# 3. Data Collection
# ===================================================================
print("Reading data and calculating Original - Counterfactual statistics...")

plot_data = []

for theme in ORDERED_THEMES:
    eng_name = THEME_MAPPING.get(theme, theme)
    
    # Get stats for 6th and 10th Terms
    m6, ci6 = get_stats(DIR_6TH, theme)
    m10, ci10 = get_stats(DIR_10TH, theme)
    
    plot_data.append({
        "Category": eng_name,
        "6th_Mean": m6, "6th_CI": ci6,
        "10th_Mean": m10, "10th_CI": ci10
    })

df_plot = pd.DataFrame(plot_data)
# Reverse order for plotting (Top item appears first)
df_plot = df_plot.iloc[::-1].reset_index(drop=True)

# ===================================================================
# 4. Plotting
# ===================================================================
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 7))
y_pos = np.arange(len(df_plot))
offset = 0.15  # Vertical offset to separate the two terms

# Baseline at 0
ax.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.8)

# --- Plot 6th Term (Blue Circles) ---
valid_6th = df_plot[df_plot["6th_Mean"].notna()]
if not valid_6th.empty:
    y_6th = y_pos[valid_6th.index] - offset
    ax.errorbar(
        x=valid_6th["6th_Mean"], y=y_6th,
        xerr=valid_6th["6th_CI"],
        fmt='o', color='#1F77B4', ecolor='#1F77B4',
        capsize=5, markersize=9, linewidth=2,
        label='6th Term'
    )
    # Add labels
    for i, row in valid_6th.iterrows():
        ax.text(row['6th_Mean'], i - offset - 0.25, f"{row['6th_Mean']:.2f}", 
                color='#1F77B4', ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- Plot 10th Term (Red Squares) ---
valid_10th = df_plot[df_plot["10th_Mean"].notna()]
if not valid_10th.empty:
    y_10th = y_pos[valid_10th.index] + offset
    ax.errorbar(
        x=valid_10th["10th_Mean"], y=y_10th,
        xerr=valid_10th["10th_CI"],
        fmt='s', color='#D62728', ecolor='#D62728',
        capsize=5, markersize=9, linewidth=2,
        label='10th Term'
    )
    # Add labels
    for i, row in valid_10th.iterrows():
        ax.text(row['10th_Mean'], i + offset + 0.35, f"{row['10th_Mean']:.2f}", 
                color='#D62728', ha='center', va='top', fontsize=9, fontweight='bold')

# ===================================================================
# 5. Styling and Saving
# ===================================================================
# Horizontal grid lines
for y in y_pos:
    ax.axhline(y=y, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

ax.set_yticks(y_pos)
ax.set_yticklabels(df_plot["Category"], fontsize=12, fontweight='bold')

# Axis Labels and Title
ax.set_xlabel("Probability Advantage (Original - Counterfactual)", fontsize=11, fontweight='bold')
ax.set_title("Causal Effect Comparison: Original Identity Advantage (6th vs. 10th Term)", fontsize=14, fontweight='bold', pad=15)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1F77B4', label='6th Term', markersize=10),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#D62728', label='10th Term', markersize=10)
]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title="Session")

# Clean up layout
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(right=0.8) 

save_path = os.path.join(OUTPUT_DIR, "Forest_Plot_Comparison_Original_minus_Rewritten.png")
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✔ Comparison Forest Plot saved to: {save_path}")