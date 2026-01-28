import pandas as pd
from tqdm import tqdm

# --- Settings ---
INPUT_CSV_PATH = r"C:\Users\yachun\Desktop\RR_RR\data\分離資料集\6th_LY\勞工.csv"
OUTPUT_CSV_PATH =r"C:\Users\yachun\Desktop\RR_RR\data\prompt\6th_LY\勞工_prompt.csv"
SOURCE_COLUMN = '關係文書'
PROMPT_COLUMN_NAME = 'prompt' # This is the field names that create_text.py will read.

# This is the complete instruction we need to give to LLM.
def create_full_prompt(original_text):
    if not isinstance(original_text, str):
        return ""
        
    return f"""你是一位頂尖的立法委員，擅長在嚴謹的法律公文格式下，基於真實資料改寫法案。

我會提供一段來自真實法案的「關係文書」文字。你的任務是將這段文字改寫成一個**不同主題**的法案提案。

在改寫時，請嚴格遵守以下規則：

1.  **保留骨架，替換血肉**：必須完整保留原文的**語法結構、句子長度、專業術語（例如：「爰擬具」、「是否有當」、「敬請公決」）、哪個黨團提案的、以及整體公文語氣**不可替換。這是最重要的規則。
2.  **辨識並替換核心概念**：找出原文中的[勞工]這個核心概念與關鍵詞，然後將核心概念與關鍵詞替換成**公務人員這個核心概念與關鍵詞**。
3.  **保持正式格式**：產出的文字須維持「正式提案公文」格式，不得出現任何額外說明、分析、或評論。
4.  **確保邏輯自洽**：改寫後的文字內部邏輯必須通順，讀起來要像一篇真的「正式提案」。

請直接輸出改寫後的「關係文書」全文，不要包含任何前言、標題或額外解釋。

**原文如下：**
{original_text}
"""

# --- Main ---
# Read Dateset CSV
print(f"Reading the original file: {INPUT_CSV_PATH}...")
df = pd.read_csv(INPUT_CSV_PATH)

new_df = pd.DataFrame()
print("Creating Prompt...")
new_df[PROMPT_COLUMN_NAME] = [create_full_prompt(text) for text in tqdm(df[SOURCE_COLUMN])]


print(f"Saving a new Prompt file to: {OUTPUT_CSV_PATH}...")
new_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
print(f"The new Prompt profile '{OUTPUT_CSV_PATH}' is ready.")
print("Now you can use create_text.py to process this new file.")