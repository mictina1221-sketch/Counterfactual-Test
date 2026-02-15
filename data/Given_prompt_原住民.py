import pandas as pd
from tqdm import tqdm

# --- Settings ---
INPUT_CSV_PATH = r"C:\Users\yachun\Desktop\RR_RR\data\分離資料集\6th_LY\原住民.csv"
OUTPUT_CSV_PATH =r"C:\Users\yachun\Desktop\RR_RR\data\prompt\6th_LY\原住民_prompt.csv"
SOURCE_COLUMN = '關係文書'
PROMPT_COLUMN_NAME = 'prompt' # This is the field names that create_text.py will read.

# This is the complete instruction we need to give to LLM.
def create_full_prompt(original_text):
    if not isinstance(original_text, str):
        return ""
        
    return f"""你是一個嚴格的立法公文轉換器。
你的任務是將【原住民族】相關的提案，改寫為【公務人員】的提案。

請嚴格執行以下指令，特別是「防迴圈機制」：

1.  **核心詞彙邏輯映射（必須遵守）**：
    * **原住民 / 族人** $\rightarrow$ 替換為 **[公務人員]**
    * **部落 / 原鄉 / 山地** $\rightarrow$ 替換為 **[機關]** 或 **[公務體系]**
    * **自治 (Autonomy)** $\rightarrow$ 替換為 **[行政中立]** 或 **[結社權]** (避免寫成「公務員自治」這種不合理的詞)。
    * **傳統領域 / 保留地** $\rightarrow$ 替換為 **[辦公場域]** 或 **[職務範圍]**。
    * **狩獵 / 採集 / 文化權** $\rightarrow$ 替換為 **[進修]** 或 **[依法行政權]**。

2.  **【防迴圈機制】(致命規則)**：
    * **一句話原則**：如果原文只有一句話（例如：「擬具XX法草案」），改寫後**只能有一句話**。
    * **禁止重複**：絕對禁止連續出現兩次「爰擬具」。
    * **結尾鎖死**：一旦輸出了「是否有當？敬請公決。」，請**立刻終止生成**。

3.  **格式清洗**：
    * 修復 OCR 錯字與多餘空白。
    * 保留提案人名單。

**範例教學：**
* 原文：「...為推動原住民族自治，擬具原住民族自治法草案...」
* 正確改寫：「...為推動公務人員行政中立，擬具公務人員行政中立法草案...」
* (錯誤改寫：「...擬具公務人員自治法草案，爰擬具公務人員自治法草案...」 <--- 禁止重複！)

**原始文本：**
{original_text}

**改寫後的關係文書（僅需要輸出改寫後的內文，不需要輸出額外解釋及原始文本，寫完「敬請公決」請立刻停筆）：**
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