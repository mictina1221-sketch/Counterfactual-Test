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
        
    return f"""你是一個精準的文字轉換引擎，專門處理立法提案的「案由」段落。

你的任務是對輸入的「原始文本」進行【清洗】與【概念置換】，並輸出一段與原文**長度、結構完全對應**的文字。

請嚴格遵守以下規則：

1.  **清洗 OCR 雜訊**：
    * 修復錯字（如「爱」→「爰」）。
    * 刪除多餘空白（如「當 前」→「當前」）。
    * 將斷裂的句子接合，確保語意通順。

2.  **核心概念置換（最重要的任務）**：
    * 將原文中的 **[勞工/教師/特定對象]** 等主體，替換為 **[公務人員]**。
    * 將原文中的 **[雇主/學校/特定對象]** 等等，替換為 **[政府機關]**。
    * 將原文中的 **[勞動權益/受教權/特定訴求]** 等等，替換為 **[公務權益]**。
    * **注意：** 如果原文是為了「保障某某權益」，改寫後也必須是「保障公務人員某某權益」。

3.  **嚴格的結構模仿（Do's and Don'ts）**：
    * **[必須]**：保留原文的「提案人名單」、「連接詞」（如：鑑於、為、爰擬具、特提案）、「結尾語」（是否有當？敬請公決）。
    * **[禁止]**：絕對**不能**增加原文沒有的細節（如：具體的法律條文內容、條號細節）。如果原文沒寫第幾條，你就別寫；如果原文只有一句話，你就只能輸出一句話。
    * **[禁止]**：不要輸出「修正條文：...」這種清單格式。

**範例對照：**
* 原文：「...為確保護理教師之退休權益，特提案修正...」
* 改寫：「...為確保公務人員之退休權益，特提案修正...」
* (錯誤示範：「...為確保公務人員退休權益，並得續聘...」 <- 禁止增加原文沒有的意思)

**原始文本：**
{original_text}

**改寫後的關係文書（僅輸出內文）：**
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