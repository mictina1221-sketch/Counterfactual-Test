import pandas as pd
from tqdm import tqdm

# --- Settings ---
INPUT_CSV_PATH = r"C:\Users\yachun\Desktop\RR_RR\data\分離資料集\6th_LY\身心障礙.csv"
OUTPUT_CSV_PATH =r"C:\Users\yachun\Desktop\RR_RR\data\prompt\6th_LY\身心障礙_prompt.csv"
SOURCE_COLUMN = '關係文書'
PROMPT_COLUMN_NAME = 'prompt' # This is the field names that create_text.py will read.

# This is the complete instruction we need to give to LLM.
def create_full_prompt(original_text):
    if not isinstance(original_text, str):
        return ""
        
    return f"""你是一個精準的立法公文改寫引擎。

原始文本來自早期掃描檔，包含 OCR 雜訊與大量關於「身心障礙正名」的重複提案。
你的任務是將這些提案的**邏輯骨架**提取出來，並改寫為**關於 [公務人員] 的提案**。

請嚴格執行以下三個步驟：

1.  **【強制清洗】**：
    * 修復錯字（如「爱」→「爰」）。
    * 刪除字與字中間的怪異空格（如「身 心」→「身心」）。
    * 將斷裂的句子接合。

2.  **【邏輯映射與改寫】（最關鍵步驟）**：
    根據原文的類型，選擇以下其中一種改寫策略：

    * **類型 A：如果是「正名/更名」提案**（例如：將「殘廢」改為「身心障礙」）：
        * 重點在於保留「**隨著時代進步，用語應修正以示尊重**」的邏輯。

    * **類型 B：如果是「福利/補助/反歧視」提案**（例如：保障身障就業、給予補助）：
        * 將「身心障礙者」替換為「**公務人員**」。
        * 將「弱勢/殘障」替換為「**執行公務**」或「**辛勞**」。
        * 將「福利/救助」替換為「**權益保障**」或「**津貼**」。

3.  **【格式輸出】**：
    * **嚴格模仿原文長度**：原文幾句話，你就寫幾句話，寫完就停，不需要再有額外說明。
    * **保留公文語氣**：只有當原文包含「爰擬具」、「是否有當」、「敬請公決」需要保留，不要再額外生成。
    * **禁止加戲**：絕對不要自己發明具體的修正條文內容（如：不要寫出「修正第五條...」除非原文就有寫）。

**原始文本：**
{original_text}

**改寫後的關係文書（僅輸出內文，不須輸出改寫策略，也不須輸出提案書等等額外的title）：**
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