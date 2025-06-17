import os
import random
import base64
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path

# 載入環境變數
here = Path(__file__).resolve().parent
load_dotenv(here / ".env")

# Groq API 憑證
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# 模型名稱
MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# 更新後的提示詞：明確告訴模型，只回傳四種瑕疵類別之一
SHELF_ANALYSIS_PROMPT = """
請分析這張貨架圖片，並從以下 4 種瑕疵類別中選出最適合的答案：  
→ 缺品、缺價卡、品項錯誤、缺串條  

以下是詳細規範，請依照規範做判斷並**只回傳瑕疵類別名稱**：

1. 商品排列：
   - 第1排：廣告展示區  
   - 第2排：6個刮鬍刀架密相連，價格牌在上方  
   - 第3排：3個刀片商品均勻分布，價格牌在下方  
   - 第4排：6個商品均勻分布，價格牌在上方  
   - 第5排：4個刮鬍泡均勻分布，價格牌在下方  
   - 串條區：xtreme3刮鬍刀，價格牌在上方  

2. 包裝顏色順序：
   - 第2排：Hydro5 Premium, Hydro5 Premium, Hydro5 Premium, Hydro5, Hydro5, Hydro5  
   - 第3排：Hydro5 刀片 ×3  
   - 第4排：Hydro5 Custom ×6  
   - 第5排：白色, 綠色, 綠色, 藍色  
   - 串條區：xtreme3  

3. 檢查項目：
   1. 商品擺放：是否有缺品？  
   2. 價格牌顯示：是否有缺價卡？  
   3. 商品正確性：品項型號、包裝顏色是否正確？  
   4. 串條區：是否缺少串條？  
"""

# 測試圖片資料夾 & 標籤資料夾
TEST_FOLDER = "/Users/y/Documents/Programming/大四上/價格辨識/中文辨識/收銀端假第2座＿測試"
LABEL_ROOT = "/Users/y/Documents/Programming/大四上/價格辨識/中文辨識/收銀端架第2座"
DEFECT_TYPES = ["缺品", "缺價卡", "品項錯誤", "缺串條"]

# 隨機抽樣張數
N = 20

# 讀取所有圖片檔名並隨機抽樣
all_images = [
    f for f in os.listdir(TEST_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
sampled = random.sample(all_images, min(N, len(all_images)))

# 初始化準確率計數
total = len(sampled)
correct = 0

for filename in sampled:
    image_path = os.path.join(TEST_FOLDER, filename)

    # 讀取標準答案
    ground_truth = "Unknown"
    for defect in DEFECT_TYPES:
        if os.path.exists(os.path.join(LABEL_ROOT, defect, filename)):
            ground_truth = defect
            break

    # 編碼圖片
    with open(image_path, "rb") as imgf:
        b64 = base64.b64encode(imgf.read()).decode()

    # 呼叫 Groq API
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SHELF_ANALYSIS_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                }
            ]
        }
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    predicted = resp.choices[0].message.content.strip()

    # 檢查是否正確
    is_correct = (predicted == ground_truth)
    if is_correct:
        correct += 1

    # 輸出當前結果
    print(f"檔名: {filename}")
    print(f"  ➜ 標準答案: {ground_truth}")
    print(f"  ➜ 系統預測: {predicted}")
    print(f"  ➜ 正確: {is_correct}\n")

# 最後輸出整體準確率
accuracy = correct / total if total > 0 else 0
print(f"總共測試：{total} 張")
print(f"正確數量：{correct} 張")
print(f"準確率：{accuracy:.2%}")
