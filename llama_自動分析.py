import os
import base64
import json
import re
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

# （可保留）Prompt Engineering 模板
PROMPT_ENGINEERING_TEMPLATE = """
請根據這張貨架圖片生成一份「提示工程」（Prompt Engineering）：
1. 明確分析目標（要偵測的產品/情境）。
2. 規範輸出格式（JSON 格式範例）。
3. 必要時補充產品參考資訊或包裝特徵。
4. 僅回傳最終要用於分析的 Prompt 文字，不要其他敘述。
5. 請嚴格遵守產品偵測模板的形式
"""


# 新增：產品偵測模板，讓模型輸出所有偵測到的產品
PRODUCT_DETECTION_TEMPLATE = """
請根據這張圖片，找出所有可見的產品，並以 JSON 陣列格式回傳。每個元素包含：
 - 廠名：佰事達物流股份有限公司  
   - 品名：豐力富即溶濃縮乳清蛋白(清爽原味)  
   - 規格：450g * 2 入  
   - 條碼：4710958448569  
   - 包裝特色：  
     1. 上半部有黃色握把造型  
     2. 左半部駝金色、右半部藍綠色漸層  
     3. 最上方有 “Fernleaf” 字樣  
     4. 中間大字 “Protein+ 即溶濃縮乳清蛋白”  

**注意**：只回傳 JSON，不要多餘文字。
"""

def call_model(messages):
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    content = resp.choices[0].message.content.strip()
    # 去除 code block 標記
    content = re.sub(r"^```(?:\w+)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content).strip()
    return content

def generate_prompt_for_image(b64, mime, template=PROMPT_ENGINEERING_TEMPLATE):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": template},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        ]
    }]
    return call_model(messages)

def detect_products_in_image(b64, mime):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": PRODUCT_DETECTION_TEMPLATE},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        ]
    }]
    text = call_model(messages)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}", "raw": text}

# 資料夾路徑，請改成你的實際路徑
TEST_FOLDER = "/Users/y/Documents/Programming/大四上/價格辨識/中文辨識/新品照片"

# 取得所有圖片檔案
all_images = [
    f for f in os.listdir(TEST_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

results = {}
for filename in all_images:
    image_path = os.path.join(TEST_FOLDER, filename)
    with open(image_path, "rb") as imgf:
        b64 = base64.b64encode(imgf.read()).decode()
    ext = os.path.splitext(filename)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    # 直接用產品偵測模板
    result = detect_products_in_image(b64, mime)
    results[filename] = result

# 將結果寫入 JSON 檔，方便後續程式讀取
output_path = os.path.join(TEST_FOLDER, "detection_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"所有檔案偵測結果已寫入：{output_path}")
