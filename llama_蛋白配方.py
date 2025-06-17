import os
import random
import base64
import json
import re 
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from PIL import Image, ImageDraw

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
請分析這張貨架圖片，並偵測是否含有「安怡優蛋白肌肉+」此產品。  
嚴格遵守以下規範，僅回傳最終 JSON 結果，**不做其他敘述**。

1. 分析規則  
   - 只偵測「安怡優蛋白肌肉+」，忽略其他品牌與包裝。  
   - 若未偵測到，則 `"found": false` 且 `"count": 0`，`"locations": []`。  
   - 若偵測到，一定要回傳每個包裝的 bounding box 座標（左上角 x1,y1，右下角 x2,y2）。

2. 產品參考資訊  
     "廠名": "佰事達物流股份有限公司",
      "品名": "安怡優蛋白肌肉+",
      "規格": "",
      "條碼": "",
      "包裝特色": [
        "1. 金色罐蓋",
        "2. 白色罐身",
        "3. 標籤上有 'Anlene 安怡' 字樣",
        "4. 標籤上有 '優蛋白肌肉+' 字樣",
        "5. 有金色漸層設計"

3. 輸出格式（**必須**遵守）  
```json
{
  "product": "安怡優蛋白肌肉+",
  "found": <true|false>,
  "locations": [
    {"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>},
    …
  ]
}

"""

# 測試圖片資料夾 & 標籤資料夾
TEST_FOLDER = "/Users/y/Documents/Programming/大四上/價格辨識/中文辨識/新品/安怡優蛋白配方"

# 隨機抽樣張數
N = 50

all_images = [
    f for f in os.listdir(TEST_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
sampled = random.sample(all_images, min(N, len(all_images)))

# 統計變數
total = len(sampled)
found_count = 0

for filename in sampled:
    image_path = os.path.join(TEST_FOLDER, filename)

    # 4.1 讀檔並轉 base64
    with open(image_path, "rb") as imgf:
        b64 = base64.b64encode(imgf.read()).decode()
    ext = os.path.splitext(filename)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    # 4.2 呼叫 API
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SHELF_ANALYSIS_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            ]
        }
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    predicted = resp.choices[0].message.content.strip()

    cleaned = re.sub(r"^```(?:json)?\s*", "", predicted)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # 4.3 解析 JSON
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"[ERROR] {filename} 無法解析 JSON:\n{cleaned}\n")
        continue

    found = result.get("found", False)
    locations = result.get("locations", [])
    count = len(locations)

    # 印出單張結果
    print(f"檔名: {filename}")
    print(f"  ➜ 偵測到 (found): {found}")
    print(f"  ➜ 數量 (count): {count}")
    print(f"  ➜ 位置 (locations): {locations}\n")

    # 累計偵測到張數
    if found:
        found_count += 1

      # 4.5 在原圖上畫出 bounding box 並存檔
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    for loc in locations:
        # 檢查完整座標
        if not all(k in loc for k in ("x1","y1","x2","y2")):
            print(f"[WARN] {filename} 跳過不完整座標：{loc}")
            continue
        x1, y1, x2, y2 = loc["x1"], loc["y1"], loc["x2"], loc["y2"]
        # 排序保證左上(x0,y0)、右下(x1_,y1_)
        x0, x1_ = sorted([x1, x2])
        y0, y1_ = sorted([y1, y2])
        draw.rectangle([x0, y0, x1_, y1_], outline="red", width=3)

    out_dir = os.path.join(TEST_FOLDER, "annotated")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"annotated_{filename}")
    img.save(save_path)
    print(f"  ➜ 已存檔：{save_path}\n")

# 迴圈外再印整體統計
not_found_count = total - found_count
print("==== 偵測統計 ====")
print(f"  總共處理: {total} 張")
print(f"  成功偵測到: {found_count} 張")
print(f"  未偵測到: {not_found_count} 張")
print(f"  偵測率: {found_count/total*100:.1f}%")