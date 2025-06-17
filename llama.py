import os
import base64
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path

# 載入環境變數
here = Path(__file__).resolve().parent
load_dotenv(here / ".env")

# Groq API 憑證
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("🔍 [DEBUG] GROQ_API_KEY:", repr(GROQ_API_KEY))



# 初始化 Groq API 客戶端
client = Groq(api_key=GROQ_API_KEY)

# 請確認你的 Groq API 支援的模型名稱
MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# 📝 SHELF_ANALYSIS_PROMPT 保持不變
SHELF_ANALYSIS_PROMPT = """
分析這張圖片中的主要貨架，判斷是牙膏或漱口水的貨架。

重要提示：
- 請只關注畫面中央的主要貨架
- 忽略周邊的紙箱、其他貨架等干擾元素
- 即使貨架較小，也要仔細觀察商品特徵

判斷原則（按優先順序）：
1. 展示牌內容判斷（如果有）：
 - 漱口水貨架：展示牌通常會有“12小時長效清新”等相關字樣
 - 牙膏貨架：展示牌通常會有“7天顯著焠白”等相關字樣

2. 商品特徵判斷（當展示牌不存在或不清晰時）：
 - 漱口水：
 * 商品為直立的瓶裝包裝
 * 能看到透明瓶身中的液體
 * 瓶身較高且圓柱形
 - 牙膏：
 * 商品為扁平軟管包裝
 * 通常橫放或斜靠擺放
 * 包裝較扁平且長方形

3. 貨架擺放方式：
 - 漱口水：通常直立擺放，每層間距較高
 - 牙膏：可能多層緊密排列，間距較小

請根據以上原則，回答商品類別（只能回答“牙膏”或“漱口水”）。

注意：
1. 只需要回答“牙膏”或“漱口水”
2. 不要添加任何額外的文字、符號或換行
3. 按照判斷原則的優先順序進行分析
4. 完全忽略主要貨架以外的區域
"""

# 圖片資料夾路徑
IMAGE_FOLDER = "/Users/y/Documents/Programming/大四上/價格辨識/中文辨識/huggingface/traindata_images拷貝"

for filename in os.listdir(IMAGE_FOLDER):
 if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
    continue

 image_path = os.path.join(IMAGE_FOLDER, filename)
 print(f"\n🔍 分析圖片：{filename}")

 # 讀取並編碼為 Base64
 with open(image_path, "rb") as img_file:
    b64_image = base64.b64encode(img_file.read()).decode("utf-8")

 # 建立多模態訊息：先傳文字 prompt，再附上 data URI 格式的圖片
 messages = [
 {
 "role": "user",
 "content": [
 {"type": "text", "text": SHELF_ANALYSIS_PROMPT},
 {
 "type": "image_url",
 "image_url": {"url": f"data:image/png;base64,{b64_image}"}
 }
 ]
 }
 ]

 # 呼叫 Groq API
 response = client.chat.completions.create(
 model=MODEL,
 messages=messages
 )

 # 印出結果
 print("🧾 結果：", response.choices[0].message.content)