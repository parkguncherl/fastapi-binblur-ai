import requests
from bs4 import BeautifulSoup
import psycopg2
from fastapi import FastAPI, HTTPException
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from datetime import date
import logging
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

DB_PARAMS = {
    "host": "3.34.21.35",
    "database": "binblurdb",
    "user": "binblur2024",
    "password": "binblur20241!",
    "port": 5432
}

TARGET_URL = "https://shopping.naver.com/window/fashion-group/category/20006056?sort=BRAND_POPULARITY"

# CNN 모델 정의
class PricePredictor(nn.Module):
    def __init__(self):
        super(PricePredictor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

def scrape_naver_shopping(url):
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)
        logger.info("Page loaded with Selenium")

        unique_urls = set()
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_attempts = 3

        while scroll_attempts < max_attempts:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".diplsayCategoryProductCard_thumbnail__rC7n3 > img"))
                )
                items = driver.find_elements(By.CSS_SELECTOR, ".diplsayCategoryProductCard_thumbnail__rC7n3 > img")
            except StaleElementReferenceException:
                logger.warning("Stale element detected, retrying...")
                time.sleep(3)
                items = driver.find_elements(By.CSS_SELECTOR, ".diplsayCategoryProductCard_thumbnail__rC7n3 > img")

            for item in items:
                try:
                    img_url = item.get_attribute("data-src") or item.get_attribute("src")
                    if img_url and not img_url.startswith("data:image"):
                        unique_urls.add(img_url)
                except StaleElementReferenceException:
                    logger.warning("Stale element in URL fetch, skipping this item")
                    continue

            new_height = driver.execute_script("return document.body.scrollHeight")
            logger.info(f"Scroll attempt {scroll_attempts + 1}: Height {new_height}, Unique items {len(unique_urls)}")

            if new_height == last_height:
                scroll_attempts += 1
                if scroll_attempts == max_attempts:
                    logger.info("Max scroll attempts reached, assuming end of content")
                    break
            else:
                scroll_attempts = 0
            last_height = new_height

        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()

        items = soup.select(".diplsayCategoryProductCard_thumbnail__rC7n3 > img")
        logger.info(f"Total items found in HTML: {len(items)}")

        products = []
        seen_urls = set()
        for item in items:
            img_url = item.get("data-src") or item.get("src")
            if img_url in seen_urls:
                continue
            seen_urls.add(img_url)

            desc_tag = item.find_next(class_="product_title")
            desc = desc_tag.text.strip() if desc_tag else item.get("alt", "No description")

            price_tag = item.find_next(class_="productPrice_number__lYegc")
            if price_tag:
                price_text = price_tag.text.strip().replace(",", "")
                try:
                    price = int(price_text)
                except ValueError:
                    price = None
            else:
                price = None

            logger.info(f"Scraped: {img_url}, {desc}, {price}")
            if img_url and not img_url.startswith("data:image"):
                products.append({
                    "image_url": img_url,
                    "description": desc,
                    "price": price
                })

        logger.info(f"Total unique products prepared: {len(products)}")
        return products
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return []

def download_image_sync(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Failed to download image {url}: {e}")
        return None

def check_existing_image_url(conn, image_url):
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT COUNT(*) FROM tb_ai_data WHERE image_url = %s",
            (image_url,)
        )
        count = cur.fetchone()[0]
        return count > 0
    except Exception as e:
        logger.error(f"Error checking existing image_url: {e}")
        return False
    finally:
        cur.close()

def insert_into_db(products):
    conn = get_db_connection()
    cur = conn.cursor()
    today = date.today()
    inserted_count = 0

    try:
        for product in products:
            if check_existing_image_url(conn, product["image_url"]):
                logger.info(f"Skipping duplicate image_url: {product['image_url']}")
                continue

            image_data = download_image_sync(product["image_url"])
            if image_data:
                cur.execute(
                    """
                    INSERT INTO public.tb_ai_data (image, prod_desc, price, retail_type, cre_tm, image_url)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (psycopg2.Binary(image_data), product["description"], product["price"], "O", today, product["image_url"])
                )
                inserted_count += 1
            else:
                logger.warning(f"Image download failed for {product['image_url']}, skipping insertion")
        conn.commit()
        logger.info(f"Inserted {inserted_count} new records into tb_ai_data")
        return inserted_count
    except Exception as e:
        conn.rollback()
        logger.error(f"Database insertion failed: {e}")
        return 0
    finally:
        cur.close()
        conn.close()

@app.get("/scrape-and-store")
async def scrape_and_store():
    logger.info("Starting scrape and store process")
    products = scrape_naver_shopping(TARGET_URL)
    if products:
        inserted_count = insert_into_db(products)
        return {"message": f"Successfully processed {len(products)} products, inserted {inserted_count} new records"}
    else:
        return {"message": "No products found or scraping failed"}

@app.post("/api/train")
async def train_model() -> Dict[str, str]:
    try:
        # 모델 초기화
        model = PricePredictor()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # retail_type 'O' 데이터 가져오기
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT image, price 
            FROM public.tb_ai_data 
            WHERE retail_type = 'O' AND image IS NOT NULL AND price IS NOT NULL
        """)
        training_data = cur.fetchall()
        cur.close()
        conn.close()

        if not training_data:
            raise HTTPException(status_code=400, detail="훈련 데이터가 없습니다")

        # 데이터 준비
        images = []
        prices = []
        for img_data, price in training_data:
            try:
                img = Image.open(io.BytesIO(img_data)).convert('RGB')  # RGB로 변환
                img_tensor = transform(img)
                images.append(img_tensor)
                prices.append(float(price))
            except Exception as e:
                logger.warning(f"Failed to process image: {e}")
                continue

        if not images:
            raise HTTPException(status_code=400, detail="유효한 이미지 데이터가 없습니다")

        images = torch.stack(images)
        prices = torch.tensor(prices).float().unsqueeze(1)

        # 모델 훈련
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, prices)
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        # 모델 저장
        torch.save(model.state_dict(), 'price_predictor.pth')
        return {"message": "모델 훈련이 성공적으로 완료되었습니다"}

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict_prices() -> Dict[str, str]:
    try:
        # 모델 로드
        model = PricePredictor()
        model.load_state_dict(torch.load('price_predictor.pth'))
        model.eval()

        # retail_type 'P' 데이터 가져오기
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, image 
            FROM public.tb_ai_data 
            WHERE retail_type = 'P' AND image IS NOT NULL
        """)
        prediction_data = cur.fetchall()

        if not prediction_data:
            cur.close()
            conn.close()
            raise HTTPException(status_code=400, detail="예측 데이터가 없습니다")

        # 예측 수행
        updated_count = 0
        with torch.no_grad():
            for id, img_data in prediction_data:
                try:
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')  # RGB로 변환
                    img_tensor = transform(img).unsqueeze(0)
                    prediction = model(img_tensor)
                    predicted_price = int(prediction.item())

                    cur.execute("""
                        UPDATE public.tb_ai_data 
                        SET predict_price = %s 
                        WHERE id = %s
                    """, (predicted_price, id))
                    updated_count += 1
                except Exception as e:
                    logger.warning(f"Failed to predict for id {id}: {e}")
                    continue

        conn.commit()
        cur.close()
        conn.close()

        return {"message": f"예측이 성공적으로 완료되었습니다. {updated_count}개의 레코드가 업데이트됨"}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)