import requests
from bs4 import BeautifulSoup
import psycopg2
from fastapi import FastAPI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from datetime import date
import logging

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
        options.add_argument("--headless")  # 브라우저 창 없이 실행
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)
        logger.info("Page loaded with Selenium")

        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()

        items = soup.select(".diplsayCategoryProductCard_thumbnail__rC7n3 > img")[:100]
        logger.info(f"Found {len(items)} items")

        products = []
        for item in items:
            img_url = item.get("data-src") or item.get("src")  # data-src 우선 사용
            desc_tag = item.find_next(class_="product_title")  # 가정된 클래스
            desc = desc_tag.text.strip() if desc_tag else item.get("alt", "No description")
            logger.info(f"Scraped: {img_url}, {desc}")
            if img_url and not img_url.startswith("data:image"):  # placeholder 제외
                products.append({"image_url": img_url, "description": desc})
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

def insert_into_db(products):
    conn = get_db_connection()
    cur = conn.cursor()
    today = date.today()

    try:
        for product in products:
            image_data = download_image_sync(product["image_url"])
            if image_data:
                cur.execute(
                    """
                    INSERT INTO public.tb_ai_data (image, prod_desc, retail_type, cre_tm)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (psycopg2.Binary(image_data), product["description"], "O", today)
                )
        conn.commit()
        logger.info(f"Inserted {len(products)} records into tb_ai_data")
    except Exception as e:
        conn.rollback()
        logger.error(f"Database insertion failed: {e}")
    finally:
        cur.close()
        conn.close()

@app.get("/scrape-and-store")
async def scrape_and_store():
    logger.info("Starting scrape and store process")
    products = scrape_naver_shopping(TARGET_URL)
    if products:
        insert_into_db(products)
        return {"message": f"Successfully stored {len(products)} products"}
    else:
        return {"message": "No products found or scraping failed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)