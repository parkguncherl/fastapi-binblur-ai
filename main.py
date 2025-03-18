import requests
from bs4 import BeautifulSoup
import psycopg2
from fastapi import FastAPI
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
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)
        logger.info("Page loaded with Selenium")

        # 스크롤을 통해 모든 상품 로드
        unique_urls = set()
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_attempts = 3

        while scroll_attempts < max_attempts:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)  # 콘텐츠 로드 대기

            # 상품 요소가 로드될 때까지 대기
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".diplsayCategoryProductCard_thumbnail__rC7n3 > img"))
                )
                items = driver.find_elements(By.CSS_SELECTOR, ".diplsayCategoryProductCard_thumbnail__rC7n3 > img")
            except StaleElementReferenceException:
                logger.warning("Stale element detected, retrying...")
                time.sleep(3)
                items = driver.find_elements(By.CSS_SELECTOR, ".diplsayCategoryProductCard_thumbnail__rC7n3 > img")

            # 고유 URL 수집
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

        # 모든 상품 추출
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
            "SELECT COUNT(*) FROM TB_AI_DATA WHERE image_url = %s",
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
                    INSERT INTO TB_AI_DATA (image, prod_desc, price, retail_type, cre_tm, image_url)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)