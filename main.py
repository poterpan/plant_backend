from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
from pydantic import BaseModel
import os
import logging

# 設定日誌級別
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SensorData(BaseModel):
    timestamp: datetime
    location: str
    co2: float
    temperature: float
    humidity: float


class DataSummary(BaseModel):
    total_records: int
    date_range: dict
    location_counts: dict


app = FastAPI(title="Plant Growth Monitoring API")

# 設定 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應該限制來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 資料庫連接設定
DB_PARAMS = {
    'host': 'techsense.panspace.me',
    'database': 'plantgrowth',
    'user': 'WTLab',
    'password': 'WTLab502'
}

# 在掛載靜態檔案前先檢查目錄是否存在
IMAGES_DIR = "images"
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)
    logger.warning(f"Created images directory: {IMAGES_DIR}")
logger.info(f"Images directory path: {os.path.abspath(IMAGES_DIR)}")

# 掛載靜態檔案目錄
try:
    app.mount("/static/images", StaticFiles(directory=IMAGES_DIR), name="images")
    logger.info("Successfully mounted images directory")
except Exception as e:
    logger.error(f"Failed to mount images directory: {e}")


def get_db_connection():
    """建立資料庫連接"""
    return psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)


@app.get("/")
async def root():
    """API 根路徑"""
    return {"message": "Plant Growth Monitoring API"}


@app.get("/summary", response_model=DataSummary)
async def get_summary():
    """獲取資料摘要"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 獲取總記錄數
            cur.execute("SELECT COUNT(*) as total FROM sensor_data")
            total_records = cur.fetchone()['total']

            # 獲取日期範圍
            cur.execute("""
                SELECT 
                    MIN(timestamp) as earliest_date,
                    MAX(timestamp) as latest_date
                FROM sensor_data
            """)
            date_range = cur.fetchone()

            # 獲取每個位置的記錄數
            cur.execute("""
                SELECT location, COUNT(*) as count
                FROM sensor_data 
                GROUP BY location
            """)
            location_counts = {row['location']: row['count'] for row in cur.fetchall()}

            return {
                'total_records': total_records,
                'date_range': {
                    'start': date_range['earliest_date'],
                    'end': date_range['latest_date']
                },
                'location_counts': location_counts
            }


@app.get("/data", response_model=List[SensorData])
async def get_data(
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        location: Optional[str] = None,
        limit: int = Query(default=2880, le=43200)
):
    """獲取感測器資料"""
    query = "SELECT * FROM sensor_data WHERE 1=1"
    params = []

    if start_time:
        query += " AND timestamp >= %s"
        params.append(start_time)

    if end_time:
        query += " AND timestamp <= %s"
        params.append(end_time)

    if location:
        query += " AND location = %s"
        params.append(location)

    query += " ORDER BY timestamp DESC LIMIT %s"
    params.append(limit)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


@app.get("/latest")
async def get_latest():
    """獲取最新的感測器資料"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT ON (location)
                    location, timestamp, co2, temperature, humidity
                FROM sensor_data
                ORDER BY location, timestamp DESC
            """)
            return cur.fetchall()


@app.get("/analysis/daily")
async def get_daily_analysis(
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        location: Optional[str] = None
):
    """獲取每日分析數據"""
    query = """
        SELECT 
            DATE(timestamp) as date,
            location,
            AVG(co2) as avg_co2,
            MAX(co2) as max_co2,
            MIN(co2) as min_co2,
            AVG(temperature) as avg_temperature,
            AVG(humidity) as avg_humidity
        FROM sensor_data
        WHERE 1=1
    """
    params = []

    if start_date:
        query += " AND timestamp >= %s"
        params.append(start_date)

    if end_date:
        query += " AND timestamp <= %s"
        params.append(end_date)

    if location:
        query += " AND location = %s"
        params.append(location)

    query += " GROUP BY DATE(timestamp), location ORDER BY date DESC"

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


@app.get("/analysis/co2-absorption")
async def calculate_co2_absorption(
        start_time: datetime,
        end_time: datetime
):
    """計算CO2吸收量"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 獲取室內外CO2差異
            cur.execute("""
                WITH indoor_data AS (
                    SELECT timestamp, co2 as indoor_co2
                    FROM sensor_data
                    WHERE location = 'indoor'
                    AND timestamp BETWEEN %s AND %s
                ),
                outdoor_data AS (
                    SELECT timestamp, co2 as outdoor_co2
                    FROM sensor_data
                    WHERE location = 'outdoor'
                    AND timestamp BETWEEN %s AND %s
                )
                SELECT 
                    indoor_data.timestamp,
                    indoor_data.indoor_co2,
                    outdoor_data.outdoor_co2,
                    (outdoor_data.outdoor_co2 - indoor_data.indoor_co2) as co2_absorption
                FROM indoor_data
                JOIN outdoor_data 
                ON DATE_TRUNC('minute', indoor_data.timestamp) = 
                   DATE_TRUNC('minute', outdoor_data.timestamp)
                ORDER BY indoor_data.timestamp
            """, [start_time, end_time, start_time, end_time])

            return cur.fetchall()


@app.get("/analysis/co2-absorption-period/{date}")
async def get_co2_absorption_period(date: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT *
                FROM co2_absorption_periods
                WHERE date = %s
            """, [date])
            result = cur.fetchone()
            return result


@app.get("/data/smoothed")
async def get_smoothed_data(
        start_time: datetime,
        end_time: datetime,
        window_length: int = 91,  # 必須是奇數
        polyorder: int = 3
):
    """獲取平滑處理後的感測器資料"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 分別獲取室內外數據
            cur.execute("""
                SELECT timestamp, location, co2
                FROM sensor_data
                WHERE timestamp BETWEEN %s AND %s
                ORDER BY timestamp ASC
            """, [start_time, end_time])

            data = cur.fetchall()

            # 分離室內外數據
            indoor_data = [d for d in data if d['location'] == 'indoor']
            outdoor_data = [d for d in data if d['location'] == 'outdoor']

            # 應用 Savitzky-Golay 濾波
            def smooth_data(data_list):
                if len(data_list) < window_length:
                    return data_list

                co2_values = [d['co2'] for d in data_list]
                smoothed = savgol_filter(co2_values, window_length, polyorder)

                return [{
                    'timestamp': d['timestamp'],
                    'location': d['location'],
                    'co2': float(s)
                } for d, s in zip(data_list, smoothed)]

            smoothed_indoor = smooth_data(indoor_data)
            smoothed_outdoor = smooth_data(outdoor_data)

            return smoothed_indoor + smoothed_outdoor


@app.get("/images/test")
async def test_images():
    """測試圖片目錄訪問"""
    try:
        # logger.info(f"Testing image directory access")
        # logger.info(f"Current working directory: {os.getcwd()}")
        # logger.info(f"Image directory absolute path: {os.path.abspath(IMAGES_DIR)}")

        files = os.listdir(IMAGES_DIR)
        logger.info(f"Files in directory: {files}")

        return {
            "status": "success",
            "files": files,
            "directory": os.path.abspath(IMAGES_DIR)
        }
    except Exception as e:
        logger.error(f"Error testing image directory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/available-dates", response_model=List[str])
async def get_available_image_dates():
    """取得有照片的日期清單"""
    try:
        # logger.info("Accessing available-dates endpoint")
        # logger.info(f"Current working directory: {os.getcwd()}")
        # logger.info(f"Image directory path: {os.path.abspath(IMAGES_DIR)}")

        # 確認目錄是否存在
        if not os.path.exists(IMAGES_DIR):
            logger.error(f"Directory does not exist: {IMAGES_DIR}")
            raise HTTPException(status_code=500, detail="Images directory not found")

        # 列出目錄內容
        files = os.listdir(IMAGES_DIR)
        # logger.info(f"Files in directory: {files}")

        # 過濾 jpg 檔案
        images = [f for f in files if f.endswith('.jpg')]
        # logger.info(f"JPG files found: {images}")

        # 解析日期
        dates = []
        for image in images:
            try:
                date_str = image.replace('.jpg', '')
                datetime.strptime(date_str, '%Y%m%d')
                dates.append(date_str)
                # logger.info(f"Successfully parsed date: {date_str}")
            except ValueError as e:
                logger.warning(f"Invalid date format in filename: {image} - {str(e)}")
                continue

        dates.sort(reverse=True)
        logger.info(f"Final dates list: {dates}")
        return dates

    except Exception as e:
        logger.error(f"Error in get_available_dates: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/by-date/{date}")
async def get_image_by_date(date: str):
    """檢查指定日期是否有照片"""
    try:
        # 驗證日期格式
        datetime.strptime(date, '%Y%m%d')

        # 構建檔案名稱
        filename = f"{date}.jpg"
        file_path = os.path.join(IMAGES_DIR, filename)

        # 檢查檔案是否存在
        if os.path.exists(file_path):
            # 返回圖片的 URL
            return {
                "exists": True,
                "url": f"/static/images/{filename}"
            }
        else:
            return {
                "exists": False,
                "url": None
            }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYYMMDD")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 可選：提供取得最新照片的端點
@app.get("/images/latest")
async def get_latest_image():
    """取得最新的照片"""
    try:
        # 取得所有 jpg 檔案
        images = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]

        if not images:
            return {
                "exists": False,
                "url": None,
                "date": None
            }

        # 解析檔名為日期並找出最新的
        valid_images = []
        for image in images:
            try:
                date_str = image.replace('.jpg', '')
                date = datetime.strptime(date_str, '%Y%m%d')
                valid_images.append((date, image))
            except ValueError:
                continue

        if not valid_images:
            return {
                "exists": False,
                "url": None,
                "date": None
            }

        # 排序並取得最新的照片
        latest_date, latest_image = max(valid_images, key=lambda x: x[0])

        return {
            "exists": True,
            "url": f"/static/images/{latest_image}",
            "date": latest_date.strftime('%Y%m%d')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/routes")
async def debug_routes():
    """列出所有已註冊的路由"""
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": route.methods if hasattr(route, "methods") else None
        })
    return routes


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
