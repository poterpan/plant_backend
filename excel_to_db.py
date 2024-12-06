import os
import pandas as pd
import psycopg2
from datetime import datetime
import re
import io
from psycopg2 import sql


class SensorDataImporter:
    def __init__(self):
        """初始化資料庫連接"""
        self.conn_params = {
            'host': 'techsense.panspace.me',
            'database': 'plantgrowth',
            'user': 'WTLab',
            'password': 'WTLab502'
        }
        self.create_database()

    def get_connection(self):
        """建立資料庫連接"""
        return psycopg2.connect(**self.conn_params)

    def create_database(self):
        """建立資料庫結構"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS sensor_data (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        location VARCHAR(10) NOT NULL,
                        co2 FLOAT,
                        temperature FLOAT,
                        humidity FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                ''')

                # 建立索引
                cur.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON sensor_data(timestamp);')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_location ON sensor_data(location);')

                conn.commit()

    def parse_filename(self, filename):
        """解析檔案名稱以取得日期範圍和位置資訊"""
        pattern = r'(\d{8})-(\d{8})_(indoor|outdoor)\.xlsx'
        match = re.match(pattern, filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")

        start_date = datetime.strptime(match.group(1), '%Y%m%d')
        end_date = datetime.strptime(match.group(2), '%Y%m%d')
        location = match.group(3)

        return start_date, end_date, location

    def import_excel(self, file_path):
        """匯入單個 Excel 檔案到資料庫"""
        filename = os.path.basename(file_path)
        _, _, location = self.parse_filename(filename)

        # 讀取 Excel 檔案，只選擇需要的欄位
        df = pd.read_excel(
            file_path,
            usecols=[
                'Date',
                'Time',
                'Ch1_Co2',
                'Ch2_T(Co2)',
                'Ch3_RH'
            ]
        )
        print(f'Found {len(df)} records in {filename}')

        # 將 Date 和 Time 欄位轉換為字串，然後合併
        df['timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

        # 重命名欄位並轉換數值型別
        df = df.rename(columns={
            'Ch1_Co2': 'co2',
            'Ch2_T(Co2)': 'temperature',
            'Ch3_RH': 'humidity'
        })

        # 確保數值欄位為浮點數
        df['co2'] = pd.to_numeric(df['co2'], errors='coerce')
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')

        # 新增位置欄位
        df['location'] = location

        # 只保留需要的欄位
        df = df[['timestamp', 'location', 'co2', 'temperature', 'humidity']]

        # 移除任何有 NaN 的資料列
        df = df.dropna()

        print(f'Starting writing {len(df)} records to the database')

        # 將 DataFrame 轉換為 CSV 格式的字串緩衝
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)

        # 使用 COPY 命令快速插入資料
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.copy_from(
                    output,
                    'sensor_data',
                    columns=('timestamp', 'location', 'co2', 'temperature', 'humidity'),
                    sep='\t'
                )
                conn.commit()

        return len(df)

    def import_directory(self, directory_path):
        """匯入目錄中的所有 Excel 檔案"""
        imported_files = []
        total_records = 0

        for filename in os.listdir(directory_path):
            if filename.endswith('.xlsx'):
                try:
                    file_path = os.path.join(directory_path, filename)
                    records = self.import_excel(file_path)
                    imported_files.append({
                        'filename': filename,
                        'records': records
                    })
                    total_records += records
                    print(f"Successfully imported {filename}: {records} records")
                except Exception as e:
                    print(f"Error importing {filename}: {str(e)}")

        return {
            'total_records': total_records,
            'imported_files': imported_files
        }

    def get_data_summary(self):
        """獲取資料庫中的資料摘要"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # 獲取總記錄數
                cur.execute("SELECT COUNT(*) FROM sensor_data")
                total_records = cur.fetchone()[0]

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
                    SELECT location, COUNT(*) 
                    FROM sensor_data 
                    GROUP BY location
                """)
                location_counts = dict(cur.fetchall())

                return {
                    'total_records': total_records,
                    'date_range': {
                        'start': date_range[0],
                        'end': date_range[1]
                    },
                    'location_counts': location_counts
                }


# 初始化匯入器
importer = SensorDataImporter()

# # 匯入單個檔案
# records = importer.import_excel('20240607-20240628_indoor.xlsx')
# print(f"Imported {records} records")

# # 或批量匯入整個目錄
# result = importer.import_directory('/Users/poterpan/Downloads/跨領域/')
# print(f"Imported {result['total_records']} records in total")

# 獲取資料摘要
summary = importer.get_data_summary()
print(summary)
