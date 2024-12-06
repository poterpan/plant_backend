import pandas as pd
import numpy as np
from scipy import stats
from sqlalchemy import create_engine, text
from datetime import datetime

# 資料庫連接
engine = create_engine('postgresql://WTLab:WTLab502@techsense.panspace.me/plantgrowth')

# 建立新表
create_table_sql = """
CREATE TABLE IF NOT EXISTS co2_absorption_periods (
   id SERIAL PRIMARY KEY,
   date DATE NOT NULL,
   start_time TIME NOT NULL,
   end_time TIME NOT NULL,
   slope FLOAT NOT NULL,
   duration_minutes INTEGER NOT NULL,
   total_absorption FLOAT,
   avg_co2_change FLOAT,
   UNIQUE (date)
);
"""

with engine.connect() as conn:
    conn.execute(text(create_table_sql))
    conn.commit()

# 取得日期範圍
query_dates = """
SELECT DISTINCT DATE(timestamp) as date
FROM sensor_data
WHERE location = 'indoor'
ORDER BY date;
"""

dates = pd.read_sql(query_dates, engine)['date']

for date in dates:
    # 查詢特定日期的數據
    query = f"""
   SELECT timestamp, co2
   FROM sensor_data
   WHERE location = 'indoor'
       AND DATE(timestamp) = '{date}'
       AND EXTRACT(HOUR FROM timestamp) BETWEEN 6 AND 12
   ORDER BY timestamp;
   """

    df = pd.read_sql(query, engine)
    df.set_index('timestamp', inplace=True)

    # 計算移動平均
    df['co2_smooth'] = df['co2'].rolling(window=20).mean()

    # 計算斜率
    window_minutes = 10
    slopes = []
    timestamps = []

    for i in range(len(df) - window_minutes):
        window = df['co2_smooth'].iloc[i:i + window_minutes]
        slope, _, _, _, _ = stats.linregress(range(len(window)), window)
        slopes.append(slope)
        timestamps.append(df.index[i])

    df['slope'] = pd.Series(slopes, index=timestamps)

    # 尋找下降區間
    descent_threshold = -0.3
    min_descent_duration = pd.Timedelta(minutes=5)

    is_descending = df['slope'] < descent_threshold
    descent_periods = []
    current_start = None

    for idx_time, is_desc in is_descending.items():
        if is_desc and current_start is None:
            current_start = idx_time
        elif not is_desc and current_start is not None:
            if idx_time - current_start >= min_descent_duration:
                descent_periods.append((current_start, idx_time))
            current_start = None

    # 合併相近區間
    merged_periods = []
    if descent_periods:
        current_start, current_end = descent_periods[0]

        for start, end in descent_periods[1:]:
            if start - current_end <= pd.Timedelta(minutes=10):
                current_end = end
            else:
                merged_periods.append((current_start, current_end))
                current_start, current_end = start, end

        merged_periods.append((current_start, current_end))

    # 儲存分析結果
    if merged_periods:
        main_period = max(merged_periods, key=lambda x: x[1] - x[0])
        main_start, main_end = main_period

        # 計算主要區間的詳細資訊
        period_data = df['co2_smooth'].loc[main_start:main_end]
        x = np.arange(len(period_data))
        main_slope, intercept, r_value, _, _ = stats.linregress(x, period_data)

        duration_minutes = (main_end - main_start).total_seconds() / 60
        total_change = period_data.iloc[-1] - period_data.iloc[0]
        avg_change = total_change / duration_minutes

        # 存入資料庫
        insert_sql = """
        INSERT INTO co2_absorption_periods 
            (date, start_time, end_time, slope, duration_minutes, total_absorption, avg_co2_change)
        VALUES (:date, :start_time, :end_time, :slope, :duration_minutes, :total_absorption, :avg_co2_change)
        ON CONFLICT (date) 
        DO UPDATE SET 
            start_time = EXCLUDED.start_time,
            end_time = EXCLUDED.end_time,
            slope = EXCLUDED.slope,
            duration_minutes = EXCLUDED.duration_minutes,
            total_absorption = EXCLUDED.total_absorption,
            avg_co2_change = EXCLUDED.avg_co2_change;
        """

        with engine.connect() as conn:
            conn.execute(text(insert_sql), {
                "date": date,
                "start_time": main_start.time(),
                "end_time": main_end.time(),
                "slope": float(main_slope),
                "duration_minutes": float(duration_minutes),
                "total_absorption": float(abs(total_change)),
                "avg_co2_change": float(avg_change)
            })
            conn.commit()

        print(f'Processed and saved data for date: {date}')

engine.dispose()
