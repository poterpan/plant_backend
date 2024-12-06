import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode Ms']
from scipy import stats
import numpy as np
import os
from datetime import datetime
from sqlalchemy import create_engine

# 建立資料庫連接
engine = create_engine('postgresql://WTLab:WTLab502@techsense.panspace.me/plantgrowth')

# 建立儲存圖片的資料夾
if not os.path.exists('co2_analysis'):
    os.makedirs('co2_analysis')

# 取得日期範圍
query_dates = """
SELECT DISTINCT DATE(timestamp) as date
FROM sensor_data
WHERE location = 'indoor'
ORDER BY date;
"""
dates = pd.read_sql(query_dates, engine)['date']

# 儲存結果
results = []

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

    # 儲存圖片
    if merged_periods:
        main_period = max(merged_periods, key=lambda x: x[1] - x[0])
        main_start, main_end = main_period

        y = df['co2_smooth'].loc[main_start:main_end]
        x = np.arange(len(y))
        main_slope, intercept, r_value, _, _ = stats.linregress(x, y)

        # 繪製圖表
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['co2'], 'gray', alpha=0.3, label='原始數據')
        plt.plot(df.index, df['co2_smooth'], 'b', label='平滑數據')

        for start, end in merged_periods:
            plt.axvspan(start, end, color='yellow', alpha=0.3)

        plt.plot(df.loc[main_start:main_end].index,
                 main_slope * np.arange(len(y)) + intercept,
                 'r', linewidth=2,
                 label=f'斜率: {main_slope:.2f} ppm/min')

        plt.title(f'CO2分析 ({date})')
        plt.xlabel('時間')
        plt.ylabel('CO2 (ppm)')
        plt.legend()
        plt.grid(True)

        # 儲存圖片
        plt.savefig(f'co2_analysis/co2_analysis_{date}.png')
        plt.close()

        # 儲存結果
        results.append({
            'date': date,
            'start_time': main_start.strftime('%H:%M'),
            'end_time': main_end.strftime('%H:%M'),
            'slope': main_slope,
            'duration_minutes': (main_end - main_start).total_seconds() / 60
        })
        print(f'Processed date: {date}, saved to co2_analysis_{date}.png')

# 輸出結果到CSV
results_df = pd.DataFrame(results)
results_df.to_csv('co2_analysis_results.csv', index=False)
engine.dispose()