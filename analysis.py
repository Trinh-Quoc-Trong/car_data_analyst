import pandas as pd
import numpy as np
from haversine import haversine, Unit
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import os

# --- Configuration ---
DATA_DIR = "data"
OUTPUT_DIR = "output"
ACTIVITY_FILE = os.path.join(DATA_DIR, 'activity_log.csv')
GPS_FILE = os.path.join(DATA_DIR, 'gps_log.csv')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_distance(group):
    """Calculates the distance traveled for a group of GPS points."""
    group = group.sort_values(by='timestamp')
    group['prev_lat'] = group['latitude'].shift(1)
    group['prev_lon'] = group['longitude'].shift(1)
    
    # Calculate distance using haversine formula for each point from the previous one
    distances = group.apply(
        lambda row: haversine(
            (row['prev_lat'], row['prev_lon']),
            (row['latitude'], row['longitude']),
            unit=Unit.KILOMETERS
        ) if pd.notnull(row['prev_lat']) else 0,
        axis=1
    )
    return distances.sum()

def process_data():
    """
    Giai đoạn 1: Thu Thập, Tổng Hợp và Làm Sạch Dữ Liệu.
    """
    print("--- Giai đoạn 1: Bắt đầu xử lý dữ liệu ---")

    # 1. Đọc và kiểm tra dữ liệu ban đầu
    activity_df = pd.read_csv(ACTIVITY_FILE)
    gps_df = pd.read_csv(GPS_FILE)

    print("Kiểm tra dữ liệu hoạt động (activity_log):")
    activity_df.info()
    print("\nKiểm tra dữ liệu GPS (gps_log):")
    gps_df.info()

    # 2. Làm sạch và Chuẩn hóa Dữ liệu Hoạt động
    print("\nLàm sạch dữ liệu hoạt động...")
    # Xử lý giá trị thiếu
    initial_rows = len(activity_df)
    activity_df.dropna(subset=['operating_hours', 'fuel_consumed'], inplace=True)
    print(f"Loại bỏ {initial_rows - len(activity_df)} dòng có giá trị thiếu.")
    
    # Loại bỏ dữ liệu phi lý
    initial_rows = len(activity_df)
    activity_df = activity_df[activity_df['operating_hours'] > 0]
    print(f"Loại bỏ {initial_rows - len(activity_df)} dòng có operating_hours <= 0.")

    # Chuẩn hóa kiểu dữ liệu
    activity_df['date'] = pd.to_datetime(activity_df['date'])
    activity_df['operating_hours'] = activity_df['operating_hours'].astype(float)
    activity_df['fuel_consumed'] = activity_df['fuel_consumed'].astype(float)
    
    # 3. Xử lý và Tính toán từ Dữ liệu GPS
    print("\nXử lý dữ liệu GPS và tính quãng đường...")
    gps_df['timestamp'] = pd.to_datetime(gps_df['timestamp'])
    gps_df['date'] = gps_df['timestamp'].dt.date
    gps_df['date'] = pd.to_datetime(gps_df['date'])

    # Sắp xếp theo máy và thời gian
    gps_df.sort_values(by=['machine_id', 'timestamp'], inplace=True)

    # Tính toán quãng đường di chuyển cho mỗi máy mỗi ngày
    # Đây là một tác vụ nặng, có thể mất một chút thời gian
    daily_distance = gps_df.groupby(['machine_id', 'date']).apply(calculate_distance).reset_index(name='distance_traveled')
    print("Tổng hợp quãng đường di chuyển mỗi ngày hoàn tất.")

    # 4. Hợp nhất thành Bảng dữ liệu cuối cùng
    print("\nHợp nhất dữ liệu...")
    final_df = pd.merge(
        activity_df,
        daily_distance,
        on=['machine_id', 'date'],
        how='left'
    )
    # Những ngày máy hoạt động nhưng không có dữ liệu GPS -> quãng đường = 0
    final_df['distance_traveled'].fillna(0, inplace=True)

    print("--- Giai đoạn 1: Hoàn thành ---")
    return final_df, gps_df


def analyze_and_visualize(final_df, gps_df):
    """
    Giai đoạn 2: Trực Quan Hóa Dữ Liệu và Báo Cáo.
    """
    print("\n--- Giai đoạn 2: Bắt đầu phân tích và trực quan hóa ---")

    # Tính toán các chỉ số hiệu suất
    final_df['fuel_per_hour'] = final_df['fuel_consumed'] / final_df['operating_hours']
    final_df['km_per_liter'] = final_df['distance_traveled'] / final_df['fuel_consumed']
    # Tránh chia cho 0
    final_df.replace([np.inf, -np.inf], 0, inplace=True)
    final_df['km_per_liter'].fillna(0, inplace=True)

    # 1. Trực quan hóa so sánh hiệu suất
    print("\n1. Tạo biểu đồ so sánh hiệu suất...")
    avg_performance = final_df.groupby('machine_id')['fuel_per_hour'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(15, 8))
    palette = ['red' if x in avg_performance.head(3).index else 'skyblue' for x in avg_performance.index]
    sns.barplot(x=avg_performance.index, y=avg_performance.values, palette=palette)
    
    plt.title('So sánh Hiệu Suất Trung Bình (Lít/Giờ)', fontsize=16)
    plt.xlabel('Mã Máy', fontsize=12)
    plt.ylabel('Nhiên liệu tiêu thụ trung bình (Lít/Giờ)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'performance_comparison.png')
    plt.savefig(output_path)
    print(f"Biểu đồ đã được lưu tại: {output_path}")
    plt.close()

    # 2. Biểu đồ phân tán tìm ngày bất thường
    print("\n2. Tạo biểu đồ phân tán để tìm ngày hoạt động bất thường...")
    plt.figure(figsize=(12, 8))
    sns.lmplot(data=final_df, x='operating_hours', y='fuel_consumed', aspect=1.5,
               scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    
    plt.title('Mối quan hệ giữa Giờ Hoạt Động và Nhiên liệu Tiêu Thụ', fontsize=16)
    plt.xlabel('Số giờ hoạt động', fontsize=12)
    plt.ylabel('Lượng nhiên liệu tiêu thụ (Lít)', fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'abnormal_days_scatterplot.png')
    plt.savefig(output_path)
    print(f"Biểu đồ đã được lưu tại: {output_path}")
    plt.close()

    # 3. Trực quan hóa Lộ trình (Case Study)
    print("\n3. Tạo bản đồ lộ trình (Case Study)...")
    # Chọn máy có hiệu suất kém nhất và một ngày hoạt động để phân tích
    target_machine = avg_performance.index[0]
    target_date = final_df[final_df['machine_id'] == target_machine]['date'].max().date()
    
    print(f"Case Study: Phân tích lộ trình máy '{target_machine}' trong ngày '{target_date}'")
    
    route_data = gps_df[
        (gps_df['machine_id'] == target_machine) &
        (gps_df['timestamp'].dt.date == target_date)
    ].sort_values('timestamp')

    if not route_data.empty:
        # Tạo bản đồ
        map_center = [route_data['latitude'].mean(), route_data['longitude'].mean()]
        route_map = folium.Map(location=map_center, zoom_start=15)

        # Vẽ lộ trình
        locations = route_data[['latitude', 'longitude']].values.tolist()
        folium.PolyLine(locations, color='blue', weight=2.5, opacity=1).add_to(route_map)

        # Đánh dấu điểm bắt đầu và kết thúc
        folium.Marker(
            location=locations[0],
            popup=f"Bắt đầu: {route_data['timestamp'].iloc[0]}",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(route_map)
        folium.Marker(
            location=locations[-1],
            popup=f"Kết thúc: {route_data['timestamp'].iloc[-1]}",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(route_map)

        output_path = os.path.join(OUTPUT_DIR, 'route_visualization.html')
        route_map.save(output_path)
        print(f"Bản đồ tương tác đã được lưu tại: {output_path}")
    else:
        print(f"Không tìm thấy dữ liệu GPS cho máy '{target_machine}' vào ngày '{target_date}'.")

    print("--- Giai đoạn 2: Hoàn thành ---")
    return avg_performance


def generate_report(avg_performance):
    """
    Tạo báo cáo tổng hợp ra console.
    """
    print("\n--- BÁO CÁO TỔNG HỢP HIỆU SUẤT HOẠT ĐỘNG ---")
    print("Phân tích dựa trên dữ liệu thu thập được.")
    print("-" * 50)

    print("\nPhần 1: Tổng quan hiệu suất đội xe")
    print("Biểu đồ 'performance_comparison.png' cho thấy sự khác biệt về mức tiêu thụ nhiên liệu trung bình (lít/giờ) giữa các máy.")
    
    top3 = avg_performance.head(3)
    print("\n=> Top 3 máy móc cần chú ý nhất (hiệu suất kém nhất):")
    for machine, value in top3.items():
        print(f"   - {machine}: {value:.2f} lít/giờ")
    
    print("\nPhần 2: Phân tích các ngày hoạt động bất thường")
    print("Biểu đồ 'abnormal_days_scatterplot.png' giúp xác định các ngày làm việc không hiệu quả.")
    print("Các điểm nằm xa và ở phía trên đường hồi quy màu đỏ thể hiện những ngày máy móc tiêu thụ nhiên liệu nhiều hơn đáng kể so với mức trung bình cho cùng một số giờ hoạt động. Đây là những trường hợp cần được điều tra thêm.")

    print("\nPhần 3: Case Study - Phân tích lộ trình chi tiết")
    print("Một bản đồ tương tác ('route_visualization.html') đã được tạo để phân tích sâu hơn về lộ trình của máy có hiệu suất kém nhất.")
    print("Hãy mở tệp HTML để xem chi tiết. Các điểm GPS cụm lại một chỗ có thể là dấu hiệu của việc chạy không tải hoặc quy trình làm việc chưa tối ưu, dẫn đến lãng phí nhiên liệu.")

    print("\n--- KẾT LUẬN & ĐỀ XUẤT ---")
    print("1. Tập trung kiểm tra, bảo dưỡng các máy trong Top 3 hiệu suất kém nhất.")
    print("2. Sử dụng các biểu đồ và bản đồ đã tạo để trao đổi với đội ngũ vận hành, tìm ra nguyên nhân cho các ngày hoạt động bất thường và các lộ trình không hiệu quả.")
    print("3. Tiếp tục thu thập và phân tích dữ liệu để theo dõi sự cải thiện theo thời gian.")
    print("-" * 50)


if __name__ == '__main__':
    final_df, gps_df = process_data()
    print("\nDataFrame cuối cùng sau khi xử lý:")
    print(final_df.head())
    
    avg_performance = analyze_and_visualize(final_df, gps_df)
    
    generate_report(avg_performance) 