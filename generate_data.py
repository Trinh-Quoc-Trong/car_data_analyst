import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Configuration
NUM_MACHINES = 15
NUM_DAYS = 30
BASE_LAT, BASE_LON = 21.028511, 105.804817  # Base location for construction site (Hanoi)
DATA_DIR = "data"

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# --- 1. Generate Activity Log ---
print("Generating activity_log.csv...")
machine_ids = [f"M-{str(i).zfill(2)}" for i in range(1, NUM_MACHINES + 1)]
dates = [datetime.now().date() - timedelta(days=i) for i in range(NUM_DAYS)]

activity_data = []
for machine_id in machine_ids:
    # Each machine has a different base fuel consumption rate
    base_fuel_rate = np.random.uniform(5, 10)
    for date in dates:
        # Not every machine works every day
        if np.random.rand() > 0.3:  # 70% chance of working
            operating_hours = np.random.uniform(4, 12)
            fuel_consumed = (base_fuel_rate * operating_hours) * np.random.uniform(0.95, 1.15) # Add some noise

            # Introduce some data quality issues
            if np.random.rand() < 0.05: # 5% chance of missing data
                operating_hours = np.nan
            elif np.random.rand() < 0.05: # 5% chance of missing data
                fuel_consumed = np.nan
            elif np.random.rand() < 0.03: # 3% chance of invalid data
                operating_hours = -1.0
            
            activity_data.append([machine_id, date, operating_hours, fuel_consumed])

activity_df = pd.DataFrame(activity_data, columns=['machine_id', 'date', 'operating_hours', 'fuel_consumed'])
activity_df.to_csv(os.path.join(DATA_DIR, 'activity_log.csv'), index=False)
print(f"Generated {len(activity_df)} rows for activity_log.csv")


# --- 2. Generate GPS Log ---
print("Generating gps_log.csv...")
gps_data = []
# Filter for valid activities to generate GPS logs
valid_activities = activity_df.dropna(subset=['operating_hours']).copy()
valid_activities = valid_activities[valid_activities['operating_hours'] > 0]

for _, row in valid_activities.iterrows():
    machine_id = row['machine_id']
    date = row['date']
    operating_hours = row['operating_hours']
    
    num_points = int(operating_hours * 20) # 20 GPS pings per hour
    
    # Each machine operates in a slightly different area of the site
    machine_offset_lat = np.random.uniform(-0.01, 0.01)
    machine_offset_lon = np.random.uniform(-0.01, 0.01)
    current_lat = BASE_LAT + machine_offset_lat
    current_lon = BASE_LON + machine_offset_lon
    
    start_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=8) # Work starts at 8 AM
    
    for i in range(num_points):
        timestamp = start_time + timedelta(minutes=(i * 3)) # 1 point every 3 minutes
        
        # Simulate movement
        current_lat += np.random.uniform(-0.0005, 0.0005)
        current_lon += np.random.uniform(-0.0005, 0.0005)
        
        gps_data.append([machine_id, timestamp, current_lat, current_lon])

gps_df = pd.DataFrame(gps_data, columns=['machine_id', 'timestamp', 'latitude', 'longitude'])
gps_df.to_csv(os.path.join(DATA_DIR, 'gps_log.csv'), index=False)
print(f"Generated {len(gps_df)} rows for gps_log.csv")
print("Data generation complete.") 