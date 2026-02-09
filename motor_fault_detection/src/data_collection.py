# src/data_collection.py

import serial
import pandas as pd
import numpy as np
from datetime import datetime

class ESP32DataCollector:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        
    def collect_data(self, duration=10, label='healthy'):
        """Collect vibration data for specified duration"""
        data = []
        start_time = datetime.now()
        
        print(f"Collecting {label} data for {duration} seconds...")
        
        while (datetime.now() - start_time).seconds < duration:
            if self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8').strip()
                try:
                    # Parse: "ax,ay,az,timestamp"
                    values = line.split(',')
                    ax, ay, az = float(values[0]), float(values[1]), float(values[2])
                    data.append({
                        'ax': ax,
                        'ay': ay,
                        'az': az,
                        'timestamp': datetime.now(),
                        'label': label
                    })
                except:
                    continue
        
        df = pd.DataFrame(data)
        filename = f"data/raw/{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} samples to {filename}")
        return df
    
    def close(self):
        self.ser.close()

# Usage
if __name__ == "__main__":
    collector = ESP32DataCollector(port='COM3')  # Change port for Windows
    
    # Collect different fault types
    collector.collect_data(duration=30, label='healthy')
    collector.collect_data(duration=30, label='bearing_fault')
    collector.collect_data(duration=30, label='rotor_imbalance')
    collector.collect_data(duration=30, label='shaft_misalignment')
    
    collector.close()