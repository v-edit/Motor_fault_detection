# src/preprocessing.py

import numpy as np
import pandas as pd
from scipy import signal

class SignalPreprocessor:
    def __init__(self, sampling_rate=1000):
        self.fs = sampling_rate
    
    def filter_signal(self, data, lowcut=5, highcut=200):
        """Bandpass filter to remove noise"""
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data)
        return filtered
    
    def normalize(self, data):
        """Normalize to zero mean, unit variance"""
        return (data - np.mean(data)) / np.std(data)
    
    def segment_data(self, data, window_size=1000, overlap=0.5):
        """Split into overlapping windows"""
        step = int(window_size * (1 - overlap))
        segments = []
        
        for i in range(0, len(data) - window_size, step):
            segment = data[i:i + window_size]
            segments.append(segment)
        
        return np.array(segments)
    
    def preprocess(self, df):
        """Full preprocessing pipeline"""
        # Filter each axis
        df['ax_filtered'] = self.filter_signal(df['ax'].values)
        df['ay_filtered'] = self.filter_signal(df['ay'].values)
        df['az_filtered'] = self.filter_signal(df['az'].values)
        
        # Normalize
        df['ax_norm'] = self.normalize(df['ax_filtered'])
        df['ay_norm'] = self.normalize(df['ay_filtered'])
        df['az_norm'] = self.normalize(df['az_filtered'])
        
        return df

# Usage
if __name__ == "__main__":
    preprocessor = SignalPreprocessor(sampling_rate=1000)
    
    df = pd.read_csv('data/raw/healthy_20260209_120000.csv')
    df_processed = preprocessor.preprocess(df)
    df_processed.to_csv('data/processed/healthy_processed.csv', index=False)