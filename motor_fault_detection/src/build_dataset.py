# src/build_dataset.py

import pandas as pd
import numpy as np
from preprocessing import SignalPreprocessor
from feature_extraction import FeatureExtractor
from glob import glob

def build_feature_dataset(data_dir='data/raw/', window_size=1000):
    """Build complete feature dataset from all CSV files"""
    
    preprocessor = SignalPreprocessor()
    extractor = FeatureExtractor()
    
    all_features = []
    
    # Process all CSV files
    for csv_file in glob(f'{data_dir}/*.csv'):
        print(f"Processing {csv_file}...")
        
        df = pd.read_csv(csv_file)
        label = df['label'].iloc[0]
        
        # Preprocess
        df_processed = preprocessor.preprocess(df)
        
        # Segment into windows
        ax_segments = preprocessor.segment_data(df_processed['ax_norm'].values, window_size)
        ay_segments = preprocessor.segment_data(df_processed['ay_norm'].values, window_size)
        az_segments = preprocessor.segment_data(df_processed['az_norm'].values, window_size)
        
        # Extract features from each window
        for i in range(len(ax_segments)):
            features = extractor.extract_features(
                ax_segments[i],
                ay_segments[i],
                az_segments[i]
            )
            features['label'] = label
            all_features.append(features)
    
    # Create DataFrame
    feature_df = pd.DataFrame(all_features)
    feature_df.to_csv('data/processed/feature_dataset.csv', index=False)
    
    print(f"\nDataset created: {len(feature_df)} samples, {len(feature_df.columns)-1} features")
    print(f"Class distribution:\n{feature_df['label'].value_counts()}")
    
    return feature_df

if __name__ == "__main__":
    dataset = build_feature_dataset()