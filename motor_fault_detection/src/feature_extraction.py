# src/feature_extraction.py

import numpy as np
import pandas as pd
from scipy import stats, fft

class FeatureExtractor:
    
    @staticmethod
    def time_domain_features(signal):
        """Extract time-domain features"""
        features = {}
        
        # Statistical features
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['peak'] = np.max(np.abs(signal))
        features['peak_to_peak'] = np.ptp(signal)
        features['crest_factor'] = features['peak'] / features['rms']
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
        features['variance'] = np.var(signal)
        
        return features
    
    @staticmethod
    def frequency_domain_features(signal, fs=1000):
        """Extract frequency-domain features using FFT"""
        features = {}
        
        # FFT
        N = len(signal)
        fft_vals = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(N, 1/fs)
        
        # Take only positive frequencies
        pos_mask = fft_freq > 0
        fft_freq = fft_freq[pos_mask]
        fft_mag = np.abs(fft_vals[pos_mask])
        
        # Spectral features
        features['spectral_energy'] = np.sum(fft_mag**2)
        features['spectral_entropy'] = stats.entropy(fft_mag + 1e-12)
        features['peak_frequency'] = fft_freq[np.argmax(fft_mag)]
        features['mean_frequency'] = np.sum(fft_freq * fft_mag) / np.sum(fft_mag)
        
        # Frequency band energies
        freq_bands = [(0, 50), (50, 100), (100, 200)]
        for i, (low, high) in enumerate(freq_bands):
            mask = (fft_freq >= low) & (fft_freq < high)
            features[f'band_{low}_{high}_energy'] = np.sum(fft_mag[mask]**2)
        
        return features
    
    def extract_features(self, signal_x, signal_y, signal_z, fs=1000):
        """Extract all features from 3-axis data"""
        all_features = {}
        
        for axis, signal in zip(['x', 'y', 'z'], [signal_x, signal_y, signal_z]):
            # Time domain
            time_feats = self.time_domain_features(signal)
            for key, val in time_feats.items():
                all_features[f'{axis}_{key}'] = val
            
            # Frequency domain
            freq_feats = self.frequency_domain_features(signal, fs)
            for key, val in freq_feats.items():
                all_features[f'{axis}_{key}'] = val
        
        # Combined magnitude
        magnitude = np.sqrt(signal_x**2 + signal_y**2 + signal_z**2)
        mag_time = self.time_domain_features(magnitude)
        mag_freq = self.frequency_domain_features(magnitude, fs)
        
        for key, val in mag_time.items():
            all_features[f'mag_{key}'] = val
        for key, val in mag_freq.items():
            all_features[f'mag_{key}'] = val
        
        return all_features

# Usage
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    df = pd.read_csv('data/processed/healthy_processed.csv')
    
    # Extract features from first 1000 samples
    features = extractor.extract_features(
        df['ax_norm'][:1000].values,
        df['ay_norm'][:1000].values,
        df['az_norm'][:1000].values
    )
    
    print(f"Extracted {len(features)} features")
    print(features)