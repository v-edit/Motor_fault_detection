# src/inference.py

import numpy as np
import joblib
import serial
from feature_extraction import FeatureExtractor
from preprocessing import SignalPreprocessor

class RealTimeClassifier:
    def __init__(self, model_path='data/models/best_model.pkl'):
        # Load trained model
        saved_data = joblib.load(model_path)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.label_encoder = saved_data['label_encoder']
        self.model_name = saved_data['model_name']
        
        self.preprocessor = SignalPreprocessor()
        self.extractor = FeatureExtractor()
        
        print(f"Loaded {self.model_name} model")
        print(f"Classes: {self.label_encoder.classes_}")
    
    def predict_from_data(self, ax, ay, az):
        """Predict fault from raw acceleration data"""
        
        # Preprocess
        ax_filtered = self.preprocessor.filter_signal(ax)
        ay_filtered = self.preprocessor.filter_signal(ay)
        az_filtered = self.preprocessor.filter_signal(az)
        
        ax_norm = self.preprocessor.normalize(ax_filtered)
        ay_norm = self.preprocessor.normalize(ay_filtered)
        az_norm = self.preprocessor.normalize(az_filtered)
        
        # Extract features
        features = self.extractor.extract_features(ax_norm, ay_norm, az_norm)
        
        # Convert to array
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Scale
        feature_scaled = self.scaler.transform(feature_array)
        
        # Predict
        prediction = self.model.predict(feature_scaled)[0]
        prediction_proba = self.model.predict_proba(feature_scaled)[0] if hasattr(self.model, 'predict_proba') else None
        
        # Decode label
        label = self.label_encoder.inverse_transform([prediction])[0]
        
        return {
            'prediction': label,
            'confidence': prediction_proba[prediction] if prediction_proba is not None else None,
            'all_probabilities': dict(zip(self.label_encoder.classes_, prediction_proba)) if prediction_proba is not None else None
        }
    
    def run_realtime(self, port='COM3', baudrate=115200, window_size=1000):
        """Run real-time classification from ESP32"""
        
        ser = serial.Serial(port, baudrate, timeout=1)
        
        buffer_x, buffer_y, buffer_z = [], [], []
        
        print("Starting real-time classification...")
        print("Waiting for data...\n")
        
        try:
            while True:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8').strip()
                    try:
                        ax, ay, az = map(float, line.split(','))
                        
                        buffer_x.append(ax)
                        buffer_y.append(ay)
                        buffer_z.append(az)
                        
                        # When buffer is full, classify
                        if len(buffer_x) >= window_size:
                            result = self.predict_from_data(
                                np.array(buffer_x),
                                np.array(buffer_y),
                                np.array(buffer_z)
                            )
                            
                            print(f"\n{'='*50}")
                            print(f"Prediction: {result['prediction']}")
                            if result['confidence']:
                                print(f"Confidence: {result['confidence']:.2%}")
                                print("\nAll probabilities:")
                                for label, prob in result['all_probabilities'].items():
                                    print(f"  {label}: {prob:.2%}")
                            print(f"{'='*50}\n")
                            
                            # Send result back to ESP32
                            ser.write(f"{result['prediction']}\n".encode())
                            
                            # Clear buffer
                            buffer_x, buffer_y, buffer_z = [], [], []
                    
                    except:
                        continue
        
        except KeyboardInterrupt:
            print("\nStopping...")
            ser.close()

# Usage
if __name__ == "__main__":
    classifier = RealTimeClassifier()
    classifier.run_realtime(port='COM3')  # Change for your system