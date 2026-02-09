# src/train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale'),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        self.best_model = None
        self.best_model_name = None
    
    def load_data(self, filepath='data/processed/feature_dataset.csv'):
        """Load and split dataset"""
        df = pd.read_csv(filepath)
        
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_all_models(self):
        """Train and compare all models"""
        X_train, X_test, y_train, y_test = self.load_data()
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            print(f"{'='*50}")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred
            }
            
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, 
                                       target_names=self.label_encoder.classes_))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title(f'{name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'data/models/{name}_confusion_matrix.png')
            plt.close()
        
        # Find best model
        self.best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"Accuracy: {results[self.best_model_name]['accuracy']:.4f}")
        print(f"{'='*50}")
        
        return results
    
    def save_model(self, filename='data/models/best_model.pkl'):
        """Save best model and preprocessing objects"""
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_name': self.best_model_name
        }, filename)
        print(f"Model saved to {filename}")

# Usage
if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.train_all_models()
    trainer.save_model()