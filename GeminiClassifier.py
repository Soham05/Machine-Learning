import os
import glob
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import RFE

# Vertex AI and Gemini Imports remain the same...
from vertexai.language_models import TextGenerationModel
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import vertexai
from google.cloud import aiplatform

import time
from ratelimit import limits, sleep_and_retry
import threading
import json
import csv

class FeatureStore:
    def __init__(self, store_path="feature_store"):
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        
    def save_features(self, features, labels, feature_names, scaler, selector, metadata):
        """
        Save extracted features and related components
        """
        data = {
            'features': features,
            'labels': labels,
            'feature_names': feature_names,
            'scaler': scaler,
            'selector': selector,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        store_file = os.path.join(self.store_path, 'feature_store.pkl')
        with open(store_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Features and components saved to {store_file}")
        
    def load_features(self):
        """
        Load saved features and components
        """
        store_file = os.path.join(self.store_path, 'feature_store.pkl')
        if not os.path.exists(store_file):
            raise FileNotFoundError("Feature store not found. Extract features first.")
            
        with open(store_file, 'rb') as f:
            data = pickle.load(f)
            
        return data

class XMLFeatureExtractor:
    def __init__(self, project_id):
        self.project_id = project_id
        self.feature_extractor = FeatureExtractor(project_id)
        self.feature_store = FeatureStore()
        
    def extract_and_store_features(self, q1_folder, q2_folder, q1_error_csv, q2_error_csv):
        """
        Extract features from XML files and store them
        """
        # Read error files
        q1_errors = pd.read_csv(q1_error_csv)['filename'].tolist()
        q2_errors = pd.read_csv(q2_error_csv)['filename'].tolist()
        all_error_files = set(q1_errors + q2_errors)
        
        # Get all XML files
        xml_files = []
        for folder in [q1_folder, q2_folder]:
            xml_files.extend(glob.glob(os.path.join(folder, '*.xml')))
            
        features = []
        labels = []
        
        # Process files with rate limiting
        num_cores = min(multiprocessing.cpu_count(), 4)
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            with tqdm(total=len(xml_files), desc="Extracting features") as pbar:
                for xml_file in xml_files:
                    is_error = os.path.splitext(os.path.basename(xml_file))[0] in all_error_files
                    future = executor.submit(
                        self.feature_extractor.process_single_file,
                        xml_file,
                        is_error
                    )
                    feature_vector, label = future.result()
                    
                    if feature_vector is not None and label is not None:
                        features.append(feature_vector)
                        labels.append(label)
                    pbar.update(1)
                    
                    # Add delay for API rate limiting
                    time.sleep(0.1)
                    
        features = np.array(features)
        labels = np.array(labels)
        
        # Create and fit scaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Create and fit selector
        selector = RFE(
            RandomForestClassifier(n_estimators=100, random_state=42),
            n_features_to_select=15,
            step=1
        )
        selector.fit(features_scaled, labels)
        
        # Create feature names (you may want to customize these)
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        # Save everything to feature store
        metadata = {
            'num_samples': len(features),
            'num_features': features.shape[1],
            'class_distribution': np.bincount(labels).tolist(),
            'extraction_date': datetime.now().isoformat()
        }
        
        self.feature_store.save_features(
            features,
            labels,
            feature_names,
            scaler,
            selector,
            metadata
        )
        
        return features, labels, scaler, selector

class EnhancedXMLClassifier:
    def __init__(self, feature_store_path="feature_store"):
        self.feature_store = FeatureStore(feature_store_path)
        self.classifier = None
        self.threshold = None
        
    def train_classifier(self, test_size=0.2):
        """
        Train classifier using stored features
        """
        try:
            # Load stored features and components
            data = self.feature_store.load_features()
            features = data['features']
            labels = data['labels']
            scaler = data['scaler']
            selector = data['selector']
            
            # Scale and select features
            features_scaled = scaler.transform(features)
            features_selected = selector.transform(features_scaled)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_selected, labels,
                test_size=test_size,
                stratify=labels,
                random_state=42
            )
            
            # Apply SMOTE-Tomek
            smote_tomek = SMOTETomek(random_state=42)
            X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
            
            # Create and train classifier
            self.classifier = self._create_stacking_classifier()
            self.classifier.fit(X_resampled, y_resampled)
            
            # Find best threshold
            y_prob = self.classifier.predict_proba(X_test)
            self.threshold = self._find_best_threshold(y_test, y_prob)
            
            # Save components for prediction
            self.scaler = scaler
            self.selector = selector
            
            return True
            
        except Exception as e:
            print(f"Error during training: {e}")
            return False
            
    def _create_stacking_classifier(self):
        rf1 = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        
        rf2 = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=3,
            class_weight='balanced',
            random_state=43
        )
        
        gbc = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=44
        )
        
        estimators = [
            ('rf1', rf1),
            ('rf2', rf2),
            ('gbc', gbc)
        ]
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(class_weight='balanced'),
            cv=5,
            n_jobs=-1
        )
        
    def _find_best_threshold(self, y_test, y_prob, thresholds=[0.3, 0.4, 0.5]):
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_prob[:, 1] >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred, average='weighted')
            print(f"\nThreshold {threshold}:")
            print(classification_report(y_test, y_pred))
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        print(f"\nBest threshold: {best_threshold}")
        return best_threshold
        
    def save_model(self, filepath):
        if not self.classifier:
            raise ValueError("Model not trained. Call train_classifier() first.")
            
        model_data = {
            'classifier': self.classifier,
            'threshold': self.threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.classifier = model_data['classifier']
        self.threshold = model_data['threshold']
        
        # Load features store components
        data = self.feature_store.load_features()
        self.scaler = data['scaler']
        self.selector = data['selector']
        
    def predict_xml_error(self, new_xml_path):
        if not self.classifier:
            raise ValueError("Model not trained/loaded.")
            
        try:
            # Use original feature extractor for new predictions
            feature_extractor = FeatureExtractor(None)  # project_id not needed for prediction
            features, _ = feature_extractor.process_single_file(new_xml_path, False)
            
            if features is None:
                raise ValueError(f"Could not process file: {new_xml_path}")
                
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            features_selected = self.selector.transform(features_scaled)
            
            prediction = self.classifier.predict(features_selected)
            probability = self.classifier.predict_proba(features_selected)
            
            confidence = abs(probability[0][1] - 0.5) * 2
            
            return {
                "is_error": bool(prediction[0]),
                "error_probability": float(probability[0][1]),
                "confidence": float(confidence),
                "threshold_used": self.threshold,
                "filename": os.path.basename(new_xml_path)
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

def predict_multiple_xmls(classifier, input_folder, results_csv):
    xml_files = [f for f in os.listdir(input_folder) if f.endswith('.xml')]
    print(f"Files to predict: {len(xml_files)}")
    predictions_array = []
    
    with tqdm(total=len(xml_files), desc="Predicting XML files") as pbar:
        for xml_file in xml_files:
            full_path = os.path.join(input_folder, xml_file)
            try:
                prediction = classifier.predict_xml_error(full_path)
                predictions_array.append({
                    'filepath': full_path,
                    'filename': prediction['filename'],
                    'is_error': prediction['is_error'],
                    'error_probability': prediction['error_probability'],
                    'confidence': prediction['confidence'],
                    'threshold_used': prediction['threshold_used']
                })
            except Exception as e:
                predictions_array.append({
                    'filepath': full_path,
                    'filename': "Error in prediction",
                    'is_error': "Error in prediction",
                    'error_probability': None,
                    'confidence': None,
                    'threshold_used': None
                })
            pbar.update(1)
    
    with open(results_csv, 'w', newline='') as csvfile:
        fieldnames = ['filepath', 'filename', 'is_error', 'error_probability', 'confidence', 'threshold_used']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pred in predictions_array:
            writer.writerow(pred)
    print("\nPredictions Complete!")

def main():
    import sys
    if "google.colab" not in sys.modules:
        import subprocess
        compute_project_id = subprocess.check_output(
            ["gcloud", "config", "get-value", "project"], text=True
        ).strip()
    
    # Configuration
    PROJECT_ID = compute_project_id
    Q1_FOLDER = ""  # Your Q1 folder path
    Q2_FOLDER = ""  # Your Q2 folder path
    Q1_ERROR_CSV = ""  # Your Q1 error CSV path
    Q2_ERROR_CSV = ""  # Your Q2 error CSV path
    MODEL_PATH = "model.pkl"
    FEATURE_STORE_PATH = "feature_store"
    
    # Choose operation mode
    mode = "predict"  # Options: "extract_features", "train", "predict", "all"
    
    try:
        if mode == "extract_features" or mode == "all":
            print("\n=== Extracting Features ===")
            feature_extractor = XMLFeatureExtractor(PROJECT_ID)
            features, labels, scaler, selector = feature_extractor.extract_and_store_features(
                Q1_FOLDER,
                Q2_FOLDER,
                Q1_ERROR_CSV,
                Q2_ERROR_CSV
            )
            print("Feature extraction and storage complete!")
        
        if mode == "train" or mode == "all":
            print("\n=== Training Model ===")
            classifier = EnhancedXMLClassifier(FEATURE_STORE_PATH)
            if classifier.train_classifier():
                print("Model training completed successfully")
                classifier.save_model(MODEL_PATH)
                print(f"Model saved to {MODEL_PATH}")
            else:
                print("Model training failed!")
                return
        
        if mode == "predict" or mode == "all":
            print("\n=== Making Predictions ===")
            classifier = EnhancedXMLClassifier(FEATURE_STORE_PATH)
            
            if classifier.load_model(MODEL_PATH):
                print("Model loaded successfully")
                
                # Single file prediction (for testing)
                test_xml_file = ""  # Your test file path
                if test_xml_file:
                    print("\nTesting single prediction:")
                    prediction = classifier.predict_xml_error(test_xml_file)
                    if prediction:
                        print("Prediction results:")
                        print(json.dumps(prediction, indent=2))
                    else:
                        print(f"Failed to predict for file: {test_xml_file}")
                
                # Multiple files prediction
                input_folder = ""  # Your input folder path
                output_csv = ""    # Your output CSV path
                if input_folder and output_csv:
                    print("\nProcessing multiple files:")
                    predict_multiple_xmls(classifier, input_folder, output_csv)
                else:
                    print("Please provide input_folder and output_csv paths for batch prediction")
            else:
                print("Failed to load model!")
                return
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
