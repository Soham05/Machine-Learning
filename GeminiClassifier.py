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

# Vertex AI and Gemini Imports
from vertexai.language_models import TextGenerationModel
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import vertexai
from google.cloud import aiplatform

## Import for API quota limits
import time
from ratelimit import limits, sleep_and_retry
import threading
import json
import csv

# In[2]:

compute_service_account = ""
compute_cmek = ""
location = ""
bucket_name = ""
storage_bucket_uri = f""
storage_project_id = ""

# In[3]:

# Global API counter
class APICounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0
        self.reset_time = time.time()
        
    def increment(self):
        with self.lock:
            current_time = time.time()
            if current_time - self.reset_time >= 60:
                self.count = 0
                self.reset_time = current_time
            self.count += 1
            return self.count
    
    def get_count(self):
        with self.lock:
            return self.count

api_counter = APICounter()

# In[4]:

# Rate limiting decorators
@sleep_and_retry
@limits(calls=580, period=60)
def rate_limited_api_call(func, *args, **kwargs):
    return func(*args, **kwargs)

# In[5]:

class FeatureExtractor:
    def __init__(self, project_id):
        self.project_id = project_id
        vertexai.init(project=project_id)
        self.embedding_model = None
        self.generative_model = None
    
    def _initialize_models(self):
        if not self.embedding_model:
            self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        if not self.generative_model:
            self.generative_model = GenerativeModel("gemini-1.5-pro")
            
    
    def extract_traditional_features(self, xml_path):
        try:
            context = ET.iterparse(xml_path, events=('start', 'end'))
            depth = 0
            max_depth = 0
            tags = set()
            total_tags = 0
            element_counts  = {}
            attribute_counts = 0
            text_lengths = []
            nesting_depths = []
            attribute_values = []
            
            for event, elem in context:
                if event == 'start':
                    depth += 1
                    max_depth = max(max_depth, depth)
                    tags.add(elem.tag)
                    total_tags += 1
                    
                    # Count element occurrences
                    element_counts[elem.tag] = element_counts.get(elem.tag, 0) + 1
                    
                    # Count attributes
                    attribute_counts += len(elem.attrib)
                    
                    # Track text content length
                    if elem.text and elem.text.strip():
                        text_lengths.append(len(elem.text.strip()))
                    
                    # Track nesting depths
                    nesting_depths.append(depth)
                    
                    # Track attribute values
                    for attr, value in elem.attrib.items():
                        attribute_values.append(len(value))
                        
                elif event == 'end':
                    depth -= 1
                    elem.clear()
            
            file_size = os.path.getsize(xml_path)
            
            # Calculate advanced metrics
            avg_text_length = np.mean(text_lengths) if text_lengths else 0
            tag_diversity = len(tags)/ total_tags if total_tags > 0 else 0
            avg_attributes = attribute_counts/total_tags if total_tags > 0 else 0
            most_common_tag_freq = max(element_counts.values())/ total_tags if total_tags > 0 else 0
            nesting_variance = np.var(nesting_depths) if nesting_depths else 0
            attribute_value_variance = np.var(attribute_values) if attribute_values else 0
            
            features = [
                len(tags),             # length of tags
                total_tags,            # Total tags count
                max_depth,             # Maximum nesting size
                file_size,             # XML file size
                tag_diversity,         # Tag diversity ratio
                avg_attributes,        # Average attributes per element
                most_common_tag_freq,  # Most common tag frequency
                avg_text_length,       # average text content length
                attribute_counts,      # Total attribute count
                total_tags / file_size, # Tag density
                nesting_variance,      # Variance in nesting depths
                attribute_value_variance # Variance in attribute values
            ]
            return features
            
        except Exception as e:
            print(f"Traditional feature extraction error for {xml_path}: {e}")
            return [0] * 12 # Return zeros for all features
    
    def extract_semantic_features(self, xml_content):
        try:
            self._initialize_models()
            count = api_counter.increment()
            
            # Add safety delay if approaching limit
            if count > 550:
                time.sleep(2)
            
            # Truncate content
            truncated_content = xml_content[:2000]
            
            # Get embeddings with retry logic
            max_retries = 3
            embedding_vector = None
            
            for attempt in range(max_retries):
                try:
                    embeddings = rate_limited_api_call(
                        self.embedding_model.get_embeddings,
                        [truncated_content]
                    )
                    embedding_vector = embeddings[0].values
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2 ** attempt)
            
            if embedding_vector is None:
                return self._default_semantic_features()
            
            features = {
                'embedding_mean': float(np.mean(embedding_vector)),
                'embedding_std': float(np.std(embedding_vector)),
                'embedding_max': float(np.max(embedding_vector)),
                'embedding_min': float(np.min(embedding_vector)),
                'embedding_dimension': len(embedding_vector)
            }
            
            # Only do generative analysis if not hitting rate limits
            if count < 500 and count % 2 == 0:
                try:
                    # Enhanced generative analysis
                    prompt = f"""
                    Analyze this XML content and assess its potential for errors:
                    XML Sample: {truncated_content}
                    Provide a concise assessment of:
                    1. Structural complexity
                    2. Potential error indicators (e.g., missing tags, invalid values)
                    3. Data consistency likelihood (e.g., numeric ranges, date formats)
                    4. Contextual consistency (e.g., tag-value relationships)
                    """
                    
                    generative_response = rate_limited_api_call(
                        self.generative_model.generate_content,
                        prompt
                    )
                    
                    # Extract additional semantic features
                    analysis_text = generative_response.text
                    features.update({
                        'structural_complexity': self._extract_complexity_score(analysis_text),
                        'semantic_error_probability': self._estimate_error_probability(analysis_text),
                        'missing_tags': self._extract_missing_tags(analysis_text),
                        'invalid_values': self._extract_invalid_values(analysis_text),
                        'data_consistency': self._extract_data_consistency(analysis_text),
                        'contextual_consistency': self._extract_contextual_consistency(analysis_text)
                    })
                except Exception:
                    features.update(self._default_complexity_scores())
            else:
                features.update(self._default_complexity_scores())
            
            return features
            
        except Exception as e:
            print(f"Semantic feature extraction error: {e}")
            return self._default_semantic_features()
    
    @staticmethod
    def _extract_complexity_score(analysis_text):
        complexity_keywords = ['complex', 'intricate', 'complicated', 'sophisticated']
        return float(any(keyword in analysis_text.lower() for keyword in complexity_keywords))
    
    @staticmethod
    def _estimate_error_probability(analysis_text):
        error_keywords = ['error', 'issue', 'problem', 'inconsistent', 'incorrect']
        return float(sum(keyword in analysis_text.lower() for keyword in error_keywords) / len(error_keywords))
    
    @staticmethod
    def _extract_missing_tags(analysis_text):
        missing_tag_keywords = ['missing', 'required', 'absent']
        return float(any(keyword in analysis_text.lower() for keyword in missing_tag_keywords))
    
    @staticmethod
    def _extract_invalid_values(analysis_text):
        invalid_value_keywords = ['invalid', 'incorrect', 'out of range']
        return float(any(keyword in analysis_text.lower() for keyword in invalid_value_keywords))
    
    @staticmethod
    def _extract_data_consistency(analysis_text):
        consistency_keywords = ['consistent', 'valid', 'correct']
        return float(any(keyword in analysis_text.lower() for keyword in consistency_keywords))
    
    @staticmethod
    def _extract_contextual_consistency(analysis_text):
        contextual_keywords = ['context', 'relationship', 'appropriate']
        return float(any(keyword in analysis_text.lower() for keyword in contextual_keywords))
    
    @staticmethod
    def _default_semantic_features():
        return {
            'embedding_mean': 0.0,
            'embedding_std': 0.0,
            'embedding_max': 0.0,
            'embedding_min': 0.0,
            'embedding_dimension': 0,
            'structural_complexity': 0.5,
            'semantic_error_probability': 0.5,
            'missing_tags': 0.0,
            'invalid_values': 0.0,
            'data_consistency': 0.5,
            'contextual_consistency': 0.5
        }
    
    @staticmethod
    def _default_complexity_scores():
        return {
            'structural_complexity': 0.5,
            'semantic_error_probability': 0.5,
            'missing_tags': 0.0,
            'invalid_values': 0.0,
            'data_consistency': 0.5,
            'contextual_consistency': 0.5
        }

# Enhanced XMLErrorClassifier that combines original functionality with feature persistence
class XMLErrorClassifier:
    def __init__(self, project_id=None, q1_folder=None, q2_folder=None, 
                 q1_error_csv=None, q2_error_csv=None, use_checkpoints=False,
                 feature_store_path="feature_store"):
        self.project_id = project_id
        self.q1_folder = q1_folder
        self.q2_folder = q2_folder
        self.q1_error_csv = q1_error_csv
        self.q2_error_csv = q2_error_csv
        self.classifier = None
        self.feature_extractor = FeatureExtractor(project_id) if project_id else None
        self.use_checkpoints = use_checkpoints
        self.feature_store = FeatureStore(feature_store_path)
        
        if use_checkpoints:
            self.checkpoint_dir = "checkpoints"
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def extract_and_store_features(self):
        """New method to extract and store features separately"""
        print("Starting feature extraction...")
        X, y = self.prepare_dataset()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Feature selection
        selector = RFE(
            RandomForestClassifier(n_estimators=100, random_state=42),
            n_features_to_select=15,
            step=1
        )
        selector.fit(X_scaled, y)
        
        # Create feature names
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Save to feature store
        metadata = {
            'num_samples': len(X),
            'num_features': X.shape[1],
            'class_distribution': np.bincount(y).tolist(),
            'extraction_date': datetime.now().isoformat()
        }
        
        self.feature_store.save_features(
            X, y, feature_names, scaler, selector, metadata
        )
        
        return X, y, scaler, selector
    
    def train_classifier(self, test_size=0.2, use_stored_features=True):
        try:
            if use_stored_features:
                # Load stored features
                data = self.feature_store.load_features()
                X = data['features']
                y = data['labels']
                scaler = data['scaler']
                selector = data['selector']
                X_scaled = scaler.transform(X)
                X_selected = selector.transform(X_scaled)
            else:
                # Extract new features
                X, y = self.prepare_dataset()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                selector = RFE(
                    RandomForestClassifier(n_estimators=100, random_state=42),
                    n_features_to_select=15,
                    step=1
                )
                X_selected = selector.fit_transform(X_scaled, y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, stratify=y, random_state=42
            )
            
            # Apply SMOTE-Tomek
            smote_tomek = SMOTETomek(random_state=42)
            X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
            
            # Create and train classifier
            advanced_classifier = AdvancedXMLClassifier()
            self.classifier = advanced_classifier.create_ensemble()
            self.classifier.fit(X_resampled, y_resampled)
            
            # Evaluate with different thresholds
            y_prob = self.classifier.predict_proba(X_test)
            self.threshold = self._find_best_threshold(y_test, y_prob)
            
            # Save components
            self.scaler = scaler
            self.selector = selector
            
            return True
            
        except Exception as e:
            print(f"Error during training: {e}")
            return False
    
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
    
    # Original methods remain the same
    def prepare_dataset(self):
        # [Original implementation]
        pass
    
    def process_single_file(self, xml_file, is_error):
        # [Original implementation]
        pass
    
    def save_model(self, filepath):
        if not self.classifier:
            raise ValueError("Model not trained. Call train_classifier() first.")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'selector': self.selector,
            'threshold': self.threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.selector = model_data['selector']
        self.threshold = model_data['threshold']
    
    def predict_xml_error(self, new_xml_path):
        if not self.classifier:
            raise ValueError("Model not trained/loaded.")
        
        try:
            features, _ = self.process_single_file(new_xml_path, False)
            
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

class AdvancedXMLClassifier:
    def __init__(self):
        self.base_models = None
        self.meta_model = None
        self.scaler = StandardScaler()
    
    def create_ensemble(self):
        # create base models
        
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
        
        # Create stacking classifier
        estimators = [
             ('rf1',rf1),
             ('rf2',rf2),
             ('gbc',gbc)
            
        ]
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(class_weight='balanced'),
            cv = 5,
            n_jobs = -1
        )
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
    # Configuration
    PROJECT_ID = "your-project-id"
    Q1_FOLDER = "path/to/q1"
    Q2_FOLDER = "path/to/q2"
    Q1_ERROR_CSV = "path/to/q1_errors.csv"
    Q2_ERROR_CSV = "path/to/q2_errors.csv"
    MODEL_PATH = "model.pkl"
    
    # Initialize classifier
    classifier = XMLErrorClassifier(
        PROJECT_ID,
        Q1_FOLDER,
        Q2_FOLDER,
        Q1_ERROR_CSV,
        Q2_ERROR_CSV,
        use_checkpoints=True
    )
    
    # Extract and store features (only need to do this once)
    features, labels, scaler, selector = classifier.extract_and_store_features()
    print("Features extracted and stored")
    
    # Train model using stored features
    if classifier.train_classifier(use_stored_features=True):
        print("Model training completed successfully")
        classifier.save_model(MODEL_PATH)
    
    # For future training without re-extracting features:
    classifier = XMLErrorClassifier()  # No need for folders/csvs
    if classifier.train_classifier(use_stored_features=True):
        print("Model trained using stored features")
        classifier.save_model("new_model.pkl")
    
    # For predictions
    classifier = XMLErrorClassifier()
    if classifier.load_model(MODEL_PATH):
        input_folder = "path/to/predict"
        output_csv = "predictions.csv"
        predict_multiple_xmls(classifier, input_folder, output_csv)

if __name__ == "__main__":
    main()
