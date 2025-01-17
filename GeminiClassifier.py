import os
import glob
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import pickle
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Vertex AI Imports
import vertexai
from vertexai.language_models import TextEmbeddingModel, GenerativeModel
from google.cloud import aiplatform

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

# Rate limiting decorators
@sleep_and_retry
@limits(calls=580, period=60)
def rate_limited_api_call(func, *args, **kwargs):
    return func(*args, **kwargs)

class FeatureExtractor:
    def __init__(self, project_id):
        self.project_id = project_id
        vertexai.init(project=project_id)
        self.embedding_model = None
        self.generative_model = None
    
    def _initialize_models(self):
        if not self.embedding_model:
            self.embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
        if not self.generative_model:
            self.generative_model = GenerativeModel("gemini-pro")
    
    def extract_traditional_features(self, xml_path):
        try:
            context = ET.iterparse(xml_path, events=('start', 'end'))
            depth = 0
            max_depth = 0
            tags = set()
            total_tags = 0
            
            for event, elem in context:
                if event == 'start':
                    depth += 1
                    max_depth = max(max_depth, depth)
                    tags.add(elem.tag)
                    total_tags += 1
                elif event == 'end':
                    depth -= 1
                    elem.clear()
            
            file_size = os.path.getsize(xml_path)
            return [len(tags), total_tags, max_depth, file_size]
            
        except Exception as e:
            print(f"Traditional feature extraction error for {xml_path}: {e}")
            return [0, 0, 0, 0]
    
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
                    prompt = f"""
                    Analyze this XML content and assess its potential for errors:
                    XML Sample: {truncated_content}
                    Provide a concise assessment of:
                    1. Structural complexity
                    2. Potential error indicators
                    3. Data consistency likelihood
                    """
                    
                    generative_response = rate_limited_api_call(
                        self.generative_model.generate_content,
                        prompt
                    )
                    
                    features.update({
                        'structural_complexity': self._extract_complexity_score(generative_response.text),
                        'semantic_error_probability': self._estimate_error_probability(generative_response.text)
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
    def _default_semantic_features():
        return {
            'embedding_mean': 0.0,
            'embedding_std': 0.0,
            'embedding_max': 0.0,
            'embedding_min': 0.0,
            'embedding_dimension': 0,
            'structural_complexity': 0.5,
            'semantic_error_probability': 0.5
        }
    
    @staticmethod
    def _default_complexity_scores():
        return {
            'structural_complexity': 0.5,
            'semantic_error_probability': 0.5
        }

class XMLErrorClassifier:
    def __init__(self, project_id, q1_folder, q2_folder, q1_error_csv, q2_error_csv):
        self.project_id = project_id
        self.q1_folder = q1_folder
        self.q2_folder = q2_folder
        self.q1_error_csv = q1_error_csv
        self.q2_error_csv = q2_error_csv
        self.classifier = None
        self.feature_extractor = FeatureExtractor(project_id)
        
        # Create checkpoint directory
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def process_single_file(self, xml_file, is_error):
        try:
            # Extract traditional features
            trad_features = self.feature_extractor.extract_traditional_features(xml_file)
            
            # Read and extract semantic features
            with open(xml_file, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            semantic_features = self.feature_extractor.extract_semantic_features(xml_content)
            
            # Convert to feature vector
            semantic_vector = [
                semantic_features['embedding_mean'],
                semantic_features['embedding_std'],
                semantic_features['embedding_max'],
                semantic_features['embedding_min'],
                semantic_features['embedding_dimension'],
                semantic_features['structural_complexity'],
                semantic_features['semantic_error_probability']
            ]
            
            return trad_features + semantic_vector, is_error
            
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            return None, None
    
    def prepare_dataset(self):
        # Load checkpoints if they exist
        latest_checkpoint = self._get_latest_checkpoint()
        if latest_checkpoint:
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            with open(latest_checkpoint, 'rb') as f:
                return pickle.load(f)
        
        # Read error files
        q1_errors = pd.read_csv(self.q1_error_csv)['filename'].tolist()
        q2_errors = pd.read_csv(self.q2_error_csv)['filename'].tolist()
        all_error_files = set(q1_errors + q2_errors)
        
        # Get all XML files
        xml_files = []
        for folder in [self.q1_folder, self.q2_folder]:
            xml_files.extend(glob.glob(os.path.join(folder, '*.xml')))
        
        # Calculate batch size
        num_cores = min(multiprocessing.cpu_count(), 4)  # Limit cores for API rate
        batch_size = max(1, min(50, len(xml_files) // (num_cores * 8)))
        
        features = []
        labels = []
        
        # Process files with rate limiting
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            with tqdm(total=len(xml_files), desc="Processing XML files") as pbar:
                for i in range(0, len(xml_files), batch_size):
                    batch = xml_files[i:i + batch_size]
                    
                    # Create tasks for batch
                    futures = []
                    for xml_file in batch:
                        is_error = os.path.basename(xml_file) in all_error_files
                        futures.append(
                            executor.submit(self.process_single_file, xml_file, is_error)
                        )
                    
                    # Process completed tasks
                    for future in futures:
                        feature_vector, label = future.result()
                        if feature_vector is not None and label is not None:
                            features.append(feature_vector)
                            labels.append(label)
                        pbar.update(1)
                    
                    # Create checkpoint every 500 files
                    if len(features) % 500 == 0:
                        self._save_checkpoint(features, labels)
                    
                    # Add delay between batches
                    time.sleep(0.1)
        
        return np.array(features), np.array(labels)
    
    def _save_checkpoint(self, features, labels):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_{len(features)}_{timestamp}.pkl"
        )
        with open(checkpoint_path, 'wb') as f:
            pickle.dump((np.array(features), np.array(labels)), f)
    
    def _get_latest_checkpoint(self):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pkl"))
        return max(checkpoints) if checkpoints else None
    
    def train_classifier(self, test_size=0.2):
        try:
            print("Starting dataset preparation...")
            X, y = self.prepare_dataset()
            
            print(f"Dataset prepared. Shape: {X.shape}")
            print(f"Class distribution: {np.bincount(y)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
            
            # Handle imbalanced dataset
            print("Applying SMOTE for class balance...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            # Train Random Forest
            print("Training Random Forest classifier...")
            self.classifier = RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5,
                n_jobs=-1
            )
            
            self.classifier.fit(X_resampled, y_resampled)
            
            # Evaluate
            print("Evaluating model performance...")
            y_pred = self.classifier.predict(X_test)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            return True
            
        except Exception as e:
            print(f"Error during training: {e}")
            return False
    
    def save_model(self, filepath):
        if not self.classifier:
            raise ValueError("Model not trained. Call train_classifier() first.")
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.classifier, f)
            print(f"Model successfully saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                self.classifier = pickle.load(f)
            print(f"Model successfully loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_xml_error(self, new_xml_path):
        if not self.classifier:
            raise ValueError("Model not trained. Call train_classifier() first.")
        
        try:
            # Process single file
            features, _ = self.process_single_file(new_xml_path, False)
            
            if features is None:
                raise ValueError(f"Could not process file: {new_xml_path}")
            
            # Predict
            features_array = np.array(features).reshape(1, -1)
            prediction = self.classifier.predict(features_array)
            probability = self.classifier.predict_proba(features_array)
            
            return {
                "is_error": bool(prediction[0]),
                "error_probability": float(probability[0][1]),
                "filename": os.path.basename(new_xml_path)
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

def main():
    # Configuration
    PROJECT_ID = "your-gcp-project-id"
    Q1_FOLDER = "XMLTesting/files/Q1-XMLs/"
    Q2_FOLDER = "XMLTesting/files/Q2-XMLs/"
    Q1_ERROR_CSV = "XMLTesting/files/Q12024_error_xml_filenames.csv"
    Q2_ERROR_CSV = "XMLTesting/files/Q22024_error_xml_filenames.csv"
    MODEL_PATH = "xml_classifier_model.pkl"
    
    # Initialize classifier
    classifier = XMLErrorClassifier(
        PROJECT_ID, 
        Q1_FOLDER, 
        Q2_FOLDER, 
        Q1_ERROR_CSV, 
        Q2_ERROR_CSV
    )
    
    # Train and save the model
    if classifier.train_classifier():
        classifier.save_model(MODEL_PATH)
    
    # Load the model and predict for a new XML file
    if classifier.load_model(MODEL_PATH):
        test_xml_file = "XMLTesting/test_file.xml"  # Replace with your test file path
        prediction = classifier.predict_xml_error(test_xml_file)
        if prediction:
            print("Prediction results:")
            print(prediction)
        else:
            print(f"Failed to predict for file: {test_xml_file}")
    
if __name__ == "__main__":
    main()
