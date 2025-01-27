#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Version 3 
# Implemented Enhanced feature engineering to add more sophisticated features. 
# Implemented stacking classifier to add Gradient Boost classifier and RFC 
# Implemented class imbalance using SMOTEomek
# Feature scaling for better performance
# Implemented dynamic threshold optimization to help balance precision and recall
# Confidence scores in predictions

#This is code is trained for all the xml files for Q1 and Q2 for 2024. 


# In[1]:


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
            
            for event, elem in context:
                if event == 'start':
                    depth += 1
                    max_depth = max(max_depth, depth)
                    tags.add(elem.tag)
                    total_tags += 1
                    
                    # Count element occurances
                    element_counts[elem.tag] = element_counts.get(elem.tag, 0) + 1
                    
                    # Count attributes
                    attribute_counts += len(elem.attrib)
                    
                    # Track text content length
                    if elem.text and elem.text.strip():
                        text_lengths.append(len(elem.text.strip()))
                        
                elif event == 'end':
                    depth -= 1
                    elem.clear()
            
            file_size = os.path.getsize(xml_path)
            
            # Calculate advanced meterics
            avg_text_length = np.mean(text_lengths) if text_lengths else 0
            tag_diversity = len(tags)/ total_tags if total_tags > 0 else 0
            avg_attributes = attribute_counts/total_tags if total_tags > 0 else 0
            most_common_tag_freq = max(element_counts.values())/ total_tags if total_tags > 0 else 0
            
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
                total_tags / file_size # Tag density
            
            ]
            return features
            
        except Exception as e:
            print(f"Traditional feature extraction error for {xml_path}: {e}")
            return [0] * 10 # Return zeros for all features
    
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


# In[6]:


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


# In[7]:


class XMLErrorClassifier:
    def __init__(self, project_id, q1_folder, q2_folder, q1_error_csv, q2_error_csv, use_checkpoints = False):
        self.project_id = project_id
        self.q1_folder = q1_folder
        self.q2_folder = q2_folder
        self.q1_error_csv = q1_error_csv
        self.q2_error_csv = q2_error_csv
        self.classifier = None
        self.feature_extractor = FeatureExtractor(project_id)
        self.use_checkpoints = use_checkpoints
        
        # Create checkpoint directory
        if use_checkpoints:
            self.checkpoint_dir = "checkpoints"
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def process_single_file(self, xml_file, is_error):
        try:
            #print("\n in process_single_file")
            # Extract traditional features
            trad_features = self.feature_extractor.extract_traditional_features(xml_file)
            #print(f"Traditional Features shape: {len(trad_features)}")
            
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
            
            #print(f"Semantic Features shape: {len(semantic_vector)}")
            final_features = trad_features + semantic_vector
            #print(f"Combined Features shape: {len(final_features)}")
            return final_features, is_error
            
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            return None, None
    
    def prepare_dataset(self):
        try: 
            # Check for checkpoints only if enabled
            if self.use_checkpoints:
                latest_checkpoint = self._get_latest_checkpoint()
                if latest_checkpoint:
                    print(f"Resuming from checkpoint: {latest_checkpoint}")
                    with open(latest_checkpoint, 'rb') as f:
                        return pickle.load(f)
            else:
                # clean up old checkpoints if they exist
                if hasattr(self,'checkpoint_dir') and os.path.exists(self.checkpoint_dir):
                    import shutil
                    shutil.rmtree(self.checkpoint_dir)
                    print("Removed old checkpoints directory")
        
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
                            is_error = os.path.splitext(os.path.basename(xml_file))[0] in all_error_files
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

                        # Create checkpoints if enabled
                        if self.use_checkpoints and len(features) % 500 == 0:
                            self._save_checkpoint(features,labels)


                        # Add delay between batches
                        time.sleep(0.1)
            print(f"Final dataset size: {len(features)} samples")
            return np.array(features), np.array(labels)
        
        except Exception as e:
            print(f"Error in prepare_dataset: {e}")
            raise
    
    
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
            
            # Scale features
            scalar = StandardScaler()
            X_scaled = scalar.fit_transform(X)
            
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, stratify=y, random_state=42
            )
            
            # Advanced resampling using SMOTEomek to handle imbalanced dataset
            print("Applying SMOTEomek for class balance...")
            smote_tomek = SMOTETomek(random_state=42)
            X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
            
            # Create and train advanced classifier
            print("Training Advanced Ensemble Classifier...")
            advanced_classifier = AdvancedXMLClassifier()
            self.classifier = advanced_classifier.create_ensemble()
            
            # Train model
            self.classifier.fit(X_resampled, y_resampled)
            
            # Evaluate with different thresholds
            print("Evaluating model performance...")
            y_prob = self.classifier.predict_proba(X_test)
            
            # Test different thresholds
            
            thresholds = [0.3,0.4,0.5]
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in thresholds:
                y_pred = (y_prob[:, 1] >= threshold).astype(int)
                f1 = f1_score(y_test, y_pred, average = 'weighted')
                print(f"\nThreshold {threshold}:")
                print(classification_report(y_test,y_pred))
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            print(f"\nBest threshold: {best_threshold}")
            self.threshold = best_threshold
            
            # Save the scalar
            self.scalar = scalar
            
            return True
            
        except Exception as e:
            print(f"Error during training: {e}")
            return False
    
    def save_model(self, filepath):
        if not self.classifier or not hasattr(self, 'scalar'):
            raise ValueError("Model not trained. Call train_classifier() first.")
        
        try:
            # Save both model and scalar as a dictionary
            model_data = {
            'classifier': self.classifier,
            'scaler' : self.scalar,
            'threshold' : self.threshold
                
            }
            
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model, scaler and threshold successfully saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.scalar = model_data['scaler']
            self.threshold = model_data['threshold']

            print(f"Model, scaler and threshold successfully loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_xml_error(self, new_xml_path):
        if not self.classifier or not hasattr(self, 'scalar'):
            raise ValueError("Model not trained. Call train_classifier() first.")
        
        try:
            # Process single file
            #print("here before calling process_single_file")
            features, _ = self.process_single_file(new_xml_path, False)
            #print(f"In predict_xml_error Features shape before reshape: {(features)}")
            
            if features is None:
                raise ValueError(f"Could not process file: {new_xml_path}")
            
            features_array = np.array(features, dtype=float).reshape(1, -1)
            features_scaled = self.scalar.transform(features_array)
            #print(f"In predict_xml_error Features shape after scaling : {(features_scaled)}")
            
            # Reshape features to 2d array
            #features_2d = np.array(features).reshape(1,-1)
            #print(f"In predict_xml_error Features shape after reshape: {features_2d.shape}")
            
            # Scale features
            
            
            # Predict with custom thresholds
            #features_array = np.array(features).reshape(1, -1)
            prediction = self.classifier.predict(features_scaled)
            probability = self.classifier.predict_proba(features_scaled)
            #prediction = (probability[0][1] >= self.threshold)
            
            
            
            
            # Calculate Confidence score
            confidence = abs(probability[0][1] - 0.5) * 2 # Scale to 0-1
            
            
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


# In[8]:


# Logic to predict multiple xmls

def predict_multiple_xmls(classifier,input_folder,results_csv):
    
    # collect xmls
    xml_files = [f for f in os.listdir(input_folder) if f.endswith('.xml')]
    print(f"Files to predict:{len(xml_files)}")
    predictions_array = []
    
    #Predict each xml
    with tqdm(total=len(xml_files), desc="Predicting XML files") as pbar:
        for xml_file in xml_files:
            full_path = os.path.join(input_folder,xml_file)
            #print(full_path)
            try:
                prediction = classifier.predict_xml_error(full_path)

                predictions_array.append({
                    'filepath':full_path,
                    'filename': prediction['filename'],
                    'is_error': prediction['is_error'],
                    'error_probability': prediction['error_probability'],
                    'confidence' : prediction['confidence'],
                    'threshold_used':prediction['threshold_used']

                })
                pbar.update(1)

            except Exception as e:
                predictions_array.append({
                    'filepath':full_path,
                    'filename': "Error in prediction",
                    'is_error': "Error in prediction",
                    'error_probability': None,
                    'confidence' : None,
                    'threshold_used':None

                })
    
    #Write to csv
    print(f"\n Writing predictions to csv: {results_csv}")
    with open(results_csv,'w',newline='') as csvfile:
        fieldnames = ['filepath','filename','is_error','error_probability','confidence','threshold_used']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for pred in predictions_array:
            writer.writerow(pred)
    print("\n Predictions Complete!")


# In[23]:


def main():
    import sys
    if "google.colab" not in sys.modules:
        import subprocess

        compute_project_id = subprocess.check_output(
            ["gcloud", "config", "get-value", "project"], text=True
        ).strip()
    
    PROJECT_ID = compute_project_id
    Q1_FOLDER = ""
    Q2_FOLDER = ""
    
    #Q1_FOLDER = ""
    #Q2_FOLDER = ""
    
    Q1_ERROR_CSV = ""
    Q2_ERROR_CSV = ""
    PICKLE_FILES = ""
    MODEL_PATH = "model.pkl"
    
     # Initialize classifier
    classifier = XMLErrorClassifier(
        PROJECT_ID, 
        Q1_FOLDER, 
        Q2_FOLDER, 
        Q1_ERROR_CSV, 
        Q2_ERROR_CSV,
        use_checkpoints=True # Enable/Disable checkpoints
    )
    
    # Train and save the model
    """
    if classifier.train_classifier():
        print("Model training completed successfully")
        classifier.save_model(MODEL_PATH)   
    
    # Single XMLS predictions Load the model and predict for a new XML file
    if classifier.load_model(MODEL_PATH):
        test_xml_file = ""  # Replace test file path
        prediction = classifier.predict_xml_error(test_xml_file)
        if prediction:
            print("Prediction results:")
            print(json.dumps(prediction,indent=2))
            print(prediction)
        else:
            print(f"Failed to predict for file: {test_xml_file}")
    """
    
    # Multiple files predictions 
    
    if classifier.load_model(MODEL_PATH):
        input_folder = ""
        output_csv = ""
        predict_multiple_xmls(classifier,input_folder,output_csv)
if __name__ == "__main__":
    main()

