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
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import RFE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, precision_recall_curve, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN, RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import fbeta_score
import numpy as np
from scipy import interp
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

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

# New FeatureStore class for persistent storage
class FeatureStore:
    def __init__(self, store_path="feature_store"):
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        
    def save_features(self, features, labels, feature_names, scaler, selector, metadata):
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
        store_file = os.path.join(self.store_path, 'feature_store.pkl')
        if not os.path.exists(store_file):
            raise FileNotFoundError("Feature store not found. Extract features first.")
            
        with open(store_file, 'rb') as f:
            data = pickle.load(f)
            
        return data


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
        element_counts = {}
        attribute_counts = 0
        text_lengths = []
        nesting_depths = []
        attribute_values = []
        siblings = []
        paths = set()
        empty_elements = 0
        repeated_elements = {}
        leaf_nodes = 0
        unique_attribute_names = set()
        numeric_content_count = 0
        
        for event, elem in context:
            if event == 'start':
                depth += 1
                max_depth = max(max_depth, depth)
                tags.add(elem.tag)
                total_tags += 1
                
                # Element counts and repetition
                element_counts[elem.tag] = element_counts.get(elem.tag, 0) + 1
                if element_counts[elem.tag] > 1:
                    repeated_elements[elem.tag] = repeated_elements.get(elem.tag, 0) + 1
                
                # Attribute tracking
                attribute_counts += len(elem.attrib)
                for attr in elem.attrib.keys():
                    unique_attribute_names.add(attr)
                
                # Text content analysis
                if elem.text and elem.text.strip():
                    text_lengths.append(len(elem.text.strip()))
                    if elem.text.strip().isdigit():
                        numeric_content_count += 1
                else:
                    empty_elements += 1
                    
                # Structure analysis    
                nesting_depths.append(depth)
                paths.add('/'.join([p.tag for p in elem.iterancestors()]))
                
                # Count leaf nodes
                if len(list(elem)) == 0:
                    leaf_nodes += 1
                
                # Sibling analysis
                parent = elem.getparent()
                if parent is not None:
                    siblings.append(len(parent))
                    
                # Track attribute values
                for value in elem.attrib.values():
                    attribute_values.append(len(value))
                        
            elif event == 'end':
                depth -= 1
                elem.clear()
        
        file_size = os.path.getsize(xml_path)
        
        # Calculate metrics
        avg_text_length = np.mean(text_lengths) if text_lengths else 0
        tag_diversity = len(tags) / total_tags if total_tags > 0 else 0
        avg_attributes = attribute_counts / total_tags if total_tags > 0 else 0
        most_common_tag_freq = max(element_counts.values()) / total_tags if total_tags > 0 else 0
        nesting_variance = np.var(nesting_depths) if nesting_depths else 0
        attribute_value_variance = np.var(attribute_values) if attribute_values else 0
        sibling_mean = np.mean(siblings) if siblings else 0
        sibling_variance = np.var(siblings) if siblings else 0
        path_diversity = len(paths) / total_tags if total_tags > 0 else 0
        empty_ratio = empty_elements / total_tags if total_tags > 0 else 0
        leaf_ratio = leaf_nodes / total_tags if total_tags > 0 else 0
        repetition_ratio = len(repeated_elements) / len(tags) if len(tags) > 0 else 0
        unique_attribute_ratio = len(unique_attribute_names) / total_tags if total_tags > 0 else 0
        numeric_content_ratio = numeric_content_count / total_tags if total_tags > 0 else 0
        
        features = [
            len(tags),                # Unique tags count
            total_tags,               # Total tags
            max_depth,                # Maximum nesting depth
            file_size,                # File size
            tag_diversity,            # Tag diversity
            avg_attributes,           # Average attributes per element
            most_common_tag_freq,     # Most common tag frequency
            avg_text_length,          # Average text content length
            attribute_counts,         # Total attributes
            total_tags / file_size,   # Tag density
            nesting_variance,         # Nesting depth variance
            attribute_value_variance, # Attribute value length variance
            sibling_mean,             # Average siblings per node
            sibling_variance,         # Variance in sibling counts
            path_diversity,           # Unique paths ratio
            empty_ratio,              # Empty elements ratio
            leaf_ratio,               # Leaf node ratio
            repetition_ratio,         # Element repetition ratio
            unique_attribute_ratio,   # Unique attribute names ratio
            numeric_content_ratio     # Numeric content ratio
        ]
        return features
           
    except Exception as e:
        print(f"Traditional feature extraction error for {xml_path}: {e}")
        return [0] * 20
    
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
        
        # Calculate semantic features
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
                    # Improved prompt for generative analysis
                prompt = f"""
                Analyze this XML content and assess its potential for errors, focusing on the following aspects:
                XML Sample: {truncated_content}

                1. **Structural Complexity**:
                   - Assess the overall structure of the XML. Is it simple or complex?
                   - Are there any deeply nested elements or redundant tags?

                2. **Tag-Value Relationships**:
                   - Are there any discrepancies between tags and their values?
                   - For example, does a numeric tag (e.g., <cost>) contain non-numeric values?
                   - Does a text-based tag (e.g., <coverage>) contain unexpected values?

                3. **Numeric Data Consistency**:
                   - Are there any numeric values that are out of range or invalid?
                   - For example, are there negative values where only positive values are expected?

                4. **Date Format Consistency**:
                   - Are all date fields in the correct format?
                   - For example, are there invalid dates (e.g., month > 12, day > 31)?

                5. **Contextual Consistency**:
                   - Do the tag-value relationships make sense in the context of healthcare plans?
                   - For example, does the <coverage> tag contain valid values (e.g., "Full", "Partial")?

                6. **Potential Error Indicators**:
                   - Are there any missing required tags or attributes?
                   - Are there any invalid or unexpected values?

                Provide a concise assessment of the XML content, highlighting any potential errors or inconsistencies.
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
                    'contextual_consistency': self._extract_contextual_consistency(analysis_text),
                    'tag_value_discrepancy': self._extract_tag_value_discrepancy(analysis_text),  # New feature
                    'numeric_data_anomalies': self._extract_numeric_data_anomalies(analysis_text),  # New feature
                    'date_format_consistency': self._extract_date_format_consistency(analysis_text),  # New feature
                    'semantic_similarity': self._calculate_semantic_similarity(truncated_content)  # New feature
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
    def _extract_tag_value_discrepancy(analysis_text):
        """
        Extract the likelihood of tag-value discrepancies.
        """
        discrepancy_keywords = ['mismatch', 'discrepancy', 'inconsistent', 'invalid']
        return float(sum(keyword in analysis_text.lower() for keyword in discrepancy_keywords) / len(discrepancy_keywords))

    @staticmethod
    def _extract_numeric_data_anomalies(analysis_text):
        """
        Extract the likelihood of numeric data anomalies.
        """
        numeric_anomaly_keywords = ['out of range', 'invalid number', 'numeric anomaly', 'incorrect value']
        return float(sum(keyword in analysis_text.lower() for keyword in numeric_anomaly_keywords) / len(numeric_anomaly_keywords))

    @staticmethod
    def _extract_date_format_consistency(analysis_text):
        """
        Extract the likelihood of date format inconsistencies.
        """
        date_format_keywords = ['date format', 'invalid date', 'inconsistent date', 'incorrect format']
        return float(sum(keyword in analysis_text.lower() for keyword in date_format_keywords) / len(date_format_keywords))

    @staticmethod
    def _calculate_semantic_similarity(xml_content):
        """
        Calculate the semantic similarity between the XML content and known error patterns.
        """
        # Example: Compare with a set of known error patterns
        error_patterns = [
            "missing required attribute",
            "invalid tag name",
            "out of range value",
            "inconsistent date format",
            "duplicate element"
        ]
        
        # Calculate similarity scores (e.g., using cosine similarity)
        similarity_scores = []
        for pattern in error_patterns:
            # Use embeddings to calculate similarity
            pattern_embedding = self.embedding_model.get_embeddings([pattern])[0].values
            content_embedding = self.embedding_model.get_embeddings([xml_content])[0].values
            similarity = np.dot(pattern_embedding, content_embedding) / (np.linalg.norm(pattern_embedding) * np.linalg.norm(content_embedding))
            similarity_scores.append(similarity)
        
        # Return the maximum similarity score
        return float(np.max(similarity_scores))

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
            'contextual_consistency': 0.5,
            'tag_value_discrepancy': 0.5,  # New feature
            'numeric_data_anomalies': 0.5,  # New feature
            'date_format_consistency': 0.5,  # New feature
            'semantic_similarity': 0.5  # New feature
        }

    @staticmethod
    def _default_complexity_scores():
        return {
            'structural_complexity': 0.5,
            'semantic_error_probability': 0.5,
            'missing_tags': 0.0,
            'invalid_values': 0.0,
            'data_consistency': 0.5,
            'contextual_consistency': 0.5,
            'tag_value_discrepancy': 0.5,  # New feature
            'numeric_data_anomalies': 0.5,  # New feature
            'date_format_consistency': 0.5,  # New feature
            'semantic_similarity': 0.5  # New feature
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
        self.enhanced_classifier = None
        
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
            print("Starting classifier training process...")
            if use_stored_features:
                print("Loading stored features...")
                data = self.feature_store.load_features()
                X = data['features']
                y = data['labels']
                scaler = data['scaler']
                selector = data['selector']
                X_scaled = scaler.transform(X)
                X_selected = selector.transform(X_scaled)
                print(f"Loaded features shape: {X_selected.shape}")
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

            print("Initializing enhanced classifier...")
            self.enhanced_classifier = EnhancedXMLClassifier(random_state=42)
            
            print("Training classifier...")
            self.enhanced_classifier.fit(X_selected, y)
            
            print("Saving components...")
            self.scaler = scaler
            self.selector = selector
            self.classifier = self.enhanced_classifier
            self.threshold = self.enhanced_classifier.threshold
            
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
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.selector = model_data['selector']
            self.threshold = model_data['threshold']
            print(f"Model, scaler,selector and threshold successfully loaded from {filepath}")
    
            return True
        except Exception as e:
            print(f"Error loading model : {e}")
            return False
        
    
    
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

class EnhancedXMLClassifier(BaseEstimator, ClassifierMixin):
    """Optimized classifier with ADASYN sampling"""
    
    def __init__(self, random_state=42, n_jobs=-1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.threshold = None
        self.base_estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=500,  # Increased from 200
                max_depth=15,      # Increased from 10
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight={0:1, 1:8},  # Increased weight for minority class
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )),
            ('gbc', GradientBoostingClassifier(
                n_estimators=300,  # Increased from 100
                learning_rate=0.05,  # Decreased for better generalization
                max_depth=7,
                subsample=0.8,
                random_state=self.random_state+1
            )),
            ('lgbm', LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                class_weight={0:1, 1:8},
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=self.n_jobs,
                random_state=self.random_state+2
            )),
            ('xgb', XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                scale_pos_weight=8,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=self.n_jobs,
                random_state=self.random_state+3
            )),
            ('catboost', CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=7,
                scale_pos_weight=8,
                subsample=0.8,
                random_seed=self.random_state+4,
                verbose=0
            ))
        ]
            # Meta classifier
        self.meta_classifier = LogisticRegression(
            C=0.1,  # Stronger regularization
            class_weight={0:1, 1:8},
            max_iter=2000,
            random_state=self.random_state
        )
        
        self.classifier = StackingClassifier(
            estimators=self.base_estimators,
            final_estimator=self.meta_classifier,
            cv=5,
            stack_method='predict_proba',
            n_jobs=self.n_jobs
        )
        
    def fit(self, X, y):
        """Efficient training process with ADASYN"""
        
        # Calculate sampling ratio based on class distribution
        n_majority = sum(y == 0)
        desired_ratio = 0.4  # Aim for minority class to be 40% of majority
        
        print("Applying ADASYN sampling...")
        try:
            adasyn = ADASYN(
                sampling_strategy=desired_ratio,
                n_neighbors=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            
        except RuntimeError as e:
            print(f"ADASYN failed: {e}. Falling back to RandomOverSampler...")
            ros = RandomOverSampler(
                sampling_strategy=desired_ratio,
                random_state=self.random_state
            )
            X_resampled, y_resampled = ros.fit_resample(X, y)
        
        print(f"Original class distribution: {np.bincount(y)}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        print(f"Training data shape after sampling: {X_resampled.shape}")
        
        # Calculate sample weights
        sample_weights = compute_sample_weights(y_resampled)
        
        # Train each model separately with sample weights
        print("Training ensemble...")
        self.classifier.fit(X_resampled, y_resampled)
        
        # Optimize threshold
        print("Optimizing threshold...")
        self._optimize_threshold(X_resampled, y_resampled)
        
        return self    
    
    def _optimize_threshold(self, X, y):
        """Threshold optimization with F-beta score"""
        print("Calculating probabilities for threshold optimization...")
        y_prob = self.classifier.predict_proba(X)[:, 1]
        
        # Use F-beta score with beta=2 to favor recall
        thresholds = np.linspace(0.2, 0.8, 20)
        best_score = 0
        best_threshold = 0.5
        
        print("Testing thresholds...")
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            # F-beta score with beta=2 weights recall higher than precision
            fbeta = fbeta_score(y, y_pred, beta=2)
            
            if fbeta > best_score:
                best_score = fbeta
                best_threshold = threshold
        
        print(f"Selected threshold: {best_threshold:.3f}")
        self.threshold = best_threshold
    
    def predict_proba(self, X):
        return self.classifier.predict_proba(X)
    
    def predict(self, X):
        """Make predictions using the optimized threshold"""
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

def compute_sample_weights(y):
    """Compute sample weights inversely proportional to class frequencies"""
    print(f"y shape: {y.shape}, y type: {type(y)}")  # Debug: Check shape and type of y
    unique_classes = np.unique(y)
    print(f"Unique classes in y: {unique_classes}")  # Debug: Check unique classes
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
    print(f"class_weights: {class_weights}, type: {type(class_weights)}")  # Debug: Check class_weights
    
    # Ensure class_weights is a 1D array
    if not isinstance(class_weights, np.ndarray):
        class_weights = np.array(class_weights)
    
    # Map class weights to sample weights
    sample_weights = np.zeros_like(y, dtype=np.float64)
    for i, cls in enumerate(unique_classes):
        sample_weights[y == cls] = class_weights[i]
    
    print(f"sample_weights shape: {sample_weights.shape}, type: {sample_weights.dtype}")  # Debug: Check sample_weights
    print(f"sample_weights[:10]: {sample_weights[:10]}")  # Debug: Check first 10 sample weights
    return sample_weights
    
# Parallel XML Prediction Implementation

def process_single_prediction(args):
    classifier, xml_file, input_folder = args
    full_path = os.path.join(input_folder, xml_file)
    try:
        prediction = classifier.predict_xml_error(full_path)
        return {
            'filepath': full_path,
            'filename': prediction['filename'],
            'is_error': prediction['is_error'],
            'error_probability': prediction['error_probability'],
            'confidence': prediction['confidence'],
            'threshold_used': prediction['threshold_used']
        }
    except Exception as e:
        return {
            'filepath': full_path,
            'filename': f"Error in prediction: {str(e)}",
            'is_error': "Error in prediction",
            'error_probability': None,
            'confidence': None,
            'threshold_used': None
        }
def predict_multiple_xmls(classifier, input_folder, results_csv, max_workers=None):
    """
    Parallel processing version of predict_multiple_xmls
    
    Args:
        classifier: XMLErrorClassifier instance
        input_folder: Path to folder containing XML files
        results_csv: Path to output CSV file
        max_workers: Maximum number of parallel workers (defaults to CPU count - 1)
    """
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    xml_files = [f for f in os.listdir(input_folder) if f.endswith('.xml')]
    total_files = len(xml_files)
    print(f"Files to predict: {total_files}")
    
    # Prepare arguments for parallel processing
    process_args = [(classifier, xml_file, input_folder) for xml_file in xml_files]
    
    # Process in parallel with progress bar
    predictions_array = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_prediction, args) for args in process_args]
        
        with tqdm(total=total_files, desc="Predicting XML files") as pbar:
            for future in futures:
                try:
                    result = future.result()
                    predictions_array.append(result)
                except Exception as e:
                    print(f"Error in worker process: {e}")
                pbar.update(1)
    
    # Sort predictions by filename to maintain consistent order
    predictions_array.sort(key=lambda x: x['filename'])
    
    # Write results to CSV
    with open(results_csv, 'w', newline='') as csvfile:
        fieldnames = ['filepath', 'filename', 'is_error', 'error_probability', 'confidence', 'threshold_used']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pred in predictions_array:
            writer.writerow(pred)
    
    print(f"\nPredictions Complete! Results saved to {results_csv}")
    
    # Return summary statistics
    success_count = sum(1 for p in predictions_array if p['is_error'] != "Error in prediction")
    error_count = sum(1 for p in predictions_array if p['is_error'] == "Error in prediction")
    
    print(f"\nSummary:")
    print(f"Successfully processed: {success_count}/{total_files}")
    print(f"Failed to process: {error_count}/{total_files}")
    
    return predictions_array
    
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
    
    # Train enhanced model using stored features
    print("Training enhanced classifier...")
    if classifier.train_classifier(use_stored_features=True):
        print("Model training completed successfully")
        print(f"Selected sampling strategy: {classifier.enhanced_classifier.sampling_strategy.best_sampler.__class__.__name__}")
        print(f"Optimized threshold: {classifier.enhanced_classifier.threshold:.3f}")
        classifier.save_model(MODEL_PATH))
    
    # For future training without re-extracting features:
    classifier = XMLErrorClassifier()  # No need for folders/csvs
    if classifier.train_classifier(use_stored_features=True):
        print("Model trained using stored features")
        classifier.save_model("new_model.pkl")
    
    # For predictions
    test_classifier = XMLErrorClassifier(project_id=PROJECT_ID)
    if test_classifier.load_model(MODEL_PATH):
        input_folder = "path/to/predict"
        output_csv = "predictions.csv"
        # Use 75% of available CPU cores
        max_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
        predict_multiple_xmls(test_classifier, input_folder, output_csv, max_workers=max_workers)

if __name__ == "__main__":
    main()
