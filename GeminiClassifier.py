import os
import glob
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
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

def analyze_xml_semantic_features(xml_content, project_id):
    """
    Standalone function for semantic feature extraction
    """
    try:
        # Initialize Vertex AI for this process
        vertexai.init(project=project_id)
        embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
        generative_model = GenerativeModel("gemini-pro")
        
        # Truncate content to avoid token limits
        truncated_content = xml_content[:2000]
        
        # Generate embeddings
        embeddings = embedding_model.get_embeddings([truncated_content])
        embedding_vector = embeddings[0].values
        
        # Compute embedding-based statistical features
        semantic_features = {
            'embedding_mean': float(np.mean(embedding_vector)),
            'embedding_std': float(np.std(embedding_vector)),
            'embedding_max': float(np.max(embedding_vector)),
            'embedding_min': float(np.min(embedding_vector)),
            'embedding_dimension': len(embedding_vector)
        }
        
        # Use generative model for additional insights
        try:
            prompt = f"""
            Analyze this XML content and assess its potential for errors:
            
            XML Sample: {truncated_content}
            
            Provide a concise assessment of:
            1. Structural complexity
            2. Potential error indicators
            3. Data consistency likelihood
            """
            
            generative_response = generative_model.generate_content(prompt)
            
            # Extract additional semantic insights
            semantic_features.update({
                'structural_complexity': _extract_complexity_score(generative_response.text),
                'semantic_error_probability': _estimate_error_probability(generative_response.text)
            })
        except Exception as gen_error:
            print(f"Generative model analysis failed: {gen_error}")
            semantic_features.update({
                'structural_complexity': 0.5,
                'semantic_error_probability': 0.5
            })
        
        return semantic_features
        
    except Exception as e:
        print(f"Semantic analysis error: {e}")
        return _default_semantic_features()

def _extract_complexity_score(analysis_text):
    """
    Extract a complexity score from generative model's analysis
    """
    complexity_keywords = ['complex', 'intricate', 'complicated', 'sophisticated']
    return float(any(keyword in analysis_text.lower() for keyword in complexity_keywords))

def _estimate_error_probability(analysis_text):
    """
    Estimate error probability based on generative model's analysis
    """
    error_keywords = ['error', 'issue', 'problem', 'inconsistent', 'incorrect']
    return float(sum(keyword in analysis_text.lower() for keyword in error_keywords) / len(error_keywords))

def _default_semantic_features():
    """
    Provide default semantic features if analysis fails
    """
    return {
        'embedding_mean': 0.0,
        'embedding_std': 0.0,
        'embedding_max': 0.0,
        'embedding_min': 0.0,
        'embedding_dimension': 0,
        'structural_complexity': 0.5,
        'semantic_error_probability': 0.5
    }

def extract_traditional_features(xml_path):
    """
    Standalone function for traditional feature extraction
    """
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
        print(f"Feature extraction error for {xml_path}: {e}")
        return [0, 0, 0, 0]

def process_single_file(args):
    """
    Process a single XML file with both traditional and semantic features
    """
    xml_file, project_id, is_error = args
    
    try:
        # Extract traditional features
        trad_features = extract_traditional_features(xml_file)
        
        # Read XML content
        with open(xml_file, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # Get semantic features
        semantic_analysis = analyze_xml_semantic_features(xml_content, project_id)
        
        # Convert semantic analysis to feature vector
        semantic_feature_vector = [
            semantic_analysis['embedding_mean'],
            semantic_analysis['embedding_std'],
            semantic_analysis['embedding_max'],
            semantic_analysis['embedding_min'],
            semantic_analysis['embedding_dimension'],
            semantic_analysis['structural_complexity'],
            semantic_analysis['semantic_error_probability']
        ]
        
        # Combine features
        features = trad_features + semantic_feature_vector
        
        return features, is_error
        
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")
        return None, None

class XMLErrorClassifier:
    def __init__(self, project_id, q1_folder, q2_folder, q1_error_csv, q2_error_csv):
        self.project_id = project_id
        self.q1_folder = q1_folder
        self.q2_folder = q2_folder
        self.q1_error_csv = q1_error_csv
        self.q2_error_csv = q2_error_csv
        self.classifier = None
        
    def prepare_dataset(self):
        """
        Prepare training dataset using multiprocessing
        """
        # Read error files
        q1_errors = pd.read_csv(self.q1_error_csv)['filename'].tolist()
        q2_errors = pd.read_csv(self.q2_error_csv)['filename'].tolist()
        all_error_files = set(q1_errors + q2_errors)
        
        # Get all XML files
        xml_files = []
        for folder in [self.q1_folder, self.q2_folder]:
            xml_files.extend(glob.glob(os.path.join(folder, '*.xml')))
        
        # Calculate optimal batch size
        num_cores = multiprocessing.cpu_count()
        batch_size = max(1, len(xml_files) // (num_cores * 4))  # 4 batches per core
        
        # Prepare arguments for parallel processing
        process_args = [
            (xml_file, self.project_id, os.path.basename(xml_file) in all_error_files)
            for xml_file in xml_files
        ]
        
        features = []
        labels = []
        
        # Process files in parallel with progress tracking
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            with tqdm(total=len(process_args), desc="Processing XML files") as pbar:
                # Process in batches
                for i in range(0, len(process_args), batch_size):
                    batch_args = process_args[i:i + batch_size]
                    batch_results = list(executor.map(process_single_file, batch_args))
                    
                    for feature, label in batch_results:
                        if feature is not None and label is not None:
                            features.append(feature)
                            labels.append(label)
                        pbar.update(1)
        
        return np.array(features), np.array(labels)
    
    def train_classifier(self, test_size=0.2):
        """
        Train the classifier with error handling
        """
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
                n_jobs=-1  # Use all available cores
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
        """
        Save the trained model with error handling
        """
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
        """
        Load a trained model with error handling
        """
        try:
            with open(filepath, 'rb') as f:
                self.classifier = pickle.load(f)
            print(f"Model successfully loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_xml_error(self, new_xml_path):
        """
        Predict error probability for a new XML file
        """
        if not self.classifier:
            raise ValueError("Model not trained. Call train_classifier() first.")
        
        try:
            # Process single file
            features, _ = process_single_file((new_xml_path, self.project_id, False))
            
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
    
    # Train or load model
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        classifier.load_model(MODEL_PATH)
    else:
        print("Training new model...")
        if classifier.train_classifier():
            classifier.save_model(MODEL_PATH)
    
    # Example prediction
    test_file = "path/to/test/file.xml"
    if os.path.exists(test_file):
        prediction = classifier.predict_xml_error(test_file)
        if prediction:
            print("\nPrediction Results:")
            print(f"File: {prediction['filename']}")
            print(f"Error Predicted: {prediction['is_error']}")
            print(f"Error Probability: {prediction['error_probability']:.2%}")

if __name__ == "__main__":
    main()
