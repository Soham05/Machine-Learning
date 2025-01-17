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

# Vertex AI imports
import vertexai
from vertexai.language_models import TextEmbeddingModel, GenerativeModel
from google.cloud import aiplatform

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
    Process a single XML file
    """
    xml_file, project_id, is_error = args
    
    try:
        # Extract traditional features
        trad_features = extract_traditional_features(xml_file)
        
        # Initialize Vertex AI and models for this process
        vertexai.init(project=project_id)
        embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
        generative_model = GenerativeModel("gemini-pro")
        
        # Read XML content
        with open(xml_file, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # Generate embeddings
        truncated_content = xml_content[:2000]
        embeddings = embedding_model.get_embeddings([truncated_content])
        embedding_vector = embeddings[0].values
        
        # Calculate semantic features
        semantic_features = [
            float(np.mean(embedding_vector)),
            float(np.std(embedding_vector)),
            float(np.max(embedding_vector)),
            float(np.min(embedding_vector)),
            len(embedding_vector),
            0.5,  # default structural_complexity
            0.5   # default semantic_error_probability
        ]
        
        # Combine features
        features = trad_features + semantic_features
        
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
        
        # Prepare arguments for parallel processing
        process_args = [
            (xml_file, self.project_id, os.path.basename(xml_file) in all_error_files)
            for xml_file in xml_files
        ]
        
        features = []
        labels = []
        
        # Process files in parallel
        num_cores = multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Process with progress bar
            with tqdm(total=len(process_args), desc="Processing XML files") as pbar:
                for feature, label in executor.map(process_single_file, process_args):
                    if feature is not None and label is not None:
                        features.append(feature)
                        labels.append(label)
                    pbar.update(1)
        
        return np.array(features), np.array(labels)
    
    def train_classifier(self, test_size=0.2):
        """
        Train the classifier
        """
        X, y = self.prepare_dataset()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Handle imbalanced dataset
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Train Random Forest
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=5
        )
        
        self.classifier.fit(X_resampled, y_resampled)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if not self.classifier:
            raise ValueError("Model not trained. Call train_classifier() first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.classifier, f)
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        with open(filepath, 'rb') as f:
            self.classifier = pickle.load(f)
    
    def predict_xml_error(self, new_xml_path):
        """
        Predict error probability for a new XML file
        """
        if not self.classifier:
            raise ValueError("Model not trained. Call train_classifier() first.")
        
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
            "error_probability": probability[0][1]
        }

# Usage example
def main():
    PROJECT_ID = "your-gcp-project-id"
    Q1_FOLDER = "XMLTesting/files/Q1-XMLs/"
    Q2_FOLDER = "XMLTesting/files/Q2-XMLs/"
    Q1_ERROR_CSV = "XMLTesting/files/Q12024_error_xml_filenames.csv"
    Q2_ERROR_CSV = "XMLTesting/files/Q22024_error_xml_filenames.csv"
    
    classifier = XMLErrorClassifier(
        PROJECT_ID, 
        Q1_FOLDER, 
        Q2_FOLDER, 
        Q1_ERROR_CSV, 
        Q2_ERROR_CSV
    )
    
    # Train and save
    classifier.train_classifier()
    classifier.save_model('xml_classifier_model.pkl')
    
    # Later, load and predict
    classifier.load_model('xml_classifier_model.pkl')
    prediction = classifier.predict_xml_error("path/to/new/xml/file.xml")
    print(prediction)

if __name__ == "__main__":
    main()
