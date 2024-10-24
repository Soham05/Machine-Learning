import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report
import xml.etree.ElementTree as ET
from pathlib import Path
import gc
from tqdm.notebook import tqdm
import warnings
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

class BERTXMLProcessor:
    def __init__(self, batch_size=100, max_length=512):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def get_bert_embedding(self, text):
        """Get BERT embedding for text"""
        inputs = self.tokenizer(text, 
                              return_tensors="pt",
                              truncation=True,
                              max_length=self.max_length,
                              padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings[0]
    
    def extract_features_batch(self, xml_files):
        """Process a batch of XML files and extract features with BERT"""
        features_list = []
        
        for xml_file in xml_files:
            try:
                with open(xml_file, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
                
                # Extract basic features
                features = self.extract_features(xml_content)
                if features:
                    features['filename'] = xml_file.name
                    features_list.append(features)
                
                del xml_content
                
            except Exception as e:
                print(f"Error processing {xml_file}: {str(e)}")
                continue
                
        gc.collect()
        return features_list
    
    def extract_features(self, xml_content):
        """Extract features including BERT embeddings"""
        try:
            root = ET.fromstring(xml_content)
            features = {}
            
            # Basic structure features
            features['total_tags'] = len(root.findall('.//*'))
            features['max_depth'] = self.get_max_depth(root)
            features['total_attributes'] = sum(len(elem.attrib) for elem in root.iter())
            
            # Get CRD/CRDID
            crd = self.extract_crd(root)
            if crd:
                features['crd'] = crd
            
            # Get BERT embeddings for XML content
            # Convert XML to string representation for BERT
            xml_str = ET.tostring(root, encoding='unicode')
            bert_embedding = self.get_bert_embedding(xml_str)
            
            # Add BERT embeddings to features
            for i, value in enumerate(bert_embedding):
                features[f'bert_{i}'] = value
            
            return features
            
        except ET.ParseError:
            return None
    
    def get_max_depth(self, element, current_depth=0):
        if len(element) == 0:
            return current_depth
        return max(self.get_max_depth(child, current_depth + 1) for child in element)
    
    def extract_crd(self, root):
        for tag in ['CRD', 'CRDID', './/CRD', './/CRDID']:
            element = root.find(tag)
            if element is not None and element.text:
                return element.text.strip()
        return None

class XMLDataPreparation:
    def __init__(self, error_crd_file, batch_size=100):
        self.error_crds = pd.read_csv(error_crd_file)
        self.error_crd_set = set(self.error_crds['filename'].str.strip())
        self.processor = BERTXMLProcessor(batch_size)
        self.batch_size = batch_size
    
    def prepare_dataset(self, xml_directory):
        xml_files = list(Path(xml_directory).glob('*.xml'))
        total_files = len(xml_files)
        
        print(f"Processing {total_files} XML files in batches of {self.batch_size}...")
        
        all_features = []
        
        for i in tqdm(range(0, total_files, self.batch_size)):
            batch_files = xml_files[i:i + self.batch_size]
            batch_features = self.processor.extract_features_batch(batch_files)
            all_features.extend(batch_features)
            gc.collect()
        
        features_df = pd.DataFrame(all_features)
        
        if 'crd' in features_df.columns:
            features_df['is_error'] = features_df['crd'].isin(self.error_crd_set)
        
        return features_df

class XMLErrorPredictor:
    def __init__(self, class_weight=10):
        # Increased weight for error class to improve recall
        weights = {0: 1, 1: class_weight}
        self.model = RandomForestClassifier(
            n_estimators=200,  # Increased number of trees
            class_weight=weights,
            random_state=42,
            n_jobs=-1
        )
    
    def prepare_features(self, features_df):
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_features]
        X = X.fillna(0)
        return X
    
    def train(self, features_df):
        X = self.prepare_features(features_df)
        y = features_df['is_error'].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        results = {
            'train_score': self.model.score(X_train, y_train),
            'test_score': self.model.score(X_test, y_test),
            'train_recall': recall_score(y_train, y_pred_train),
            'test_recall': recall_score(y_test, y_pred_test),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_)),
            'classification_report': classification_report(y_test, y_pred_test)
        }
        
        return results
    
    def predict(self, features_df):
        X = self.prepare_features(features_df)
        return self.model.predict_proba(X)[:, 1]

def main(xml_directory, error_crd_file, batch_size=100, class_weight=10):
    try:
        print("Initializing data preparation...")
        data_prep = XMLDataPreparation(error_crd_file, batch_size=batch_size)
        
        print("Processing XML files...")
        features_df = data_prep.prepare_dataset(xml_directory)
        
        print("\nValidation Statistics:")
        print(f"Total files processed: {len(features_df)}")
        print(f"Error files found: {features_df['is_error'].sum()}")
        print(f"Error rate: {(features_df['is_error'].sum() / len(features_df)) * 100:.2f}%")
        
        print("\nTraining model...")
        predictor = XMLErrorPredictor(class_weight=class_weight)
        results = predictor.train(features_df)
        
        print("\nModel Results:")
        print(f"Training Score: {results['train_score']:.3f}")
        print(f"Testing Score: {results['test_score']:.3f}")
        print(f"Training Recall: {results['train_recall']:.3f}")
        print(f"Testing Recall: {results['test_recall']:.3f}")
        print("\nDetailed Classification Report:")
        print(results['classification_report'])
        
        features_df.to_pickle('processed_features.pkl')
        
        return predictor, results
        
    except Exception as e:
        print(f"Error in main pipeline: {str(e)}")
        return None, None

if __name__ == "__main__":
    xml_dir = "path/to/xml/files"
    error_crd_file = "path/to/error_crds.csv"
    
    predictor, results = main(xml_dir, error_crd_file, batch_size=50, class_weight=10)
