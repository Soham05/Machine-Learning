import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xml.etree.ElementTree as ET
from collections import defaultdict
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

class MemoryEfficientXMLProcessor:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.tfidf = TfidfVectorizer(max_features=50)  # Reduced features
        
    def extract_features_batch(self, xml_files):
        """Process a batch of XML files and extract features"""
        features_list = []
        
        for xml_file in xml_files:
            try:
                # Read file content
                with open(xml_file, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
                
                # Extract basic features
                features = self.extract_basic_features(xml_content)
                if features:
                    features['filename'] = xml_file.name
                    features_list.append(features)
                
                # Clear content from memory
                del xml_content
                
            except Exception as e:
                print(f"Error processing {xml_file}: {str(e)}")
                continue
                
        # Force garbage collection
        gc.collect()
        return features_list
    
    def extract_basic_features(self, xml_content):
        """Extract essential features from XML content"""
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
            
            return features
        except ET.ParseError:
            return None
    
    def get_max_depth(self, element, current_depth=0):
        """Calculate maximum depth of XML tree"""
        if len(element) == 0:
            return current_depth
        return max(self.get_max_depth(child, current_depth + 1) for child in element)
    
    def extract_crd(self, root):
        """Extract CRD/CRDID value"""
        for tag in ['CRD', 'CRDID', './/CRD', './/CRDID']:
            element = root.find(tag)
            if element is not None and element.text:
                return element.text.strip()
        return None

class XMLDataPreparation:
    def __init__(self, error_crd_file, batch_size=100):
        """Initialize with error CRDs file"""
        self.error_crds = pd.read_csv(error_crd_file)
        self.error_crd_set = set(self.error_crds['filename'].str.strip())
        self.processor = MemoryEfficientXMLProcessor(batch_size)
        self.batch_size = batch_size
    
    def prepare_dataset(self, xml_directory):
        """Prepare dataset in batches"""
        xml_files = list(Path(xml_directory).glob('*.xml'))
        total_files = len(xml_files)
        
        print(f"Processing {total_files} XML files in batches of {self.batch_size}...")
        
        all_features = []
        file_mapping = []
        
        # Process files in batches
        for i in tqdm(range(0, total_files, self.batch_size)):
            batch_files = xml_files[i:i + self.batch_size]
            
            # Process batch
            batch_features = self.processor.extract_features_batch(batch_files)
            
            # Add to results
            all_features.extend(batch_features)
            
            # Clear memory
            gc.collect()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Create labels
        if 'crd' in features_df.columns:
            features_df['is_error'] = features_df['crd'].isin(self.error_crd_set)
        
        return features_df

class XMLErrorPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
    
    def prepare_features(self, features_df):
        """Prepare features for model training"""
        # Select numeric features
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_features]
        
        # Fill missing values
        X = X.fillna(0)
        
        return X
    
    def train(self, features_df):
        """Train the model"""
        X = self.prepare_features(features_df)
        y = features_df['is_error'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
    
    def predict(self, features_df):
        """Make predictions"""
        X = self.prepare_features(features_df)
        return self.model.predict_proba(X)[:, 1]

def main(xml_directory, error_crd_file, batch_size=100):
    """Main function with memory-efficient processing"""
    try:
        # Initialize data preparation
        print("Initializing data preparation...")
        data_prep = XMLDataPreparation(error_crd_file, batch_size=batch_size)
        
        # Process XML files and extract features
        print("Processing XML files...")
        features_df = data_prep.prepare_dataset(xml_directory)
        
        # Print validation statistics
        print("\nValidation Statistics:")
        print(f"Total files processed: {len(features_df)}")
        print(f"Error files found: {features_df['is_error'].sum()}")
        print(f"Error rate: {(features_df['is_error'].sum() / len(features_df)) * 100:.2f}%")
        
        # Train model
        print("\nTraining model...")
        predictor = XMLErrorPredictor()
        results = predictor.train(features_df)
        
        print("\nModel Results:")
        print(f"Training Score: {results['train_score']:.3f}")
        print(f"Testing Score: {results['test_score']:.3f}")
        
        # Save features DataFrame to disk to free memory
        features_df.to_pickle('processed_features.pkl')
        
        return predictor, results
        
    except Exception as e:
        print(f"Error in main pipeline: {str(e)}")
        return None, None

# Example usage
if __name__ == "__main__":
    xml_dir = "path/to/xml/files"
    error_crd_file = "path/to/error_crds.csv"
    
    predictor, results = main(xml_dir, error_crd_file, batch_size=50)
