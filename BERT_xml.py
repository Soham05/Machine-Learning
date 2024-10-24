import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch import nn
import xml.etree.ElementTree as ET
from collections import defaultdict
import os
from pathlib import Path

class XMLDataPreparation:
    def __init__(self, error_crd_file):
        """
        Initialize with the CSV file containing error CRD/CRDIDs
        
        Args:
            error_crd_file (str): Path to CSV file with error CRDs
        """
        self.error_crds = pd.read_csv(error_crd_file)
        self.error_crd_set = set(self.error_crds['filename'].str.strip())
    
    def extract_crd_from_xml(self, xml_string):
        """Extract CRD/CRDID from XML content"""
        try:
            root = ET.fromstring(xml_string)
            # Search for CRD tag - adjust xpath based on actual XML structure
            crd_element = root.find('.//CRD')
            if crd_element is not None:
                return crd_element.text.strip()
            crd_id_element = root.find('.//CRDID')
            if crd_id_element is not None:
                return crd_id_element.text.strip()
            return None
        except ET.ParseError:
            return None

    def prepare_dataset(self, xml_directory):
        """
        Prepare dataset by matching XMLs with error CRDs
        
        Args:
            xml_directory (str): Directory containing XML files
            
        Returns:
            tuple: (xml_contents, labels, file_mapping)
        """
        xml_contents = []
        labels = []
        file_mapping = []
        
        # Process each XML file in the directory
        for xml_file in Path(xml_directory).glob('*.xml'):
            try:
                with open(xml_file, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
                
                crd = self.extract_crd_from_xml(xml_content)
                if crd:
                    xml_contents.append(xml_content)
                    # Check if this CRD is in our error list
                    is_error = crd in self.error_crd_set
                    labels.append(1 if is_error else 0)
                    file_mapping.append({
                        'filename': xml_file.name,
                        'crd': crd,
                        'is_error': is_error
                    })
            except Exception as e:
                print(f"Error processing {xml_file}: {str(e)}")
                continue
        
        return np.array(xml_contents), np.array(labels), pd.DataFrame(file_mapping)

    def validate_data_preparation(self, file_mapping):
        """
        Validate the data preparation process
        
        Args:
            file_mapping (pd.DataFrame): DataFrame with file mapping information
            
        Returns:
            dict: Validation statistics
        """
        total_error_crds = len(self.error_crd_set)
        matched_error_crds = sum(file_mapping['is_error'])
        
        validation_stats = {
            'total_files_processed': len(file_mapping),
            'total_error_crds': total_error_crds,
            'matched_error_crds': matched_error_crds,
            'unmatched_error_crds': total_error_crds - matched_error_crds,
            'error_rate': matched_error_crds / len(file_mapping),
            'missing_crds': list(self.error_crd_set - set(file_mapping[file_mapping['is_error']]['crd']))
        }
        
        return validation_stats

class XMLErrorPredictor:
    class XMLErrorPredictor:
    def __init__(self):
        self.feature_extractor = XMLFeatureExtractor()
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        
    def find_common_tags(self, xml_files):
        """Identify common tags across all XML files"""
        tag_counts = defaultdict(int)
        total_files = len(xml_files)
        
        for xml in xml_files:
            structure = self.feature_extractor.extract_tag_structure(xml)
            if structure:
                unique_tags = set(tag.split('/')[-1] for tag in structure['paths'])
                for tag in unique_tags:
                    tag_counts[tag] += 1
        
        # Consider tags present in at least 80% of files as common
        self.feature_extractor.common_tags = {
            tag for tag, count in tag_counts.items()
            if count >= 0.8 * total_files
        }
    
    def prepare_dataset(self, xml_files, labels):
        """Convert XML files to feature matrices"""
        features_list = []
        valid_indices = []
        
        for i, xml in enumerate(xml_files):
            features = self.feature_extractor.process_xml_file(xml)
            if features:
                features_list.append(features)
                valid_indices.append(i)
        
        X = pd.DataFrame(features_list)
        y = labels[valid_indices]
        return X, y
    
    def train(self, train_xml_files, train_labels):
        """Train the model"""
        # Find common tags first
        self.find_common_tags(train_xml_files)
        
        # Prepare training data
        X, y = self.prepare_dataset(train_xml_files, train_labels)
        
        # Train the model
        self.model.fit(X, y)
    
    def predict(self, xml_file):
        """Predict if XML needs testing"""
        features = self.feature_extractor.process_xml_file(xml_file)
        if not features:
            return True  # If can't process XML, better to test it
            
        features_df = pd.DataFrame([features])
        probability = self.model.predict_proba(features_df)[0][1]
        return probability

def evaluate_model(predictor, test_xml_files, test_labels, threshold=0.5):
    """Evaluate model performance"""
    predictions = []
    for xml in test_xml_files:
        prob = predictor.predict(xml)
        predictions.append(prob > threshold)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    tp = np.sum((predictions == 1) & (test_labels == 1))
    fp = np.sum((predictions == 1) & (test_labels == 0))
    tn = np.sum((predictions == 0) & (test_labels == 0))
    fn = np.sum((predictions == 0) & (test_labels == 1))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_negatives': fn  # Important as these are errors we failed to catch
    }
class XMLFeatureExtractor:
    def __init__(self):
        # Initialize BERT for semantic understanding
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.common_tags = set()
        
    def extract_tag_structure(self, xml_string):
        """Extract hierarchical structure and content from XML"""
        try:
            root = ET.fromstring(xml_string)
            structure = defaultdict(list)
            
            def traverse(element, path=""):
                current_path = f"{path}/{element.tag}"
                # Store tag content and attributes
                structure['paths'].append(current_path)
                structure['contents'].append(element.text)
                structure['attributes'].append(dict(element.attrib))
                
                for child in element:
                    traverse(child, current_path)
                    
            traverse(root)
            return structure
        except ET.ParseError:
            return None

    def get_bert_embeddings(self, text):
        """Get BERT embeddings for text content"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def process_xml_file(self, xml_file):
        """Process single XML file and extract features"""
        structure = self.extract_tag_structure(xml_file)
        if not structure:
            return None
        
        features = {}
        
        # Structural features
        features['num_tags'] = len(structure['paths'])
        features['max_depth'] = max(path.count('/') for path in structure['paths'])
        features['num_attributes'] = sum(len(attrs) for attrs in structure['attributes'])
        
        # Content features
        all_text = ' '.join(filter(None, structure['contents']))
        if all_text.strip():
            bert_embedding = self.get_bert_embeddings(all_text)
            for i, val in enumerate(bert_embedding[0]):
                features[f'bert_embedding_{i}'] = val
        
        # Tag presence features
        for tag in self.common_tags:
            features[f'has_tag_{tag}'] = any(tag in path for path in structure['paths'])
            
        return features   

def main(xml_directory, error_crd_file):
    """
    Main function to run the entire pipeline
    
    Args:
        xml_directory (str): Directory containing XML files
        error_crd_file (str): Path to CSV file with error CRDs
    """
    # Initialize data preparation
    data_prep = XMLDataPreparation(error_crd_file)
    
    # Prepare dataset
    print("Preparing dataset...")
    xml_contents, labels, file_mapping = data_prep.prepare_dataset(xml_directory)
    
    # Validate data preparation
    print("Validating data preparation...")
    validation_stats = data_prep.validate_data_preparation(file_mapping)
    print("\nValidation Statistics:")
    for key, value in validation_stats.items():
        print(f"{key}: {value}")
    
    # Check if we have sufficient matched data
    if validation_stats['matched_error_crds'] < validation_stats['total_error_crds'] * 0.9:
        print("\nWARNING: More than 10% of error CRDs could not be matched to XML files!")
        print("Please verify the CRD extraction logic and file paths.")
    
    # Initialize and train model
    predictor = XMLErrorPredictor()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        xml_contents, labels, 
        test_size=0.2, 
        stratify=labels,
        random_state=42
    )
    
    # Train model
    print("\nTraining model...")
    predictor.train(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation_results = evaluate_model(predictor, X_test, y_test)
    print("\nModel Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")
    
    return predictor, file_mapping, validation_stats

# Example usage:
if __name__ == "__main__":
    xml_dir = "path/to/xml/files"
    error_crd_file = "path/to/error_crds.csv"
    predictor, file_mapping, stats = main(xml_dir, error_crd_file)
