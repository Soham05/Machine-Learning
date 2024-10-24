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
    def __init__(self):
        self.feature_extractor = XMLFeatureExtractor()
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
    
    # ... [rest of the XMLErrorPredictor class remains the same as in previous artifact]

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
