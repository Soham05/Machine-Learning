import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
import xml.etree.ElementTree as ET
from pathlib import Path
import gc
from tqdm.notebook import tqdm
import warnings
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

class XMLAugmenter:
    def __init__(self, error_probability=0.3):
        self.error_probability = error_probability
        
    def augment_xml(self, xml_content):
        """Create augmented versions of error XMLs"""
        try:
            root = ET.fromstring(xml_content)
            augmented_versions = []
            
            # Original version
            augmented_versions.append(xml_content)
            
            # Version 1: Modify attributes
            if np.random.random() < self.error_probability:
                modified = self.modify_attributes(root.copy())
                augmented_versions.append(ET.tostring(modified, encoding='unicode'))
            
            # Version 2: Swap siblings
            if np.random.random() < self.error_probability:
                modified = self.swap_siblings(root.copy())
                augmented_versions.append(ET.tostring(modified, encoding='unicode'))
            
            # Version 3: Remove random tags
            if np.random.random() < self.error_probability:
                modified = self.remove_random_tags(root.copy())
                augmented_versions.append(ET.tostring(modified, encoding='unicode'))
            
            return augmented_versions
            
        except ET.ParseError:
            return [xml_content]
    
    def modify_attributes(self, element):
        """Modify random attributes to create errors"""
        for elem in element.iter():
            if elem.attrib and np.random.random() < self.error_probability:
                attr_key = np.random.choice(list(elem.attrib.keys()))
                elem.attrib[attr_key] = elem.attrib[attr_key] + '_modified'
        return element
    
    def swap_siblings(self, element):
        """Swap random sibling elements"""
        for elem in element.iter():
            if len(elem) >= 2 and np.random.random() < self.error_probability:
                idx1, idx2 = np.random.choice(len(elem), 2, replace=False)
                elem[idx1], elem[idx2] = elem[idx2], elem[idx1]
        return element
    
    def remove_random_tags(self, element):
        """Remove random tags while preserving content"""
        for elem in element.iter():
            if len(elem) > 0 and np.random.random() < self.error_probability:
                child_idx = np.random.randint(0, len(elem))
                child = elem[child_idx]
                elem.remove(child)
                for grandchild in child:
                    elem.append(grandchild)
        return element

class BERTXMLProcessor:
    def __init__(self, batch_size=100, max_length=512):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.1)
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
    def extract_features_batch(self, xml_files):
        """Process a batch of XML files and extract features with BERT"""
        features_list = []
        
        for xml_file in xml_files:
            try:
                with open(xml_file, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
                
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
    
    def get_bert_embedding(self, text):
        """Get BERT embedding for text with dropout"""
        inputs = self.tokenizer(text, 
                              return_tensors="pt",
                              truncation=True,
                              max_length=self.max_length,
                              padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Apply dropout to the CLS token embedding
            embeddings = self.dropout(outputs.last_hidden_state[:, 0, :]).cpu().numpy()
        
        return embeddings[0]
    
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
            
            # Get BERT embeddings
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
        self.augmenter = XMLAugmenter(error_probability=0.3)
    
    def prepare_dataset(self, xml_directory):
        xml_files = list(Path(xml_directory).glob('*.xml'))
        total_files = len(xml_files)
        
        print(f"Processing {total_files} XML files in batches of {self.batch_size}...")
        
        all_features = []
        
        for i in tqdm(range(0, total_files, self.batch_size)):
            batch_files = xml_files[i:i + self.batch_size]
            batch_features = self.processor.extract_features_batch(batch_files)
            
            # Augment error examples
            augmented_features = []
            for features in batch_features:
                if features['crd'] in self.error_crd_set:
                    xml_file = batch_files[batch_features.index(features)]
                    with open(xml_file, 'r', encoding='utf-8') as f:
                        xml_content = f.read()
                    augmented_versions = self.augmenter.augment_xml(xml_content)
                    
                    for aug_xml in augmented_versions[1:]:
                        aug_features = self.processor.extract_features(aug_xml)
                        if aug_features:
                            aug_features['filename'] = f"aug_{features['filename']}"
                            augmented_features.append(aug_features)
            
            all_features.extend(batch_features)
            all_features.extend(augmented_features)
            gc.collect()
        
        features_df = pd.DataFrame(all_features)
        
        if 'crd' in features_df.columns:
            features_df['is_error'] = features_df['crd'].isin(self.error_crd_set)
        
        return features_df

class XMLErrorPredictor:
    def __init__(self, class_weight=10, n_splits=5):
        self.n_splits = n_splits
        weights = {0: 1, 1: class_weight}
        self.model = RandomForestClassifier(
            n_estimators=200,
            class_weight=weights,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    def prepare_features(self, features_df):
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_features]
        X = X.fillna(0)
        return X
    
    def cross_validate(self, features_df):
        """Perform cross-validation and return detailed metrics"""
        X = self.prepare_features(features_df)
        y = features_df['is_error'].astype(int)
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        fold_metrics = []
        feature_importance_avg = np.zeros(X.shape[1])
        
        print(f"\nPerforming {self.n_splits}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            
            metrics = {
                'fold': fold,
                'accuracy': self.model.score(X_val, y_val),
                'recall': recall_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'support_errors': sum(y_val == 1),
                'total_samples': len(y_val)
            }
            
            feature_importance_avg += self.model.feature_importances_
            
            fold_metrics.append(metrics)
            
            print(f"\nFold {fold} Results:")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"F1 Score: {metrics['f1']:.3f}")
            print(f"Error samples in validation: {metrics['support_errors']}/{metrics['total_samples']}")
        
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
            'recall': np.mean([m['recall'] for m in fold_metrics]),
            'precision': np.mean([m['precision'] for m in fold_metrics]),
            'f1': np.mean([m['f1'] for m in fold_metrics]),
            'std_accuracy': np.std([m['accuracy'] for m in fold_metrics]),
            'std_recall': np.std([m['recall'] for m in fold_metrics]),
            'feature_importance': dict(zip(X.columns, feature_importance_avg / self.n_splits))
        }
        
        print("\nCross-validation Summary:")
        print(f"Average Accuracy: {avg_metrics['accuracy']:.3f} (±{avg_metrics['std_accuracy']:.3f})")
        print(f"Average Recall: {avg_metrics['recall']:.3f} (±{avg_metrics['std_recall']:.3f})")
        print(f"Average Precision: {avg_metrics['precision']:.3f}")
        print(f"Average F1 Score: {avg_metrics['f1']:.3f}")
        
        return avg_metrics, fold_metrics
    
    def train(self, features_df):
        """Train final model on entire dataset after cross-validation"""
        X = self.prepare_features(features_df)
        y = features_df['is_error'].astype(int)
        
        # First perform cross-validation
        avg_metrics, fold_metrics = self.cross_validate(features_df)
        
        # Train final model on all data
        self.model.fit(X, y)
        
        return {
            'cross_validation_metrics': avg_metrics,
            'fold_metrics': fold_metrics,
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
    
    def predict(self, features_df):
        """Make predictions on new data"""
        X = self.prepare_features(features_df)
        return self.model.predict_proba(X)[:, 1]

def main(xml_directory, error_crd_file, batch_size=100, class_weight=10, n_splits=5):
    try:
        print("Initializing data preparation...")
        data_prep = XMLDataPreparation(error_crd_file, batch_size=batch_size)
        
        print("Processing XML files...")
        features_df = data_prep.prepare_dataset(xml_directory)
        
        print("\nDataset Statistics:")
        print(f"Total files processed: {len(features_df)}")
        print(f"Error files found: {features_df['is_error'].sum()}")
        print(f"Error rate: {(features_df['is_error'].sum() / len(features_df)) * 100:.2f}%")
        
        print("\nTraining model with cross-validation...")
        predictor = XMLErrorPredictor(class_weight=class_weight, n_splits=n_splits)
        results = predictor.train(features_df)
        
        # Save features and model
        features_df.to_pickle('processed_features.pkl')
        
        return predictor, results
        
    except Exception as e:
        print(f"Error in main pipeline: {str(e)}")
        return None, None

if __name__ == "__main__":
    xml_dir = "path/to/xml/files"
    error_crd_file = "path/to/error_crds.csv"
    
    predictor, results = main(xml_dir, error_crd_file, 
                            batch_size=50, 
                            class_weight=10, 
                            n_splits=5)

    if results:
        print("\nFeature Importance:")
        importance_dict = results['cross_validation_metrics']['feature_importance']
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1],
