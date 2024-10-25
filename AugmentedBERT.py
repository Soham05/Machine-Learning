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
import copy
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
                modified_root = ET.fromstring(xml_content)  # Create new root
                modified = self.modify_attributes(modified_root)
                augmented_versions.append(ET.tostring(modified, encoding='unicode'))
            
            # Version 2: Swap siblings
            if np.random.random() < self.error_probability:
                modified_root = ET.fromstring(xml_content)  # Create new root
                modified = self.swap_siblings(modified_root)
                augmented_versions.append(ET.tostring(modified, encoding='unicode'))
            
            # Version 3: Remove random tags
            if np.random.random() < self.error_probability:
                modified_root = ET.fromstring(xml_content)  # Create new root
                modified = self.remove_random_tags(modified_root)
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
            children = list(elem)
            if len(children) >= 2 and np.random.random() < self.error_probability:
                idx1, idx2 = np.random.choice(len(children), 2, replace=False)
                # Remove and insert to swap
                child1, child2 = children[idx1], children[idx2]
                elem.remove(child1)
                elem.remove(child2)
                if idx1 < idx2:
                    elem.insert(idx1, child2)
                    elem.insert(idx2, child1)
                else:
                    elem.insert(idx2, child1)
                    elem.insert(idx1, child2)
        return element
    
    def remove_random_tags(self, element):
        """Remove random tags while preserving content"""
        for elem in element.iter():
            children = list(elem)
            if children and np.random.random() < self.error_probability:
                child_idx = np.random.randint(0, len(children))
                child = children[child_idx]
                elem.remove(child)
                # Insert all grandchildren where the child was
                for i, grandchild in enumerate(list(child)):
                    elem.insert(child_idx + i, grandchild)
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
###################### testing ################################

def prepare_test_data(xml_directory, error_crd_file, batch_size=100):
    """Prepare test data using existing pipeline"""
    print("Initializing test data preparation...")
    data_prep = XMLDataPreparation(error_crd_file, batch_size=batch_size)
    
    print("Processing test XML files...")
    features_df = data_prep.prepare_dataset(xml_directory)
    
    print("\nTest Dataset Statistics:")
    print(f"Total files processed: {len(features_df)}")
    if 'is_error' in features_df.columns:
        print(f"Known error files: {features_df['is_error'].sum()}")
        print(f"Error rate: {(features_df['is_error'].sum() / len(features_df)) * 100:.2f}%")
    
    return features_df

def test_new_xmls(trained_predictor, features_df, output_file='predictions.csv', crd_output_file='crd_predictions.csv'):
    """Make predictions on new data and save results with CRD comparison"""
    # Get predictions
    predictions_prob = trained_predictor.predict(features_df)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'filename': features_df['filename'],
        'crd': features_df['crd'],
        'prediction_probability': predictions_prob,
        'predicted_error': predictions_prob > 0.5
    })
    
    if 'is_error' in features_df.columns:
        results_df['actual_error'] = features_df['is_error']
        
        # Calculate metrics if we have actual labels
        y_true = features_df['is_error']
        y_pred = predictions_prob > 0.5
        
        print("\nTest Set Metrics:")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
        print(f"Recall: {recall_score(y_true, y_pred):.3f}")
        print(f"Precision: {precision_score(y_true, y_pred):.3f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Create CRD comparison dataframe
        crd_comparison_df = results_df[['crd', 'actual_error', 'predicted_error', 'prediction_probability']]
        crd_comparison_df = crd_comparison_df.sort_values('prediction_probability', ascending=False)
        
        # Add prediction status
        crd_comparison_df['prediction_status'] = 'Correct'
        crd_comparison_df.loc[crd_comparison_df['actual_error'] != crd_comparison_df['predicted_error'], 
                            'prediction_status'] = 'Incorrect'
        
        # Save CRD comparisons
        crd_comparison_df.to_csv(crd_output_file, index=False)
        print(f"\nCRD predictions saved to {crd_output_file}")
        
        # Print summary of predictions
        print("\nPrediction Summary by CRD:")
        print("True Positives (Correctly identified errors):")
        tp_df = crd_comparison_df[(crd_comparison_df['actual_error'] == True) & 
                                (crd_comparison_df['predicted_error'] == True)]
        print(tp_df[['crd', 'prediction_probability']].to_string())
        
        print("\nFalse Negatives (Missed errors):")
        fn_df = crd_comparison_df[(crd_comparison_df['actual_error'] == True) & 
                                (crd_comparison_df['predicted_error'] == False)]
        print(fn_df[['crd', 'prediction_probability']].to_string())
        
        print("\nFalse Positives (False alarms):")
        fp_df = crd_comparison_df[(crd_comparison_df['actual_error'] == False) & 
                                (crd_comparison_df['predicted_error'] == True)]
        print(fp_df[['crd', 'prediction_probability']].to_string())
    
    # Sort by prediction probability in descending order
    results_df = results_df.sort_values('prediction_probability', ascending=False)
    
    # Save full predictions
    results_df.to_csv(output_file, index=False)
    print(f"\nFull predictions saved to {output_file}")
    
    return results_df, crd_comparison_df if 'is_error' in features_df.columns else None

# Set paths for your new data
new_xml_dir = "path/to/new/xml/files"
new_error_crd_file = "path/to/new/error_crds.csv"

# Prepare the test data
test_features_df = prepare_test_data(new_xml_dir, new_error_crd_file)

# Make predictions and get results
results_df = test_new_xmls(predictor, test_features_df, output_file='new_predictions.csv')
# View top predicted errors
print("\nTop 10 Most Likely Errors:")
print(results_df.head(10)[['filename', 'prediction_probability']])

# If you have actual labels, view confusion matrix
if 'actual_error' in results_df.columns:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(results_df['actual_error'], results_df['predicted_error'])
    print("\nConfusion Matrix:")
    print(cm)
