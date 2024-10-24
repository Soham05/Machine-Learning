import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, confusion_matrix
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from tqdm.notebook import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

class XMLFeatureExtractor:
    def __init__(self, use_bert=False, batch_size=50):
        self.use_bert = use_bert
        self.batch_size = batch_size
        
        if use_bert:
            try:
                from transformers import DistilBertTokenizer, DistilBertModel
                import torch
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
                print("BERT initialized successfully")
            except ImportError:
                print("BERT modules not available. Falling back to TF-IDF.")
                self.use_bert = False
                
        # Initialize TF-IDF as backup
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(max_features=100)
    
    def get_text_embeddings(self, text):
        """Get embeddings using either BERT or TF-IDF"""
        if not text or text.isspace():
            return np.zeros(100)
            
        if self.use_bert:
            try:
                import torch
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).numpy()[0]
            except Exception as e:
                print(f"BERT processing error: {e}. Falling back to TF-IDF.")
                self.use_bert = False
        
        # TF-IDF fallback
        return self.tfidf.fit_transform([text]).toarray()[0]

    def extract_features(self, xml_content):
        """Extract comprehensive features from XML"""
        try:
            root = ET.fromstring(xml_content)
            features = {}
            
            # Structural features
            features['total_tags'] = len(root.findall('.//*'))
            features['max_depth'] = self.get_max_depth(root)
            features['total_attributes'] = sum(len(elem.attrib) for elem in root.iter())
            
            # Content features
            all_text = ' '.join(
                elem.text for elem in root.iter() 
                if elem.text and not elem.text.isspace()
            )
            
            # Get text embeddings
            embeddings = self.get_text_embeddings(all_text)
            for i, val in enumerate(embeddings):
                features[f'text_embedding_{i}'] = val
            
            # Tag frequency features
            tag_counts = defaultdict(int)
            for elem in root.iter():
                tag_counts[elem.tag] += 1
            
            # Add top 20 most common tags as features
            top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            for tag, count in top_tags:
                features[f'tag_{tag}_count'] = count
            
            return features
            
        except ET.ParseError:
            return None
    
    def get_max_depth(self, element, current_depth=0):
        if len(element) == 0:
            return current_depth
        return max(self.get_max_depth(child, current_depth + 1) for child in element)

class HighRecallXMLPredictor:
    def __init__(self, use_bert=False, recall_weight=2.0):
        """
        Initialize predictor with focus on recall
        
        Args:
            use_bert (bool): Whether to use BERT embeddings
            recall_weight (float): Weight for false negatives (higher means more focus on recall)
        """
        self.feature_extractor = XMLFeatureExtractor(use_bert=use_bert)
        
        # Use RandomForest with class weights focused on recall
        self.model = RandomForestClassifier(
            n_estimators=200,  # Increased from 100
            class_weight={0: 1.0, 1: recall_weight},  # Weight errors more heavily
            random_state=42,
            n_jobs=-1
        )
    
    def prepare_features(self, features_df):
        """Prepare features for model"""
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_features]
        return X.fillna(0)
    
    def train_and_evaluate(self, features_df, threshold=0.3):
        """Train model and evaluate with focus on recall"""
        X = self.prepare_features(features_df)
        y = features_df['is_error'].astype(int)
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Get probabilities
        train_probs = self.model.predict_proba(X_train)[:, 1]
        test_probs = self.model.predict_proba(X_test)[:, 1]
        
        # Apply threshold
        train_preds = (train_probs >= threshold).astype(int)
        test_preds = (test_probs >= threshold).astype(int)
        
        # Calculate metrics
        results = {
            'train': self.calculate_metrics(y_train, train_preds, train_probs),
            'test': self.calculate_metrics(y_test, test_preds, test_probs),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        return results
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive metrics with focus on recall"""
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Calculate recall for error class
        error_recall = recall_score(y_true, y_pred, pos_label=1)
        
        # Calculate false negative rate
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'error_recall': error_recall,
            'false_negative_rate': false_negative_rate,
            'total_errors_missed': fn
        }

def main(xml_directory, error_crd_file, use_bert=False, recall_weight=2.0, threshold=0.3):
    """Run complete pipeline with both BERT and non-BERT options"""
    # Initialize predictor
    predictor = HighRecallXMLPredictor(use_bert=use_bert, recall_weight=recall_weight)
    
    # Load and prepare data (using previous data preparation code)
    # ... (previous data preparation code remains the same)
    
    # Train and evaluate
    results = predictor.train_and_evaluate(features_df, threshold=threshold)
    
    # Print detailed results
    print("\nModel Performance Summary:")
    print("\nTest Set Metrics:")
    print(f"Recall (Error Detection Rate): {results['test']['error_recall']:.3f}")
    print(f"False Negative Rate: {results['test']['false_negative_rate']:.3f}")
    print(f"Total Errors Missed: {results['test']['total_errors_missed']}")
    
    print("\nConfusion Matrix:")
    print(results['test']['confusion_matrix'])
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, test_preds))
    
    return predictor, results

# Example usage
if __name__ == "__main__":
    # Compare BERT and non-BERT approaches
    results_no_bert = main(xml_dir, error_crd_file, use_bert=False, recall_weight=2.0)
    results_bert = main(xml_dir, error_crd_file, use_bert=True, recall_weight=2.0)
