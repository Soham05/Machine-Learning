import os
import glob
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Vertex AI Imports
import vertexai
from vertexai.language_models import TextEmbeddingModel, GenerativeModel
from google.cloud import aiplatform

class GeminiXMLAnalyzer:
    def __init__(self, project_id):
        vertexai.init(project=project_id)
        self.embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
        self.generative_model = GenerativeModel("gemini-pro")
    
    def analyze_xml_semantic_features(self, xml_content):
        """
        Extract semantic features using embeddings and generative model insights
        
        Args:
            xml_content (str): XML file content
        
        Returns:
            dict: Semantic features and error insights
        """
        try:
            # Truncate content to avoid token limits
            truncated_content = xml_content[:2000]
            
            # Generate embeddings
            embeddings = self.embedding_model.get_embeddings([truncated_content])
            embedding_vector = embeddings[0].values
            
            # Compute embedding-based statistical features
            semantic_features = {
                'embedding_mean': float(np.mean(embedding_vector)),
                'embedding_std': float(np.std(embedding_vector)),
                'embedding_max': float(np.max(embedding_vector)),
                'embedding_min': float(np.min(embedding_vector)),
                'embedding_dimension': len(embedding_vector)
            }
            
            # Optional: Use generative model for additional insights
            try:
                prompt = f"""
                Analyze this XML content and assess its potential for errors:
                
                XML Sample: {truncated_content}
                
                Provide a concise assessment of:
                1. Structural complexity
                2. Potential error indicators
                3. Data consistency likelihood
                """
                
                generative_response = self.generative_model.generate_content(prompt)
                
                # Extract additional semantic insights
                semantic_features.update({
                    'structural_complexity': self._extract_complexity_score(generative_response.text),
                    'semantic_error_probability': self._estimate_error_probability(generative_response.text)
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
            return self._default_semantic_features()
    
    def _extract_complexity_score(self, analysis_text):
        """
        Extract a complexity score from generative model's analysis
        """
        complexity_keywords = ['complex', 'intricate', 'complicated', 'sophisticated']
        return float(any(keyword in analysis_text.lower() for keyword in complexity_keywords))
    
    def _estimate_error_probability(self, analysis_text):
        """
        Estimate error probability based on generative model's analysis
        """
        error_keywords = ['error', 'issue', 'problem', 'inconsistent', 'incorrect']
        return float(sum(keyword in analysis_text.lower() for keyword in error_keywords) / len(error_keywords))
    
    def _default_semantic_features(self):
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

class XMLErrorClassifier:
    def __init__(self, project_id, q1_folder, q2_folder, q1_error_csv, q2_error_csv):
        """
        Initialize XML Error Classifier
        """
        self.project_id = project_id
        self.q1_folder = q1_folder
        self.q2_folder = q2_folder
        self.q1_error_csv = q1_error_csv
        self.q2_error_csv = q2_error_csv
        
        self.gemini_analyzer = GeminiXMLAnalyzer(project_id)
        self.classifier = None
    
    def _extract_traditional_features(self, xml_path):
        """
        Extract traditional XML structural features
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            unique_tags = len(set(elem.tag for elem in root.iter()))
            total_tags = len(list(root.iter()))
            
            def get_max_depth(element):
                if len(element) == 0:
                    return 1
                return 1 + max(get_max_depth(child) for child in element)
            
            max_depth = get_max_depth(root)
            file_size = os.path.getsize(xml_path)
            
            return [unique_tags, total_tags, max_depth, file_size]
        
        except Exception as e:
            print(f"Feature extraction error for {xml_path}: {e}")
            return [0, 0, 0, 0]
    
    def prepare_dataset(self):
        """
        Prepare training dataset with traditional and semantic features
        """
        # Read error file lists
        q1_errors = pd.read_csv(self.q1_error_csv)['filename'].tolist()
        q2_errors = pd.read_csv(self.q2_error_csv)['filename'].tolist()
        all_error_files = set(q1_errors + q2_errors)
        
        features = []
        labels = []
        
        # Process XMLs from both quarters
        xml_folders = [self.q1_folder, self.q2_folder]
        for folder in xml_folders:
            for xml_file in glob.glob(os.path.join(folder, '*.xml')):
                filename = os.path.basename(xml_file)
                
                # Traditional features
                trad_features = self._extract_traditional_features(xml_file)
                
                # Read XML content for semantic analysis
                with open(xml_file, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
                
                # Semantic features from Gemini Analyzer
                semantic_analysis = self.gemini_analyzer.analyze_xml_semantic_features(xml_content)
                
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
                
                # Combine traditional and semantic features
                combined_features = trad_features + semantic_feature_vector
                
                features.append(combined_features)
                labels.append(1 if filename in all_error_files else 0)
        
        return np.array(features), np.array(labels)
    
    def train_classifier(self, test_size=0.2):
        """
        Train XML error classification model
        """
        # Prepare dataset
        X, y = self.prepare_dataset()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Handle imbalanced dataset
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Train Random Forest with focus on error detection
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=5
        )
        
        self.classifier.fit(X_resampled, y_resampled)
        
        # Evaluate model
        y_pred = self.classifier.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    
    def deploy_to_vertex_ai(self, model_name="xml-error-classifier"):
        """
        Deploy trained model to Vertex AI
        """
        if not self.classifier:
            raise ValueError("Model not trained. Call train_classifier() first.")
        
        aiplatform.init(project=self.project_id)
        
        vertex_model = aiplatform.Model.create(
            display_name=model_name,
            artifact=self.classifier,
            sync=True
        )
        
        endpoint = vertex_model.deploy(
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=3
        )
        
        return endpoint
    
    def predict_xml_error(self, new_xml_path):
        """
        Predict error probability for a new XML file
        """
        if not self.classifier:
            raise ValueError("Model not trained. Call train_classifier() first.")
        
        # Traditional features
        trad_features = self._extract_traditional_features(new_xml_path)
        
        # Read XML content for semantic analysis
        with open(new_xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # Semantic features
        semantic_analysis = self.gemini_analyzer.analyze_xml_semantic_features(xml_content)
        
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
        combined_features = np.array(trad_features + semantic_feature_vector).reshape(1, -1)
        
        # Predict
        prediction = self.classifier.predict(combined_features)
        probability = self.classifier.predict_proba(combined_features)
        
        return {
            "is_error": bool(prediction[0]),
            "error_probability": probability[0][1],
            "semantic_insights": semantic_analysis
        }

# Main Execution
def main():
    PROJECT_ID = "your-gcp-project-id"
    Q1_FOLDER = "XMLTesting/files/Q1-XMLs/"
    Q2_FOLDER = "XMLTesting/files/Q2-XMLs/"
    Q1_ERROR_CSV = "XMLTesting/files/Q12024_error_xml_filenames.csv"
    Q2_ERROR_CSV = "XMLTesting/files/Q22024_error_xml_filenames.csv"
    
    # Initialize Classifier
    xml_classifier = XMLErrorClassifier(
        PROJECT_ID, 
        Q1_FOLDER, 
        Q2_FOLDER, 
        Q1_ERROR_CSV, 
        Q2_ERROR_CSV
    )
    
    # Train Classifier
    xml_classifier.train_classifier()
    
    # Optional: Deploy to Vertex AI
    # vertex_endpoint = xml_classifier.deploy_to_vertex_ai()
    
    # Predict for a new XML
    new_xml_path = "path/to/new/xml/file.xml"
    prediction = xml_classifier.predict_xml_error(new_xml_path)
    print(prediction)

if __name__ == "__main__":
    main()
