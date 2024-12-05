import os
import glob
import json
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Vertex AI and Gemini Imports
from vertexai.language_models import TextGenerationModel
import vertexai
from google.cloud import aiplatform

class GeminiXMLAnalyzer:
    def __init__(self, project_id):
        """
        Initialize Gemini XML Analyzer with Vertex AI configuration
        
        Args:
            project_id (str): Google Cloud Project ID
        """
        vertexai.init(project=project_id)
        self.model = TextGenerationModel.from_pretrained("gemini-pro")
    
    def analyze_xml_semantic_features(self, xml_content):
        """
        Use Gemini to extract semantic features and error potential
        
        Args:
            xml_content (str): XML file content
        
        Returns:
            dict: Semantic features and error insights
        """
        prompt = f"""
        Analyze this XML and provide a comprehensive semantic feature extraction:

        XML Content:
        ```xml
        {xml_content[:2000]}  # Limit content to avoid token overflow
        ```

        Please provide a detailed JSON analysis focusing on:
        1. Structural complexity
        2. Data type consistency
        3. Potential semantic errors
        4. Information richness
        5. Validation complexity
        6. Estimated error probability (0-1)

        Respond strictly in JSON format with numerical scores.
        """
        
        try:
            response = self.model.predict(
                prompt, 
                temperature=0.2,  # Low randomness for consistent analysis
                max_output_tokens=500
            )
            
            # Parse and validate JSON response
            semantic_features = self._parse_gemini_response(response.text)
            return semantic_features
        
        except Exception as e:
            print(f"Gemini analysis error: {e}")
            return self._default_semantic_features()
    
    def _parse_gemini_response(self, response_text):
        """
        Parse Gemini's JSON response with error handling
        
        Args:
            response_text (str): Raw response from Gemini
        
        Returns:
            dict: Parsed semantic features
        """
        try:
            # Extract JSON from markdown code block or raw text
            json_match = response_text.split('```json')[-1].split('```')[0].strip()
            if not json_match:
                json_match = response_text.strip()
            
            semantic_features = json.loads(json_match)
            
            # Validate and normalize features
            required_keys = [
                'structural_complexity', 
                'data_type_consistency', 
                'semantic_error_probability',
                'information_richness',
                'validation_complexity'
            ]
            
            for key in required_keys:
                if key not in semantic_features:
                    semantic_features[key] = 0.5  # Default neutral value
            
            return semantic_features
        
        except json.JSONDecodeError:
            return self._default_semantic_features()
    
    def _default_semantic_features(self):
        """
        Provide default semantic features if Gemini analysis fails
        
        Returns:
            dict: Default semantic feature set
        """
        return {
            'structural_complexity': 0.5,
            'data_type_consistency': 0.5,
            'semantic_error_probability': 0.5,
            'information_richness': 0.5,
            'validation_complexity': 0.5
        }

class XMLErrorClassifier:
    def __init__(self, project_id, q1_folder, q2_folder, q1_error_csv, q2_error_csv):
        """
        Initialize XML Error Classifier
        
        Args:
            project_id (str): Google Cloud Project ID
            q1_folder (str): Quarter 1 XML folder path
            q2_folder (str): Quarter 2 XML folder path
            q1_error_csv (str): Quarter 1 error filenames CSV
            q2_error_csv (str): Quarter 2 error filenames CSV
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
        
        Args:
            xml_path (str): Path to XML file
        
        Returns:
            list: Structural features
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
        
        Returns:
            tuple: Features array and labels array
        """
        # Read error file lists
        q1_errors = pd.read_csv(self.q1_error_csv)['filename'].tolist()
        q2_errors = pd.read_csv(self.q2_error_csv)['filename'].tolist()
        all_error_files = set(q1_errors + q2_errors)
        
        features = []
        semantic_features = []
        labels = []
        
        # Process XMLs from both quarters
        xml_folders = [self.q1_folder, self.q2_folder]
        for folder in xml_folders:
            for xml_file in glob.glob(os.path.join(folder, '*.xml')):
                filename = os.path.basename(xml_file)
                
                # Traditional features
                trad_features = self._extract_traditional_features(xml_file)
                
                # Read XML content for Gemini analysis
                with open(xml_file, 'r') as f:
                    xml_content = f.read()
                
                # Semantic features from Gemini
                semantic_analysis = self.gemini_analyzer.analyze_xml_semantic_features(xml_content)
                semantic_feature_vector = [
                    semantic_analysis['structural_complexity'],
                    semantic_analysis['data_type_consistency'],
                    semantic_analysis['semantic_error_probability'],
                    semantic_analysis['information_richness'],
                    semantic_analysis['validation_complexity']
                ]
                
                features.append(trad_features + semantic_feature_vector)
                semantic_features.append(semantic_feature_vector)
                labels.append(1 if filename in all_error_files else 0)
        
        return np.array(features), np.array(labels)
    
    def train_classifier(self, test_size=0.2):
        """
        Train XML error classification model
        
        Args:
            test_size (float): Proportion of test dataset
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
        
        Args:
            model_name (str): Name of the model in Vertex AI
        
        Returns:
            aiplatform.Model: Deployed Vertex AI model
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
        
        Args:
            new_xml_path (str): Path to new XML file
        
        Returns:
            dict: Prediction results
        """
        if not self.classifier:
            raise ValueError("Model not trained. Call train_classifier() first.")
        
        # Traditional features
        trad_features = self._extract_traditional_features(new_xml_path)
        
        # Read XML content for Gemini analysis
        with open(new_xml_path, 'r') as f:
            xml_content = f.read()
        
        # Semantic features from Gemini
        semantic_analysis = self.gemini_analyzer.analyze_xml_semantic_features(xml_content)
        semantic_feature_vector = [
            semantic_analysis['structural_complexity'],
            semantic_analysis['data_type_consistency'],
            semantic_analysis['semantic_error_probability'],
            semantic_analysis['information_richness'],
            semantic_analysis['validation_complexity']
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

# Example Usage
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
