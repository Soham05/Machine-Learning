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
                # Randomly modify an attribute
                attr_key = np.random.choice(list(elem.attrib.keys()))
                elem.attrib[attr_key] = elem.attrib[attr_key] + '_modified'
        return element
    
    def swap_siblings(self, element):
        """Swap random sibling elements"""
        for elem in element.iter():
            if len(elem) >= 2 and np.random.random() < self.error_probability:
                # Get random indices to swap
                idx1, idx2 = np.random.choice(len(elem), 2, replace=False)
                elem[idx1], elem[idx2] = elem[idx2], elem[idx1]
        return element
    
    def remove_random_tags(self, element):
        """Remove random tags while preserving content"""
        for elem in element.iter():
            if len(elem) > 0 and np.random.random() < self.error_probability:
                # Randomly remove a child while preserving its children
                child_idx = np.random.randint(0, len(elem))
                child = elem[child_idx]
                elem.remove(child)
                # Add child's children to parent
                for grandchild in child:
                    elem.append(grandchild)
        return element


###########################In the XMLDataPreparation class:##############################
def prepare_dataset(self, xml_directory):
    # Add after loading features but before creating DataFrame
    augmenter = XMLAugmenter(error_probability=0.3)
    
    # Augment error examples
    augmented_features = []
    for features in all_features:
        if features['crd'] in self.error_crd_set:
            # Create augmented versions
            xml_file = features['filename']
            with open(xml_file, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            augmented_versions = augmenter.augment_xml(xml_content)
            
            # Process each augmented version
            for aug_xml in augmented_versions[1:]:  # Skip original
                aug_features = self.processor.extract_features(aug_xml)
                if aug_features:
                    aug_features['filename'] = f"aug_{features['filename']}"
                    augmented_features.append(aug_features)
    
    all_features.extend(augmented_features)


############################In the XMLErrorPredictor class:###################################
def __init__(self, class_weight=10):
    self.model = RandomForestClassifier(
        n_estimators=200,
        class_weight={0: 1, 1: class_weight},
        # Add regularization parameters
        max_depth=10,  # Limit tree depth
        min_samples_split=5,  # Require more samples to split
        min_samples_leaf=2,  # Require more samples in leaves
        random_state=42,
        n_jobs=-1
    )

#########################In the BERTXMLProcessor class:########################################
def get_bert_embedding(self, text):
    # Add dropout for regularization
    self.model.train()  # Enable dropout
    inputs = self.tokenizer(text, 
                          return_tensors="pt",
                          truncation=True,
                          max_length=self.max_length,
                          padding=True)
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = self.model(**inputs, dropout_prob=0.1)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    self.model.eval()  # Disable dropout for inference
    return embeddings[0]
