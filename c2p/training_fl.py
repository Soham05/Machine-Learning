#!/usr/bin/env python
# coding: utf-8

#------------------------------------------------------------------------------------------------------------------------------------------#


import os
import sys
from configparser import ConfigParser
import pyodbc
import pandas as pd
import traceback
import numpy as np
import warnings
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500) 
pd.set_option('display.max_rows', 500) 



#------------------------------------------------------------------------------------------------------------------------------------------#


config_filename = str(sys.argv[1])
#config_filename = '/claims_cert_pbmdev/claimscert/claims2plan_featurelevel/bin/dev_fl_config.ini'
print("Reading config details from: ", config_filename)

config_object = ConfigParser()
config_object.read(config_filename)
training_config = config_object["TRAINING"]

date = training_config['date']
working_dir = training_config['working_dir']
pickle_dir = training_config['pickle_dir']

training_data_filename = training_config['training_data_filename']
training_data_filename = training_data_filename + '_' + date + '.csv'

target_fields_list_name = training_config['target_fields_list_name']
target_fields_list_name = pickle_dir + 'lists/' + target_fields_list_name + '_' + date
unique_value_target_fields = training_config['unique_value_target_fields']
unique_value_target_fields = pickle_dir + 'lists/' + unique_value_target_fields + '_' + date + '.csv'
dummy_columns_list_name = training_config['dummy_columns_list_name']
dummy_columns_list_name = pickle_dir + 'lists/' + dummy_columns_list_name + '_' + date

label_encoder_name = training_config['label_encoder_name']
label_encoder_name = label_encoder_name + '_' + date
model_name = training_config['model_name']
model_name = model_name + '_' + date

sql_server = training_config['sql_server']
sql_db = training_config['sql_db']
sql_usr = training_config['sql_usr']
sql_pwd = training_config['sql_pwd']
sql_training_data_details_table = training_config['sql_training_data_details_table']


#------------------------------------------------------------------------------------------------------------------------------------------#
# ## Functions
#------------------------------------------------------------------------------------------------------------------------------------------#

def try_convert(string):
    '''
    Function to convert values that are integer/float but treated as strings. 
    '''
    
    try:
        string = str(float(string))
        return string
    except:
        return string



#------------------------------------------------------------------------------------------------------------------------------------------#


def clean_data(data, cat_data_list, num_data_list): 
    '''
    Function to clean data with missing value imputation and some other cleaning steps.
    '''
    
    # Fill NA in categorical variables. (To be used in Training & Prediction script)
    # Cleaning
    
    # Replace field that's entirely space (or empty) with NaN
    data.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x) # Remove if causing problems

    # Convert Object to String to avoid same INT and STR values considered separately. (To be used in Training & Prediction script)
    data[cat_data_list] = data[cat_data_list].astype(str)

    # Replace 'nan' string to np.nan. To be later filled with 'NA'
    for col in cat_data_list:
        data[col].replace('nan', np.nan, inplace=True)  
        data[col].fillna(value='NA', inplace=True)
        # Replace leading zeros in strings.
        data[col] = data[col].map(lambda x: x.lstrip('0') if(len(x)>1 and '0.' not in x)  else x)
        # Convert strings like 5000.0 to 5000
        data[col] = data[col].map(lambda x: try_convert(x))
        
    # Filter values in numeric columns where strings are present.
    for col in num_data_list: 
        data[col] = data[col].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
        
    data.replace('', 'NA', inplace = True)
        
    data[cat_data_list].fillna(value='NA', inplace=True)
    data[num_data_list].fillna(value=0, inplace=True)
    
    return data



#------------------------------------------------------------------------------------------------------------------------------------------#


def check_unique_values(data):   
    '''
    Function to check unique value fields. Columns having same value throughout.
    '''
    
    # Save list of target fields with unique values. 
    unique_value_columns_list = data.columns[data.nunique() <= 1]
    unique_value_columns_df = pd.DataFrame(columns=['target','value'])
    
    for i in range(len(unique_value_columns_list)):
        col = unique_value_columns_list[i]
        unique_value_columns_df.loc[i, 'target'] = col
        unique_value_columns_df.loc[i, 'value'] = data[col].unique()[0]

    unique_value_columns_df['value'].replace('nan', np.nan, inplace=True)    
    unique_value_columns_df.fillna(value='NA', inplace=True)
    
    return unique_value_columns_df, unique_value_columns_list



#------------------------------------------------------------------------------------------------------------------------------------------#


def get_features_dynamically(num_of_features, data, target, all_features):
    '''
    Function to dynamically select features for each encoded target variable.
    '''
    
    # Function to get top K features dynamically.
    X = data[all_features]
    y = data[target]
    
    # Features must be selected on the data that is to be used for training. Keep the ratio same.
    #try: 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 123, stratify = y)
    #except:
    #   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
    
    # Apply the SelectKBest object to the features and target
    fvalue_selector = SelectKBest(f_classif, k = num_of_features)
    X_kbest = fvalue_selector.fit_transform(X_train, y_train)
    features = X_train.columns[fvalue_selector.get_support(indices=True)]
    
    print("Number of features selected dynamically: ", len(features))
    
    return features



#------------------------------------------------------------------------------------------------------------------------------------------#


def train_models_quick(train_df, features, target):
    '''
    Function to train models. Model with best precision score is promoted..
    '''
    
    # Function to train models quickly. No hyperparameter tuning and cross-validaiton steps.
    X = train_df[features]
    y = train_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 123, stratify = y)
        
    # Logistic Regression
    lr = LogisticRegression(class_weight = 'balanced')
    lr = lr.fit(X_train, y_train)
    yhat = lr.predict(X_test)
    lr_precision_score = precision_score(y_test, yhat)
    lr_accuracy_score = accuracy_score(y_test, yhat)
    print("Precision and Accuracy Scores for Logistic Regression: ", lr_precision_score, lr_accuracy_score)

    # AdaBoost 
    ab = AdaBoostClassifier(n_estimators = 75) 
    ab = ab.fit(X_train, y_train)
    yhat = ab.predict(X_test)
    ab_precision_score = precision_score(y_test, yhat)
    ab_accuracy_score = accuracy_score(y_test, yhat)
    print("Precision and Accuracy Scores for AdaBoost: ", ab_precision_score, ab_accuracy_score)
    
    # Random Forest 
    rf = RandomForestClassifier(n_estimators = 75) 
    rf = rf.fit(X_train, y_train)
    yhat = rf.predict(X_test)
    rf_precision_score = precision_score(y_test, yhat)
    rf_accuracy_score = accuracy_score(y_test, yhat)
    print("Precision and Accuracy Scores for Random Forest: ", rf_precision_score, rf_accuracy_score)
   
    
    model_path = pickle_dir + 'models/' + model_name + '_' + target
    
    if(lr_precision_score > rf_precision_score and lr_precision_score > ab_precision_score):
        print("Saving the Logistic Regression model..")
        pickle.dump(lr, open(model_path, 'wb'))
        return 'lr', lr_precision_score, lr_accuracy_score
    elif(rf_precision_score > lr_precision_score and rf_precision_score > ab_precision_score):
        print("Saving the Random Forest model..")
        pickle.dump(rf, open(model_path, 'wb'))
        return 'rf', rf_precision_score, rf_accuracy_score
    else:
        print("Saving the Adaboost model..")
        pickle.dump(ab, open(model_path, 'wb'))
        return 'ab', ab_precision_score, ab_accuracy_score



#------------------------------------------------------------------------------------------------------------------------------------------#


def export_decision_tree_rules(train_df, features, target):
    '''
    Function to export decision tree rules for each encoded target field. Just for analysis purpose.
    '''
    
    # Export Decision Tree Rules
    X = train_df[features]
    y = train_df[target]
    decision_tree = DecisionTreeClassifier(random_state = 123, max_depth = 5)
    decision_tree = decision_tree.fit(X, y)
    rules = export_text(decision_tree, feature_names = features)
           
    return rules  



#------------------------------------------------------------------------------------------------------------------------------------------#


def get_distinct_features(features_list):
    '''
    Function to get list of all distinct features that are required. This is useful for mapping.
    '''
    
    # Create a list of distinct features required. This will have to added to the mapping SQL table manually.
    features_list.fillna('NA', inplace=True)
    distinct_features = []

    for i in range(len(features_list)):
        distinct_features = distinct_features + features_list['features'].iloc[i].split(',')

    cleaned_features = []
    for feature in distinct_features:
        idx = feature.find('_=_') 
        if idx != -1:
            cleaned_features.append(feature[:idx])
        else:
            cleaned_features.append(feature)

    cleaned_features = list(dict.fromkeys(cleaned_features))
    
    if 'NA' in cleaned_features:
        cleaned_features.remove('NA')
    if 'single value' in cleaned_features:
        cleaned_features.remove('single value')
        
    cleaned_features.append('final_plan_code')
    cleaned_features.append('rxclaim_number')
    print("Number of distinct features: ", len(cleaned_features))
    
    return cleaned_features     


#------------------------------------------------------------------------------------------------------------------------------------------#
# ## Main Execution
#------------------------------------------------------------------------------------------------------------------------------------------#

print("Starting Training...")

print(training_data_filename)

df = pd.read_csv(training_data_filename)
df.columns = df.columns.str.lower()

# Below line is for testing on a sample. 
#df = df.sample(n = 1000)

print("Size of training data: ", len(df))
print("\nTraining data: ", df.head(1))

# To be used in Training & Prediction script
file = open(target_fields_list_name,'rb')
target_fields_list = pickle.load(file)
file.close()

cat_data_list = df.select_dtypes(include='object').columns
# Line below to be used in Training Script only. Not to use in prediction script because target variables won't be there.
cat_data_list = cat_data_list.tolist() + target_fields_list
num_data_list = list(set(df.columns) - set(cat_data_list))

# Clean data (To be used in Prediction also)
print('Cleaning data...')
df = clean_data(df, cat_data_list, num_data_list)
print("Data cleaning done!")
print("Number of unique values: ", df.nunique())

# Unique Values in Target fields
print("Checking for unique values in target fields...")
unique_value_columns_df, unique_value_columns_list = check_unique_values(df)
print("Count of target fields with a single value: ", len(unique_value_columns_df))
print("\nTarget fields with a single value: ", unique_value_columns_df.head(50))

# Writing the list of fields with their unique values so that they can used during prediction.
# Not to be done again in Training script.
unique_value_columns_df.to_csv(unique_value_target_fields, header = True, index = False)

# Remove these columns from df
df.drop(columns=unique_value_columns_list, inplace=True, axis=1)
print("\nCount of columns after filtering unique value columns: ", len(df.columns))

target_fields_list = list(set(target_fields_list) - set(unique_value_columns_list))
print("\nFiltered target list after dropping columns with 1 value: ", target_fields_list)

# To be used in Training Script
claims_columns_list = list(set(df.columns.tolist()) - set(target_fields_list))
claims_columns_list.remove('rxclaim_number')
claims_columns_list.remove('final_plan_code')

# Create dummy columns for claim fields.
dummy_columns_list = df[claims_columns_list].select_dtypes(include='object').columns
print("Count of columns from claim fields to create dummies: ", len(dummy_columns_list))
print("Columns from claim fields to create dummies: ", dummy_columns_list)

df = pd.get_dummies(df, prefix_sep='_=_', columns=dummy_columns_list, drop_first=False)
print("New column count after creating dummies for claim fields: ", len(df.columns))

# Save this dummy columns list for the prediction script to use.
open_file = open(dummy_columns_list_name, "wb")
pickle.dump(dummy_columns_list, open_file)
open_file.close()
print("Saved the list of columns to create dummies in claim fields.")

# Features = Columns from Claim Fields (Dummies included)
all_features = list(set(df.columns) - set(target_fields_list))
all_features = list(set(all_features) - set(['rxclaim_number', 'final_plan_code', 'base_plan_id', 'xref_plan_code']))
print("Entire feature list: ", all_features)

# Remove plan_ids from target fields
target_fields_list = list(set(target_fields_list) - set(['rxclaim_number', 'final_plan_code', 'base_plan_id', 'xref_plan_code']))

individual_target_feature_list = pd.DataFrame(columns = ['consolidated_target','target','features','mapped_to','model','precision','accuracy'])

iterations_count = 0

os.makedirs(os.path.dirname(pickle_dir + 'label_encoders/'), exist_ok=True)
os.makedirs(os.path.dirname(pickle_dir + 'models/'), exist_ok=True)
os.makedirs(os.path.dirname(pickle_dir + 'dt_rules/'), exist_ok=True)


#------------------------------------------------------------------------------------------------------------------------------------------#

for i in range(0, len(target_fields_list)):
    print("\nValue of i: ", i)
    target = target_fields_list[i]
    print("\nFor Consolidated Target: ", target)
    
    train_df = df[[target] + all_features]

    # Label Encoding Target Fields
    le = LabelEncoder()
    train_df[target] = le.fit_transform(train_df[target].values).astype(str)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Mapping of labels for {0}: {1}".format(target, le_name_mapping))
    print("")
    
    # Saving label encoders. To be used to retrieve original values from encoded values.
    output = open(pickle_dir + 'label_encoders/' + label_encoder_name + '_' + target + '.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    
    # Creating dummies for Target Fields
    train_df = pd.get_dummies(train_df, prefix=target, prefix_sep='_=_', columns=[target], drop_first=False)
    target_string = target + '_=_' 
    individual_targets = [string for string in train_df.columns if target_string in string]
    
    for j in range(0, len(individual_targets)):
        print("\nValue of i: {0} and j: {1}.".format(i, j))
        target = individual_targets[j]
        print("For Individual Target: ", target)
        # Get actual label name
        unmapped_label = list(le_name_mapping.keys())[list(le_name_mapping.values()).index(j)]
        print("Mapped label for: ", unmapped_label)
        num_of_features = 8
        features = get_features_dynamically(num_of_features, train_df, target, all_features)
        features = features.tolist()
        print("Features selected dynamically: ", features)
        iterations_count += 1
        print("Number of iterations: ", iterations_count)

        try:
            # Using quick training approach.
            model_selected, precision, accuracy = train_models_quick(train_df, features, target)
            
            # Write Decision Tree Rules to a .txt file
            print("Extracting decision tree rules...")
            dt_rules = export_decision_tree_rules(train_df, features, target)
            dt_rules_file = pickle_dir + 'dt_rules/' + target + '___' + date + '.txt'
            dt_rules = unmapped_label + "\n\n\n" + dt_rules
            with open(dt_rules_file, "w") as outfile:
                outfile.write(dt_rules)
            
            # Writing individual target names and the associated features to a CSV. 
            # This will make implementation of prediction code easier.
            features_string = ','.join(features)
            new_row = {'consolidated_target':target_string[:-3], 'target':target, 'features':features_string, 'mapped_to':unmapped_label, 'model':model_selected, 'precision':precision, 'accuracy':accuracy}
            # Append new row to the dataframe.
            individual_target_feature_list = individual_target_feature_list.append(new_row, ignore_index = True)
        except:
            print("Something went wrong while training the model for: ", target)
            traceback.print_exc()
            continue 
    try:
        individual_target_feature_list_df = pd.read_csv(pickle_dir + 'lists/' + model_name + '_individual_target_feature_list.csv')
        individual_target_feature_list = pd.concat([individual_target_feature_list_df, individual_target_feature_list], axis=0)
        individual_target_feature_list.to_csv(pickle_dir + 'lists/' + model_name + '_individual_target_feature_list.csv', index = False)
    except:
        individual_target_feature_list.to_csv(pickle_dir + 'lists/' + model_name + '_individual_target_feature_list.csv', index = False)   
    finally:
        # Write unique value fields as fields with 100% accuracy.
        for col in unique_value_columns_df.target:
            new_row = {'consolidated_target':col, 'target':col, 'features':'Not_Applicable', 'mapped_to':'Not_Applicable', 'model':'Not_Applicable', 'precision':1.0, 'accuracy':1.0}
            # Append new row to the dataframe.
            individual_target_feature_list = individual_target_feature_list.append(new_row, ignore_index = True)

        # This may not be used for final code. But for testing there may be multiple runs which may cause duplicates.
        individual_target_feature_list_df = pd.read_csv(pickle_dir + 'lists/' + model_name + '_individual_target_feature_list.csv')
        individual_target_feature_list = pd.concat([individual_target_feature_list_df, individual_target_feature_list], axis=0)
        individual_target_feature_list.drop_duplicates(subset = ['target'], keep = "last", inplace = True)
        individual_target_feature_list.to_csv(pickle_dir + 'lists/' + model_name + '_individual_target_feature_list.csv', index = False)     
        
print("\nFinished with all models!")


#------------------------------------------------------------------------------------------------------------------------------------------#

print("Getting a list of required distinct features...")
distinct_features = get_distinct_features(individual_target_feature_list)
distinct_features_df = pd.DataFrame(columns=['training_column_name', 'prediction_column_name'])
# By default write same column names to both columns. Change manually if the mappings of column names are differrent.

# Remove Not_Applicable from features list if present.
try:
    distinct_features.remove('Not_Applicable')
except:
    print("Not_Applicable not found int list.")
    
distinct_features_df['training_column_name'] = distinct_features
distinct_features_df['prediction_column_name'] = distinct_features
distinct_features_df.to_csv(pickle_dir + 'lists/distinct_features_mapping_list_' + date + '.csv', index = False)


#------------------------------------------------------------------------------------------------------------------------------------------#


# Writing model information to SQL table
#conn = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)

conn = pyodbc.connect('DRIVER=/opt/microsoft/msodbcsql17/lib64/libmsodbcsql-17.7.so.2.1;SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)

cursor = conn.cursor()
data_pipeline_name = 'Not Applicable (Python Feature Level Code)'
insert_query = "INSERT INTO " + sql_training_data_details_table + " VALUES ('" + model_name + "', '" + data_pipeline_name + "', '" + label_encoder_name + "')"
print("Inserting training model name details using query: ", insert_query)
# Remove comment once code if finalized.
cursor.execute(insert_query).commit()
conn.close()

print('Training file execution completed...')


#------------------------------------------------------------------------------------------------------------------------------------------#
