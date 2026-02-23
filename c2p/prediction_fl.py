#!/usr/bin/env python
# coding: utf-8
# Always remember to delete the prediction file if running the script for same client. Otherwise, all the previous results will also be used for aggregation.

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
import stringcase
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500) 
pd.set_option('display.max_rows', 500) 


#------------------------------------------------------------------------------------------------------------------------------------------#


config_filename = str(sys.argv[1])
#config_filename = '/claims_cert_pbmdev/claimscert/claims2plan_featurelevel/bin/dev_fl_config.ini'
print("Reading config details from: ", config_filename)

config_object = ConfigParser()
config_object.read(config_filename)
prediction_config = config_object["PREDICTION"]

date = prediction_config['date']
sql_server = prediction_config['sql_server']
sql_db = prediction_config['sql_db']
sql_usr = prediction_config['sql_usr']
sql_pwd = prediction_config['sql_pwd']
sql_claims_data_metadata_table = prediction_config['sql_claims_data_metadata_table']
#sql_claims_data_mapping_table = prediction_config['sql_claims_data_mapping_table']

working_dir = prediction_config['working_dir']
claims_data_dir = prediction_config['claims_data_dir']
pickle_dir = prediction_config['pickle_dir']

unique_value_target_fields = prediction_config['unique_value_target_fields']
unique_value_target_fields = pickle_dir + 'lists/' + unique_value_target_fields + '_' + date + '.csv'
dummy_columns_list_name = prediction_config['dummy_columns_list_name']
dummy_columns_list_name = pickle_dir + 'lists/' + dummy_columns_list_name + '_' + date
distinct_features_mapping_list = prediction_config['distinct_features_mapping_list']
distinct_features_mapping_list = pickle_dir + 'lists/' + distinct_features_mapping_list + '_' + date + '.csv'

label_encoder_name = prediction_config['label_encoder_name']
label_encoder_name = label_encoder_name + '_' + date
model_name = prediction_config['model_name']
model_name = model_name + '_' + date


#------------------------------------------------------------------------------------------------------------------------------------------#
# # Functions
#------------------------------------------------------------------------------------------------------------------------------------------#


def fill_missing_dummies(df, prediction_config):
    '''
    Function to create missing dummy columns that are required. These will have 0 throughout.
    '''
    
    pickle_dir = prediction_config['pickle_dir']
    model_name = prediction_config['model_name']
    model_name = model_name + '_' + date
    column_names_df = pd.read_csv(pickle_dir + 'lists/' + model_name + '_individual_target_feature_list.csv')
    features_values = column_names_df['features'].tolist()
    # Remove NaN values from this list. These values may be there from the unique list.
    # nan is not equal to nan (nan != nan). https://stackoverflow.com/questions/21011777/how-can-i-remove-nan-from-list-python-numpy
    features_values = [x for x in features_values if x == x]
    # Not_Applicable will be there in fields with unique value.
    features_values = [x for x in features_values if x != 'Not_Applicable']
    #print("\ndebug 0: ", features_values)
    #print("\ndebug 0.5: ", type(features_values))
    features_values = ','.join(features_values)
    #print("\ndebug 1: ", features_values)
    #print("\ndebug 2: ", type(features_values))
    features_values = features_values.split(',')
    features_values_set = set(features_values)
    #print("\ndebug 3: ", features_values_set)
    #features_values_set = features_values_set.remove(',')
    all_unique_features_list = list(features_values_set)
    #print("All unique features: ", all_unique_features_list)
    #print("All columns in df: ", df.columns.values)
    dummy_cols_to_create = list(features_values_set - set(df.columns.values))
    #print("Dummy columns to create: ", dummy_cols_to_create)
    
    # Create missing dummy columns and fill them with zeros.
    for col in dummy_cols_to_create:
        df[col] = 0
    
    return df 



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


def process_data(client_code, claims_data_dir, claims_data_filename, prediction_config):
    '''
    Function to read client claims data files and do some basic processing and creates dummy columns.
    Acceptable claims files are: 
    1) .xlsx files with single sheet
    2) .txt files with '|' delimiter
    '''
    
    # Code will work for xlsx files only and with a single sheet. 
    # Can be scaled to be used with multiple sheets or txt files with a delimiter. (Future Scope)
    print("Processing data...")
    
    #sql_claims_data_mapping_table = prediction_config['sql_claims_data_mapping_table']
    pickle_dir = prediction_config['pickle_dir']
    distinct_features_mapping_list = prediction_config['distinct_features_mapping_list']
    distinct_features_mapping_list = pickle_dir + 'lists/' + distinct_features_mapping_list + '_' + date + '.csv'
    
    # Code only works for xlsx (with 1 sheet) and txt files as of now.
    if '.xlsx' in claims_data_filename:
        df = pd.read_excel(claims_data_dir + claims_data_filename)
        #df = df.sample(n = 500) # Testing
    elif '.txt' in claims_data_filename:
        df = pd.read_csv(claims_data_dir + claims_data_filename, sep = '|')
    else:
        print("File format of the client claims is not yet supported. Only xlsx with a single sheet and txt is supported.")
        raise Exception
        
    # Considering only PAID claims.
    df.columns= df.columns.str.lower()
    df = df[df.claim_status.eq('P')]
    
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.astype(str).str.replace(" ", "_")
    df.columns = df.columns.astype(str).str.replace("/", "_")
    #print(df.head(2))
    
    # Commenting this as for feature level, dynamic fetaures are selected and updating SQL tables again and again was time consuming.
    #query = "select * from " + sql_claims_data_mapping_table + " where client_code='" + client_code + "'"  
    #mapping_df = pd.read_sql(query, conn)
    
    # For feature_level, reading mapping information from a stored CSV file.
    # For now, assuming that the column names are same for training and prediction.
    # In the future, when there may be differrent namings, we might have to add a column called as client_code in CSV.
    # After adding client_code, we will have to add a line after reading the file to filter only the mapping data for that client.
    # It will be a single line of code after the line below.
    mapping_df = pd.read_csv(distinct_features_mapping_list)
    
    print("Columns list after mapping: ", df.columns.values)
    #df = df.rename(columns={"3rdpartyexceptioncode" : "ardpartyexceptioncode"}, errors="raise")
    df = df.rename(columns={"3rdpartyexceptioncode" : "third_party_exception_code"}, errors="raise")
    print("Columns list after rename: ", df.columns.values)
    # Select only columns that are required as features for prediction. "prediction_column_name" is a field in SQL table which has the mapping details.
    df = df[mapping_df['prediction_column_name'].values]
    #print(df.columns.values)
    
    dummy_columns_list_name = prediction_config['dummy_columns_list_name']
    dummy_columns_list_name = pickle_dir + 'lists/' + dummy_columns_list_name + '_' + date
    
    open_file = open(dummy_columns_list_name, "rb")
    cat_cols = pickle.load(open_file)
    open_file.close()
    print("Cat columns list: ", cat_cols)
    
    # Replace field that's entirely space (or empty) with NaN 
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    df = df[df.final_plan_code != '0']
    df = df[df.final_plan_code != 0]

    dummy_columns_list = list(set(df.columns) & set(cat_cols))
    df[dummy_columns_list] = df[dummy_columns_list].astype(str)
    
    str_columns = dummy_columns_list + ['rxclaim_number', 'final_plan_code']
    all_columns = df.columns.tolist()
    numerical_columns = [x for x in all_columns if x not in str_columns]

    # Features Field Processing (Float) - DataTypes.
    #df[numerical_columns] = df[numerical_columns].astype(float)
    
    # Remove trailing whitespaces.
    #df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    df = clean_data(df, str_columns, numerical_columns)
    
    # Creating dummies for Feature Fields
    df = pd.get_dummies(df, prefix_sep='_=_', columns=dummy_columns_list, drop_first=False)
    try:
        df = fill_missing_dummies(df, prediction_config)
    except:
        traceback.print_exc()
        print("Exception occured while checking dummy columns..")
    
    #print("Columns after creating missing dummy columns: ", df.columns.values)
    
    return df



#------------------------------------------------------------------------------------------------------------------------------------------#


def prediction_featurelevel_function(client_code, run_date, claims_data_filename, prediction_config):
    '''
    Function which uses pickled models to do predictions.
    '''
    
    working_dir = prediction_config['working_dir']
    claims_data_dir = prediction_config['claims_data_dir']
    model_name = prediction_config['model_name']
    model_name = model_name + '_' + date
    pickle_dir = prediction_config['pickle_dir']
    #print(client_code, run_date, claims_data_filename, config_filename)
    
    column_names_df = pd.read_csv(pickle_dir + 'lists/' + model_name + '_' + 'individual_target_feature_list.csv')
    column_names_list = column_names_df['target'].values
    
    pred_df = pd.DataFrame(columns = column_names_list)
        
    # Read client claims data. Process it.
    df = process_data(client_code, claims_data_dir, claims_data_filename, prediction_config)
    
    # Writing rxclaim_number to prediction dataframe.
    pred_df['rxclaim_number'] = df['rxclaim_number']
    pred_df['client_plan_id'] = df['final_plan_code']
    
    # Predicting for each individual target.
    #for target in column_names_list:
    for i in range(0, len(column_names_df)):
        print("\nValue of i: ", i)
        target = column_names_df.iloc[i]['target']
        features = column_names_df.iloc[i]['features'].split(',')
        
        # This length check is for unique value fields. We don't want to predict using models for these fields.
        if len(features) > 2:
            model_selected =  column_names_df.iloc[i]['model']
            print("\nFor Target: ", target)
            print("Features list: ", features)

            test_df = df[features]
            print("Features found in processed data: ", test_df.columns.values)
            target_model_name = model_name + '_' + target
            model_path = pickle_dir + 'models/' + target_model_name
            clf = pickle.load(open(model_path, 'rb'))
            try:
                pred_df[target] = clf.predict(test_df)
            except:
                print("Feature columns mismatch for {}. Moving to the next target".format(target))
                traceback.print_exc()
                continue
        else:
            print("\nFor Target: ", target)
            print("Features list: ", features)
            print("Skipping prediction for target {0} as it must be a unique value field.".format(target))
        
    return pred_df



#------------------------------------------------------------------------------------------------------------------------------------------#


def aggregate_results(prediction_filename, client_code, prediction_config):
    '''
    Function to aggregare prediction results per base plan in the client claims data using voting logic.
    '''
    
    # Delete records with same client_code and isfeature_level_prediction=1 SQL table (predicted_brmd_master) before loading new values. 
    print("Aggregating prediction results...")
    label_encoder_name = prediction_config['label_encoder_name']
    label_encoder_name = label_encoder_name + '_' + date
    model_name = prediction_config['model_name']
    model_name = model_name + '_' + date
    pickle_dir = prediction_config['pickle_dir']
    unique_value_target_fields = prediction_config['unique_value_target_fields']
    unique_value_target_fields = pickle_dir + 'lists/' + unique_value_target_fields + '_' + date + '.csv'
    
    individual_target_feature_list_df = pd.read_csv(pickle_dir + 'lists/' + model_name + '_' + 'individual_target_feature_list.csv')
    df = pd.read_csv(prediction_filename)
    df = df[df.client_plan_id != '0']
    cols_list = list(set(df.columns.tolist()) - set(['client_plan_id', 'rxclaim_number', 'run_date']))
    #print(cols_list)
    
    plan_df = pd.DataFrame(columns=['client_plan_id', 'client_code'])
    
    # Read values from unique_value_target_fields list.
    unique_value_target_fields_df = pd.read_csv(unique_value_target_fields)
    
    for i in range(0, len(df.client_plan_id.unique())):
        plan = df.client_plan_id.unique()[i]
        #print(plan)
        plan_df.loc[i, 'client_plan_id'] = plan
        plan_df.loc[i, 'client_code'] = client_code
        
        # Add values from unique list to plan_df.
        for j in range(len(unique_value_target_fields_df)):
            unique_target = str(unique_value_target_fields_df['target'].iloc[j])
            unique_value = str(unique_value_target_fields_df['value'].iloc[j])
            plan_df.loc[i, unique_target] = unique_value

        claims_count_per_plan = len(df[df.client_plan_id == plan])
        print("Claims count for plan {0}: {1}".format(plan, claims_count_per_plan))
        accuracy_json = {}

        grouped_df = df[df.client_plan_id == plan].groupby("client_plan_id")[cols_list].sum()
        
        for target_col in individual_target_feature_list_df['consolidated_target'].unique():
            #print(col)
            dummy_cols_list = [s for s in df.columns if target_col+'_=_' in s]
            if(len(dummy_cols_list) > 0):
                #print(dummy_cols_list)
                max_values = grouped_df[dummy_cols_list].idxmax(axis=1)
                value_string = max_values[0]
                sum_df = grouped_df[value_string]
                accuracy = round(float(sum_df[0]/claims_count_per_plan) * 100, 1)
                # Hard-coding for now.
                if accuracy < 33.0:
                    accuracy_json[stringcase.camelcase(target_col)] = 33.0
                else:
                    accuracy_json[stringcase.camelcase(target_col)] = accuracy
                value_string = value_string[value_string.find('_=_')+3:]
                plan_df.loc[i, target_col] = value_string
                plan_df.fillna(0, inplace = True)
            else:
                # Unique Values
                accuracy = 100.0
                accuracy_json[stringcase.camelcase(target_col)] = accuracy
                
        accuracy_json = str(accuracy_json).replace("'",'"')
        accuracy_json = str(accuracy_json).replace("_",'')
        plan_df.loc[i, 'field_accuracy_json'] = str(accuracy_json)
             
    for target_col in individual_target_feature_list_df['consolidated_target'].unique():
        # Decoding labels back to original form.
        pickle_file = pickle_dir + 'label_encoders/' + label_encoder_name + '_' + target_col + '.pkl'
        try:
            with open(pickle_file, 'rb') as handle:
                label_encoder = pickle.load(handle)
            plan_df[target_col] = label_encoder.inverse_transform(plan_df[target_col].values.astype(int))
        except:
            print('No label encoder found for: ', target_col)
            continue
    
    print("\nResulting aggreated plans look like: ", plan_df)
    
    return plan_df



#------------------------------------------------------------------------------------------------------------------------------------------#


def update_sql_tables(plan_df, client_code, prediction_config, conn):
    '''
    Function to load final results and other information in SQL tables.
    '''
    
    sql_predicted_brmd_master_table = prediction_config['sql_predicted_brmd_master_table']
    sql_event_master_table = prediction_config['sql_event_master_table']
    plan_df.rename(columns = {'client_plan_id':'plan_id'}, inplace = True)
    
    plan_df.replace('nan', np.nan, inplace=True)  
    plan_df.fillna(value='NA', inplace=True)
    
    cursor = conn.cursor()
    
    if(len(plan_df) != 0):
        print("Deleting previous plans...")
        cursor.execute("""DELETE FROM {0} WHERE  client_code = '{1}' AND
                       isfeature_level_prediction = 1""".format(sql_predicted_brmd_master_table, client_code)).commit()   
    else:
        print("No new plans to load.")
        raise Exception
        
    print("Loading results in SQL tables...")
    sql_predicted_brmd_master_table = prediction_config['sql_predicted_brmd_master_table']
    sql_event_master_table = prediction_config['sql_event_master_table']
    plan_df.rename(columns = {'client_plan_id':'plan_id'}, inplace = True)
    plan_df['isfeature_level_prediction'] = 1.0
    
    # Compulsory Fields.
    plan_df['created_timestamp'] = pd.datetime.now()
    plan_df['status'] = 'A'
	
    plan_df = plan_df.drop(['selected_prediction_mode', 'otc_coverage'], axis=1)
    print(plan_df.columns.values)
    
    myDict = dict()
    for i in range(len(plan_df)):
        myDict['key' + str(i)] = plan_df.loc[i].tolist()
		
    print(myDict)
    
    insertQuery  = '''INSERT into {0} ({1}) VALUES ({2})'''.format(sql_predicted_brmd_master_table, ",".join(plan_df.columns), ','.join('?' * len(plan_df.columns)))
    
    print("Insert query: ", insertQuery)
    
    for key in myDict:
        cursor.execute(insertQuery, myDict[key])
        conn.commit()
    print("Inserted plan details in SQL table: ", sql_predicted_brmd_master_table)
    
    print("Generating Prediction Event in SQL table event_master...")
    event_type = 'PREDICTION_FEATURE'
    event_message = '{"client_code":"' + client_code + '" }'
    status = 'PENDING'
    cursor.execute('''INSERT into {0} VALUES (?,?,?)'''.format(sql_event_master_table), (event_type, event_message, status)).commit()


#------------------------------------------------------------------------------------------------------------------------------------------#
# # Main Execution
#------------------------------------------------------------------------------------------------------------------------------------------#

#conn = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)

conn = pyodbc.connect('DRIVER=/opt/microsoft/msodbcsql17/lib64/libmsodbcsql-17.7.so.2.1;SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)

cursor = conn.cursor()

select_query = "SELECT client_code, run_date, claims_data_filename, data_load_status, featurelevel_prediction_status FROM " + sql_claims_data_metadata_table + " WHERE data_load_status = '1' AND featurelevel_prediction_status = '0'"
print("Query: ", select_query)  
new_data_check_df = pd.read_sql(select_query, conn)

print(new_data_check_df.head(5))

if(len(new_data_check_df) != 0):
    for i in range(len(new_data_check_df)):
        print("\nFile number from the SQL table: ", i)
        client_code = new_data_check_df['client_code'].loc[i]
        run_date = new_data_check_df['run_date'].loc[i]
        claims_data_filename = new_data_check_df['claims_data_filename'].loc[i]
        print("Client Code: ", client_code)
        print("Claims Data Filename: ", claims_data_filename)

        print("Starting the prediction scripts...")

        try:
            predictions = prediction_featurelevel_function(client_code, run_date, claims_data_filename, prediction_config)
            predictions['run_date'] = run_date
            
            prediction_filename = working_dir + 'predictions/' + client_code + '_predictions.csv'
            if os.path.isfile(prediction_filename):
                old_pred_df = pd.read_csv(prediction_filename)
                predictions = predictions.append(old_pred_df, ignore_index=True)
                predictions.drop_duplicates(subset = ['rxclaim_number', 'client_plan_id'], keep='last', inplace=True)
                predictions.to_csv(prediction_filename, index = False)
            else:
                predictions.drop_duplicates(subset = ['rxclaim_number', 'client_plan_id'], keep='last', inplace=True)
                predictions.to_csv(prediction_filename, index = False)
            os.chmod(prediction_filename, 0o777)
            try:
                print("Aggregating prediction results for: ", client_code, run_date)
                plan_df = aggregate_results(prediction_filename, client_code, prediction_config)
            except Exception:
                print('Error occurred in aggregate_results() part.\n')
                traceback.print_exc()
            else:
                update_sql_tables(plan_df, client_code, prediction_config, conn)  
        except Exception:
            traceback.print_exc()
            print('Error occurred in prediction_featurelevel_function() part.\n')
            
        else:
            print("Successfully finished executing the prediction scripts...")
            update_query = "UPDATE " + sql_claims_data_metadata_table + " SET featurelevel_prediction_status = '1' WHERE client_code ='" + client_code + "' AND run_date='" + run_date + "'"
            print("Updating featurelevel_prediction_status using query: ", update_query)
            cursor.execute(update_query).commit()   
            print("\n\n")
else:
    print("No new data found...")

conn.close()


#------------------------------------------------------------------------------------------------------------------------------------------#
