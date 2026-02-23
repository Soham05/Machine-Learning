import prediction, importc2pdetail, plansummary
import sys
from configparser import ConfigParser
import pyodbc
import pandas as pd
import traceback

config_filename = str(sys.argv[1])
print("Reading config details from: ", config_filename)

config_object = ConfigParser()
config_object.read(config_filename)
predictionstarter_config = config_object["PREDICTIONSTARTER"]
sql_server = predictionstarter_config['sql_server']
sql_db = predictionstarter_config['sql_db']
sql_usr = predictionstarter_config['sql_usr']
sql_pwd = predictionstarter_config['sql_pwd']
sql_claims_data_metadata_table = predictionstarter_config['sql_claims_data_metadata_table']

# Deleting Previous Records from SQL Tables
#**conn = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)

conn = pyodbc.connect('DRIVER=/opt/microsoft/msodbcsql17/lib64/libmsodbcsql-17.7.so.2.1;SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)

cursor = conn.cursor()

# Testing: change data_load_status to 1 after testing
select_query = "SELECT client_code, run_date, data_load_status, prediction_status FROM " + sql_claims_data_metadata_table + " WHERE data_load_status = '1' AND prediction_status = '0'"
print("Query: ", select_query)
new_data_check_df = pd.read_sql(select_query, conn)
conn.close()
print(new_data_check_df.head(5))
print(new_data_check_df.columns)

if(len(new_data_check_df) != 0):
    for i in range(len(new_data_check_df)):
        client_code = new_data_check_df['client_code'].loc[i]
        run_date = new_data_check_df['run_date'].loc[i]
        print("Client Code: ", client_code)
        print("Run Date: ", run_date)

        print("Starting the prediction scripts...")

        try:
            print("Running the 1st script: prediction...")
            prediction.prediction_function(client_code, run_date, config_filename)
        except Exception:
            print('Error occurred in prediction_function() part. Stopping execution.\n')
            traceback.print_exc()
        else:
            try:
                print("Running the 2nd script: importc2pdetail...")
                importc2pdetail.importc2pdetail_function(client_code, run_date, config_filename)
            except:
                print("Error occurred in importc2pdetail_function() part. Stopping execution.\n")
                traceback.print_exc()
            else:
                try:
                    print("Running the 3rd script: plansummary...")
                    plansummary.plansummary_function(client_code, run_date, config_filename)
                except:
                    print("Error occurred in plansummary_function() part. Stopping execution.\n")
                    traceback.print_exc()
                else:
                    print("Successfully finished executing the prediction scripts...")
                    #**conn = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};SERVER='+ sql_server + ';DATABASE=' + sql_db + ';UID=' + sql_usr + ';PWD=' + sql_pwd)
                    conn = pyodbc.connect('DRIVER=/opt/microsoft/msodbcsql17/lib64/libmsodbcsql-17.7.so.2.1;SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)
                    cursor = conn.cursor()
                    update_query = "UPDATE " + sql_claims_data_metadata_table + " SET prediction_status = '1' WHERE client_code ='" + client_code + "' AND run_date='" + run_date + "'"
                    print("Updating prediction_status using query: ", update_query)
                    cursor.execute(update_query).commit()
                    conn.close()
else:
    print("No new data found...")
