#!/usr/bin/env python
# coding: utf-8

# # Claims to Plans: Prediction
import os
#*os.system("export PYSPARK_PYTHON=/home/batch/certbat/.conda/envs/py36/bin/python")
#*os.system("kinit -kt /etc/security/keytabs/certbat.keytab certbat@TEST.CVSCAREMARK.COM")

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import isnan, when, count, col, countDistinct, length, trim, udf
from pyspark import SparkContext, SparkConf, SQLContext
import sys
from pathlib import Path
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StringIndexerModel, IndexToString
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import pyspark.sql.functions as F
import pyspark.sql.types as T
import joblib
import pandas as pd
import numpy as np
from pyspark.sql.types import StructField,IntegerType, StructType,StringType, DoubleType
from configparser import ConfigParser
from pyspark.sql.functions import lit

print('debug: Libraries imported')

def prediction_function(client_id, run_date, config_filename):
    print('debug: inside prediction_function()')

    print("Loading configurations from: ", config_filename)
    config_object = ConfigParser()
    config_object.read(config_filename)
    prediction_config = config_object["PREDICTION"]

    working_dir = prediction_config['working_dir']
    #**hive_db = prediction_config['hive_db']
    hive_claims_table = prediction_config['hive_claims_table']
    hive_predictions_table = prediction_config['hive_predictions_table']
    HDFS_PATH = prediction_config['HDFS_PATH']
    jars_path = prediction_config['jars_path']

    model_name = prediction_config['model_name']
    data_pipeline_name = prediction_config['data_pipeline_name']
    label_pipeline_name = prediction_config['label_pipeline_name']

    print(working_dir, hive_claims_table, hive_predictions_table, HDFS_PATH)

    # PATH Definitions
    cwd = working_dir # Root directory of the project.
    print('Current Working Directory: ', cwd)
    pickle_dir = cwd + 'pickle'

    print('client_code: ', client_id)
    print('run_date: ', run_date)
	
    #** Smart Compare changes
    snowflake_config = config_object["SNOWFLAKE_CRED"] 
    sfOptions = {}
    sfOptions["sfurl"] = snowflake_config['sf_url']
    sfOptions["sfuser"] = snowflake_config['sf_user']
    sfOptions["sfpassword"] = snowflake_config['sf_password']
    sfOptions["sfdatabase"] = snowflake_config['sf_database']
    sfOptions["sfschema"] = snowflake_config['sf_schema']
    sfOptions["sfrole"] = snowflake_config['sf_role']
    sfOptions["sfwarehouse"] = snowflake_config['sf_warehouse']
	
    SNOWFLAKE_SOURCE_NAME = snowflake_config['snowflake_source_name']

    spark = SparkSession.builder.appName("PySpark - Claims to Plan Prediction: Prediction Script")\
        .config("spark.driver.extraClassPath", jars_path + "mssql-jdbc-7.0.0.jre8.jar")\
        .config("spark.executor.memory", "128g")\
        .config("spark.executor.instances", "40")\
        .config("spark.driver.memory", "128g")\
        .getOrCreate()

    SparkContext.setSystemProperty('spark.executor.memory', '128g')
    SparkContext.setSystemProperty('spark.driver.memory', '128g')

    sc = spark.sparkContext
    sqlContext=SQLContext(sc)
    #**sqlContext.sql('USE ' + hive_db)

    print("Reading data from HIVE table...")

    query = "SELECT * FROM " + hive_claims_table + " where client_code=" + "'" + client_id + "'" + " and " + "rundate=" + "'" + run_date + "'"


    print("Query is: ", query)
    #**df = sqlContext.sql(query)
	#** Smart Compare changes
    df = spark.read.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("query",  query).load()

    # Columns that we want from Hive table.
    ###cols_hive = ['transaction_id','client_plan_id','serv_msg','copay_amount',
    ###      'oop_amount_appld','prior_auth_ind','refill_number','cob_indicator',
    ###      'dispense_as_wrtn','number_of_fills','maintenance_drug','dea_class_of_drg',
    ###      'dispense_qty','days_supply','drug_indicator','route_of_admin',
    ###      'drug_class','gender','usual_customary','dispensing_fee',
    ###      'total_sales_tax','other_coveragecd','person_code','hra']
    cols_hive = ['rxclaim_number','splty_flg','copay_amount','individual_oop_ptd_','pa_ind',
            'fill_number','cob_indicator','daw_psc_code','rflallwd','maintenance_drug_ind',
            'dea_code','quantity_dispensed','days_supply','med_d_drug_indicator','route_of_admin',
            'multi_source_ind','submitted_patient_gender','submitted_usual_and_customary_amount','dispensing_fee',
            'total_sales_tax_amt','submitted_other_coverage_code','submitted_person_code','hra_amount',
            'final_plan_code']

    # Filtering the columns required. (Might have to filter using clientid or clientcode and rundate.)
    df = df.select(cols_hive)
    
    # Added new on 12 Nov 2020
    df = df.withColumnRenamed("final_plan_code", "client_plan_id")

    print(df.show(1, False))
    print(" ")
    print("Columns from Hive table are: ", df.columns)
    print("Data read from Hive table successfully.")

    
    # Remove rows with blank client_plan_id
    df.registerTempTable('temp_table')
    query = "SELECT * FROM temp_table WHERE client_plan_id not like ' %'"
    df = sqlContext.sql(query)
    
    # Added new on 12 Nov 2020
    df = df.withColumn("final_plan_code", F.trim(df.client_plan_id))

    # Data Quality Test
    c=0
    x=df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).collect()
    y= [j for row in x for j in row]

    for i in y :
        if i!=0:
            print('The data is incorrect, Prepare the data correctly')

            break  
        c+=1
        if c==len(y):
            print('Data quality after missing value test is ok!')



    # List contains "brmd_base_plan_id" as an extra field from the other list defined below. Change letter when opitimizing code.
    ###cat_cols = ['transaction_id','client_plan_id','brmd_base_plan_id','serv_msg',
    ###            'prior_auth_ind','cob_indicator','dispense_as_wrtn','maintenance_drug',
    ###            'dea_class_of_drg','drug_indicator','route_of_admin','drug_class',
    ###            'gender','other_coveragecd','person_code']
    cat_cols = ['splty_flg', 'pa_ind','cob_indicator', 'daw_psc_code', 'maintenance_drug_ind',
            'dea_code', 'med_d_drug_indicator', 'route_of_admin', 'multi_source_ind', 'submitted_patient_gender',
            'submitted_other_coverage_code', 'submitted_person_code', 'client_plan_id']

    print("debug: column types before conversion: ", df.dtypes)   
    
    # Data Procesing: Copy same code to Prediction Script
    for col in df.columns:
        df = df.withColumn(col, when(F.col(col) == "", "Blank").otherwise(F.col(col)))
        if col not in cat_cols:
            df = df.withColumn(col, df[col].cast(DoubleType()))

    df = df.fillna('UNK', subset=cat_cols)
    df = df.fillna(0)

    ###print("debug: print value of a double column: ", df.select('hra').show(3, False))
    print("debug: print value of a double column: ", df.select('hra_amount').show(3, False))
    print("debug: column types after conversion: ", df.dtypes)
    print("Loading Data Processing Pipelines...")

    pipelineModel = PipelineModel.load('file://' + str(pickle_dir) + '/pipeline/' + data_pipeline_name + '/')
    df = pipelineModel.transform(df)

    print("Loading ML Models...")

    try:
        clf = RandomForestClassificationModel.load('file://' + str(pickle_dir) + '/model/' + model_name + '/')
    except:
        clf = LogisticRegressionModel.load('file://' + str(pickle_dir) + '/model/' + model_name + '/')

    pred = clf.transform(df)

    print("debug: predicted labels pred df: ", pred.select('prediction').show(10))

    pipelineLabel = StringIndexerModel.load('file://' + str(pickle_dir) + '/pipeline/' + label_pipeline_name + '/')
    PredConverter = IndexToString(inputCol='prediction', outputCol='predicted_plan_id', labels=pipelineLabel.labels)
    pred = PredConverter.transform(pred)

    print('Saving the predictions...')

    to_array = F.udf(lambda v: v.toArray().tolist(), T.ArrayType(T.FloatType()))
    pred = pred.withColumn('prob_array', to_array('probability'))

    def max_prob(an_array):

        return max(an_array)

    max_probUDF = udf(lambda an_array: max_prob(an_array), DoubleType())
    pred = pred.withColumn("probability_score", max_probUDF(pred["prob_array"]))

    print('Saving the predictions...')
    print("debug: columns of pred df: ", pred.columns)

    ###pred_to_csv = pred.select("transaction_id", "client_plan_id", "predicted_plan_id", "probability_score")
    pred_to_csv = pred.select("rxclaim_number", "client_plan_id", "predicted_plan_id", "probability_score")

    print("debug: pred_to_csv value: ", pred_to_csv.show(5, False))

    path = HDFS_PATH + "/" + hive_predictions_table + "/client_id=" + client_id + "/run_date=" + run_date
    pred_to_csv.write.mode('overwrite').parquet(path)
	
    pred_to_csv = pred_to_csv.withColumn("CLIENT_ID", lit(client_id))
    pred_to_csv = pred_to_csv.withColumn("RUN_DATE", lit(run_date))
    pred_to_csv = pred_to_csv.withColumn("TRANSACTION_ID", lit(""))
    pred_to_csv.show()
	
    try:
	    delete_query = "DELETE FROM " + hive_predictions_table + " WHERE client_id = '" + client_id + "' and run_date = '" + run_date + "'"
	    print ("Delete Query " + delete_query)
	    spark.sparkContext._jvm.net.snowflake.spark.snowflake.Utils.runQuery(sfOptions, delete_query)
		
    except Exception as e:
	    print("Failed to delete the existing records of the Snowflake table " + str(e))
	
    pred_to_csv.select("TRANSACTION_ID", "CLIENT_PLAN_ID", "PREDICTED_PLAN_ID", "PROBABILITY_SCORE","CLIENT_ID", "RUN_DATE").write.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("dbtable", hive_predictions_table).mode('append').save()

    #**print("debug: Altering hive table...")

    #**sql = "alter table " + hive_predictions_table + " add if not exists partition( client_id='" + client_id + "', run_date='" + run_date + "') location '" + path + "'"
    #**spark.sql('use ' + hive_db)
    #**spark.sql(sql)

    print("debug: Hive table altered...")

    spark.stop()
    print('Prediction Completed!')


