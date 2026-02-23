#!/usr/bin/env python
# coding: utf-8

# # Claims to Plans: Model Training

# # Notes
# - Following a specific directory structure. Code will only work for this structure.
# - "file://" extension is required before the file path when loading local files. Remove/modify according to the need.

import pyspark
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark import SparkContext, SparkConf, SQLContext
import pyspark
import sys, os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import isnan, when, count, col, countDistinct
from pyspark.sql.types import StringType
import datetime as dt
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import pandas as pd
from pyspark.sql.types import StructField,IntegerType, StructType,StringType, DoubleType
from configparser import ConfigParser
import pyodbc

sfOptions = {
    "sfURL" : "cvscdwprd.us-central1.gcp.snowflakecomputing.com",
    "sfUser" : "APP_PBM_SMRTCMP_PRD",
    "sfPassword" : "Fm$5#_9J",
    "sfDatabase" : "EDP_PBM_APPS_PROD",
    "sfSchema" : "APP_SMRTCMP",
    "sfRole" : "EDP_SMRTCMP_PROD_FUNC_ROLE",
    "sfWarehouse" : "WH_SMRTCMP_APP_PROD"
    }
snowflake_source_name = "net.snowflake.spark.snowflake"


# Import Config Details
#Read config.ini file. Expects the whole path of config file.
config_filename = str(sys.argv[1])
print("Loading configurations from: ", config_filename)
config_object = ConfigParser()
config_object.read(config_filename)
training_config = config_object["TRAINING"]

jars_path = training_config['jars_path']
#**hive_db = training_config['hive_db']
hive_training_table = training_config['hive_training_table']
hive_training_table_load_date = training_config['hive_training_table_load_date']
working_dir = training_config['working_dir']
model_name = training_config['model_name']
data_pipeline_name = training_config['data_pipeline_name']
label_pipeline_name = training_config['label_pipeline_name']

sql_server = training_config['sql_server']
sql_db = training_config['sql_db']
sql_usr = training_config['sql_usr']
sql_pwd = training_config['sql_pwd']
sql_training_data_details_table = training_config['sql_training_data_details_table']

print(jars_path,  hive_training_table, hive_training_table_load_date, working_dir, model_name, data_pipeline_name, label_pipeline_name)

spark = SparkSession.builder.appName("PySpark - Claims to Plan Prediction: Model Training Script").config("spark.driver.extraClassPath", jars_path + "mssql-jdbc-7.0.0.jre8.jar").config("spark.driver.memory", "12g").getOrCreate()


sc = spark.sparkContext
sqlContext=SQLContext(sc)
#sqlContext.sql('use ' + hive_db)

print("Reading data from HIVE table...")

###cmpDF = sqlContext.sql("SELECT * FROM " + hive_training_table + " where load_date='" + hive_training_table_load_date + "'")
#**cmpDF = sqlContext.sql("SELECT * FROM " + hive_training_table + " where rundate='" + hive_training_table_load_date + "'")
cmpDF = sqlContext.read.format(snowflake_source_name).options(**sfOptions).option("query","SELECT * FROM " + hive_training_table + " where load_date='" + hive_training_table_load_date + "'").load()
cmpDF = cmpDF.withColumn("FINAL_PLAN_CODE", F.trim(cmpDF.FINAL_PLAN_CODE))
#cmpDF.createOrReplaceTempView("temp_claims_data")

#query = "SELECT * FROM temp_claims_data where final_plan_code in ('INRX-CA01', 'INRX-DC01', 'INRX-GA01', 'INRX-GA02', 'INRX-IA01', 'INRX-IN01', 'INRX-IN02', 'INRX-KY01', 'INRX-KY02', 'INRX-LA01', 'INRX-MD', 'INRX-MD01', 'INRX-NJ01', 'INRX-NJ03', 'INRX-NV01', 'INRX-NY01', 'INRX-SC01', 'INRX-TX01', 'INRX-TX02', 'INRX-VA01', 'INRX-VA02', 'INRX-WA01', 'INRXNYEP1', 'INRXNYEP2', 'IRX-WNY01')"
#cmpDF = sqlContext.sql(query)

'''
query = "select distinct(final_plan_code) from temp_claims_data"
distinct_cmpDF = sqlContext.sql(query)
print("Distinct plans: ", distinct_cmpDF.show(200))

query = "select count(*), final_plan_code from (SELECT * FROM temp_claims_data where final_plan_code in ('INRX-CA01', 'INRX-DC01', 'INRX-GA01', 'INRX-GA02', 'INRX-IA01', 'INRX-IN01', 'INRX-IN02', 'INRX-KY01', 'INRX-KY02', 'INRX-LA01', 'INRX-MD', 'INRX-MD01', 'INRX-NJ01', 'INRX-NJ03', 'INRX-NV01', 'INRX-NY01', 'INRX-SC01', 'INRX-TX01', 'INRX-TX02', 'INRX-VA01', 'INRX-VA02', 'INRX-WA01', 'INRXNYEP1', 'INRXNYEP2', 'IRX-WNY01')) group by final_plan_code"
cmpDF = sqlContext.sql(query)
print("Count of claims for plans: ", cmpDF.show(30))

raise Exception
'''

# Columns that we want from Hive table.
###cols_hive = ['rxc_plan_id', 'scenario', 'rxc_transaction_id','rxc_base_plan',
###             'rxc_serv_msg','rxc_copay_amount','rxc_oop_amount_appld','rxc_prior_auth_ind',
###             'rxc_refill_number','rxc_cob_indicator','rxc_dispense_as_wrtn','rxc_number_of_fills',
###             'rxc_maintenance_drug','rxc_dea_class_of_drg','rxc_dispense_qty','rxc_days_supply',
###             'rxc_drug_indicator','rxc_route_of_admin','rxc_drug_class','rxc_gender',
###             'rxc_usual_customary','rxc_dispensing_fee','rxc_total_sales_tax','rxc_other_coveragecd',
###             'rxc_person_code','rxc_hra', 'brmd_base_plan_id'
###            ]

cols_hive = ['rxclaim_number','splty_flg','copay_amount','individual_oop_ptd_','pa_ind',
            'fill_number','cob_indicator','daw_psc_code','rflallwd','maintenance_drug_ind',
            'dea_code','quantity_dispensed','days_supply','med_d_drug_indicator','route_of_admin',
             'multi_source_ind','submitted_patient_gender','submitted_usual_and_customary_amount','dispensing_fee',
            'total_sales_tax_amt','submitted_other_coverage_code','submitted_person_code','hra_amount',
             'final_plan_code']

print("Count of records from Hive table after sampling: ", cmpDF.count())

# Filtering the columns required.
cmpDF = cmpDF.select(cols_hive)

print("Data read from Hive table successfully.")

#cmpDF = cmpDF.dropDuplicates()
###cmpDF = cmpDF.withColumnRenamed("rxc_plan_id","plan_id")
###cmpDF = cmpDF.filter("plan_id is not null")

# Target column.
#cmpDF = cmpDF.filter("base_plan_id is not null")

###for name in cmpDF.schema.names:
###    if name not in ('rxc_transaction_id','rxc_base_plan'):
###        cmpDF = cmpDF.withColumnRenamed(name, name.replace('rxc_',''))

# Renaming base_plan_id to brmd_base_plan_id
#cmpDF = cmpDF.withColumnRenamed("base_plan_id", "brmd_base_plan_id")

# Col that gives error = 'transaction_sts' and 'brmd_base_plan_id' is here just referred as 'base_plan_id'
###train_cols = ['rxc_transaction_id','rxc_base_plan','scenario','brmd_base_plan_id',
###              'serv_msg','copay_amount','oop_amount_appld','prior_auth_ind',
###             'refill_number','cob_indicator','dispense_as_wrtn','number_of_fills',
###              'maintenance_drug','dea_class_of_drg','dispense_qty','days_supply',
###              'drug_indicator','route_of_admin','drug_class','gender',
###              'usual_customary','dispensing_fee','total_sales_tax','other_coveragecd',
###              'person_code','hra']

###train = cmpDF.select(train_cols)
train = cmpDF

print("Count of records from training data: ", train.count())

cwd = working_dir
print('Current Working Directory: ', cwd)
pickle_dir = cwd + 'pickle/'
print("debug: ", pickle_dir)

# List of columns to exclude.
###cols_to_exclude = ['rxc_transaction_id','rxc_base_plan','scenario']
cols_to_exclude = ['rxclaim_number', 'temp']

# List "excluded" has list of columns that are ACTUALLY excluded by comparing with the "cols_to_exclude" list.
col_excluded = [x for x in train.columns if x in cols_to_exclude]

# Keeping a copy of excluded data (just in case).
excluded_df = train.select(*col_excluded)

# Filtering data.
train = train.select([c for c in train.columns if c not in col_excluded])

# Getting a list of count of distinct values for each column.
# print('Number of distinct values for each column: ', train.agg(*(countDistinct(col(c)).alias(c) for c in train.columns)).collect())

# List contains "brmd_base_plan_id" as an extra field from the other list defined below. Change letter when opitimizing code.
###cat_cols = ['brmd_base_plan_id','serv_msg','prior_auth_ind','cob_indicator',
###            'dispense_as_wrtn','maintenance_drug','dea_class_of_drg','drug_indicator',
###            'route_of_admin','drug_class','gender','other_coveragecd',
###            'person_code']

cat_cols = ['splty_flg', 'pa_ind','cob_indicator', 'daw_psc_code', 'maintenance_drug_ind',
            'dea_code', 'med_d_drug_indicator', 'route_of_admin', 'multi_source_ind', 'submitted_patient_gender',
            'submitted_other_coverage_code', 'submitted_person_code', 'final_plan_code']


# Data Procesing: Copy same code to Prediction Script
for col in train.columns:
    train = train.withColumn(col, when(F.col(col) == "", "Blank").otherwise(F.col(col)))
    if col not in cat_cols:
        train = train.withColumn(col, train[col].cast(DoubleType()))

train = train.fillna('UNK', subset=cat_cols)
train = train.fillna(0)

print(train.show(2))

###label = 'brmd_base_plan_id'
label = 'final_plan_code'

###categoricalColumns = ['serv_msg','prior_auth_ind','cob_indicator','dispense_as_wrtn',
###                      'maintenance_drug','dea_class_of_drg','drug_indicator','route_of_admin',
###                      'drug_class','gender','other_coveragecd','person_code']
categoricalColumns = ['splty_flg', 'pa_ind','cob_indicator', 'daw_psc_code', 'maintenance_drug_ind',
            'dea_code', 'med_d_drug_indicator', 'route_of_admin', 'multi_source_ind', 'submitted_patient_gender',
            'submitted_other_coverage_code', 'submitted_person_code']


###numericalColumns = ['copay_amount','oop_amount_appld','refill_number','number_of_fills',
###                    'dispense_qty','days_supply','usual_customary','dispensing_fee',
###                    'total_sales_tax', 'hra']
numericalColumns = ['copay_amount','individual_oop_ptd_','fill_number','rflallwd',
                    'quantity_dispensed','days_supply','submitted_usual_and_customary_amount','dispensing_fee',
                    'total_sales_tax_amt', 'hra_amount']

# OHE
categoricalColumnsclassVec = [c + "classVec" for c in categoricalColumns]

stages = []

for categoricalColumn in categoricalColumns:
    # Category Indexing with StringIndexer. "setHandleInvalid = keep" to accomodate invalid/new values in prediction dataset.
    stringIndexer = StringIndexer(inputCol=categoricalColumn, outputCol = categoricalColumn+"Index").setHandleInvalid("keep")
    # Using OneHotEncoder to convert categorical variables into binary SparseVectors.
    # To accomodate unknown values in prediction dataset we should set handleInvalid = 'keep' and dropLast = False.
    # handleInvalid argument not compatible for this version 2.2.0 of spark.

    # OHE
    encoder = OneHotEncoder(inputCol=categoricalColumn+"Index", outputCol=categoricalColumn+"classVec", dropLast = False)

    # Adding stages.
    # OHE
    stages += [stringIndexer, encoder]

assemblerInputs = categoricalColumnsclassVec + numericalColumns
assembler = VectorAssembler(inputCols = assemblerInputs, outputCol = "features")
stages += [assembler]

pipelineModel = Pipeline().setStages(stages).fit(train)
train_vector_transformed = pipelineModel.transform(train)

pipelineLabel = StringIndexer(inputCol = label, outputCol = 'label').fit(train_vector_transformed)

train_vector_transformed = pipelineLabel.transform(train_vector_transformed)
train_vector_transformed.show(1)

# Train Test Split
#tn, ts = train_vector_transformed.randomSplit([0.8, 0.2], seed = 987654)
tn = train_vector_transformed

print("Training Dataset Count: " + str(tn.count()))
###print("Test Dataset Count: " + str(ts.count()))

evaluator = MulticlassClassificationEvaluator(labelCol = 'label',
                                                  predictionCol = 'prediction',
                                                  metricName = 'f1')

# Random Forest
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
p_rf = ParamGridBuilder().addGrid(rf.numTrees, [30, 60]).addGrid(rf.maxDepth, [5, 8]).build()
crossval = CrossValidator(estimator = rf,
                       estimatorParamMaps = p_rf,
                      evaluator = evaluator,
                      numFolds = 4)
cvModel_rf = crossval.fit(tn)
print('Average cross validation score on 4 random samples for the Random Forest model is: {}'.format(cvModel_rf.avgMetrics))


# Logistic Regression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label')
p_lr = ParamGridBuilder().addGrid(lr.threshold, [0.4, 0.6]).build()
crossval = CrossValidator(estimator = lr,
                       estimatorParamMaps = p_lr,
                      evaluator = evaluator,
                      numFolds = 4)
cvModel_lr = crossval.fit(tn)
print('Average cross validation score on 4 random samples for the Logistic Regression model is: {}'.format(cvModel_lr.avgMetrics))


# Saving the data processing methods and prediction model. This will be used in the prediction script.
print("Saving pipelines and models...")

pipelineModel.write().overwrite().save('file://' + str(pickle_dir) + '/pipeline/' + data_pipeline_name)
pipelineLabel.write().overwrite().save('file://' + str(pickle_dir) + '/pipeline/' + label_pipeline_name)

print("debug: Saved pipelines and models!")

if(cvModel_rf.avgMetrics > cvModel_lr.avgMetrics):
    print("Best Random Forest Model using numTrees: ", cvModel_rf.bestModel._java_obj.getNumTrees())
    print("Best Random Forest Model using MaxDepth: ", cvModel_rf.bestModel._java_obj.getMaxDepth())
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label',
                                numTrees = int(cvModel_rf.bestModel._java_obj.getNumTrees()),
                                maxDepth = int(cvModel_rf.bestModel._java_obj.getMaxDepth()))
    print("Saving the Random Forest model..")
    clf = rf.fit(tn)
    clf.write().overwrite().save('file://' + str(pickle_dir) + 'model/' + model_name)
else:
    print("Best Logistic Regression Model using threshold: ", cvModel_lr.bestModel._java_obj.getThreshold())
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label',
                           threshold = float(cvModel_lr.bestModel._java_obj.getThreshold()))
    print("Saving the Logistic Regression model..")
    clf = lr.fit(tn)
    clf.write().overwrite().save('file://' + str(pickle_dir) + 'model/' + model_name)

spark.stop()

print('Writing model information to SQL table...')

# Writing model information to SQL table
#**conn = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)
conn = pyodbc.connect('DRIVER=/opt/microsoft/msodbcsql17/lib64/libmsodbcsql-17.7.so.2.1;SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)
cursor = conn.cursor()

insert_query = "INSERT INTO " + sql_training_data_details_table + " VALUES ('" + model_name + "', '" + data_pipeline_name + "', '" + label_pipeline_name + "')"
print("Inserting training model name details using query: ", insert_query)
cursor.execute(insert_query).commit()
conn.close()

print('Training file execution completed...')

