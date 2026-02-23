import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
import sys
from configparser import ConfigParser
import pyodbc
from pyspark.sql.functions import col

def plansummary_function(client_id, run_date, config_filename):
    # Import Config Details
    #Read config.ini file. Expects the whole path of config file.
    print("Loading configurations from: ", config_filename)
    config_object = ConfigParser()
    config_object.read(config_filename)
    plansummary_config = config_object["PLANSUMMARY"]

    jars_path = plansummary_config['jars_path']
    #hive_db = plansummary_config['hive_db']
    hive_claim_prediction_detail_table = plansummary_config['hive_claim_prediction_detail_table']
    sql_db_url = plansummary_config['sql_db_url']
    sql_plan_summary_table = plansummary_config['sql_plan_summary_table'] 
    sql_event_master_table = plansummary_config['sql_event_master_table']
    sql_server = plansummary_config['sql_server']
    sql_db = plansummary_config['sql_db']
    sql_usr = plansummary_config['sql_usr']
    sql_pwd = plansummary_config['sql_pwd']
	
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

    # Deleting Previous Records from SQL Table
    #**conn = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)
    conn = pyodbc.connect('DRIVER=/opt/microsoft/msodbcsql17/lib64/libmsodbcsql-17.7.so.2.1;SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)
    cursor = conn.cursor()

    print("plan_summary_query = ", "DELETE FROM " + sql_plan_summary_table + " WHERE client_code='" + client_id + "' AND run_date='" + run_date + "'")
    print("event_master_query = ", "DELETE FROM " + sql_event_master_table + " WHERE event_message='{" + '"client_code":"' + client_id + '" }' + "'")
    cursor.execute("DELETE FROM " + sql_plan_summary_table + " WHERE client_code='" + client_id + "' AND run_date='" + run_date + "'").commit()
    cursor.execute("DELETE FROM " + sql_event_master_table + " WHERE event_message='{" + '"client_code":"' + client_id + '" }' + "'").commit()
    conn.close()

    print("Previous records deleted from SQL tables...")

    # Spark Code  
    spark = SparkSession \
        .builder \
        .appName("Plan Summary") \
        .config("spark.driver.extraClassPath", jars_path + "mssql-jdbc-7.0.0.jre8.jar") \
        .getOrCreate()

    sc = spark.sparkContext
    sqlContext=SQLContext(sc)
    #**sqlContext.sql("use " + hive_db)

    print("Reading prediction claims data...")

    claims_query = "select * from " + hive_claim_prediction_detail_table + " where client_id='" + client_id + "' and run_date='" + run_date + "'"
    print(claims_query)
    #**df = sqlContext.sql(claims_query)
    #** Smart Compare
    df = spark.read.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("query",  claims_query).load()

    #df = df.filter(df.probability_score > 0.15) # Threshold set to 15%
    df = df.filter(col("probability_score") > 0.15) # Threshold set to 15%
    df.createOrReplaceTempView("temp_claims_data")
    claimCountQuery = "select client_plan_id, predicted_plan_id, count(*) as claim_count, avg(probability_score) as probability from temp_claims_data group by client_plan_id, predicted_plan_id"
    summaryDF = sqlContext.sql(claimCountQuery)

    summaryDF = summaryDF.withColumn("client_code", lit(client_id))
    summaryDF = summaryDF.withColumn("run_date", lit(run_date))
    
    print("\n\n\n")
    print(summaryDF.show(10, False))
    print(summaryDF.count())
    print("\n\n\n")

    summaryDF.write \
    .format("jdbc").mode("append").options(url = sql_db_url, dbtable = sql_plan_summary_table)\
    .save()
    print("Generating PREDICTION event...")


    # The table expects a colon sign in between
    event_message= '{"client_code":"' + client_id + '" }'
    x=[["PREDICTION", event_message, "PENDING"]]

    cSchema = StructType([StructField("event_type", StringType()),StructField("event_message",StringType()),
                    StructField("status",StringType())])

    eventDF = spark.createDataFrame(x,schema = cSchema) 

    eventDF.write \
    .format("jdbc").mode("append").options(url = sql_db_url, dbtable = sql_event_master_table)\
    .save()

    print("Aggregation Completed Successfully...")
    sc.stop()
    print("plansummary execution finished..")

