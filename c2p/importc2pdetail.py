from pyspark.sql import SparkSession,SQLContext
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
import sys
from configparser import ConfigParser
from pyspark.sql.functions import lit

def importc2pdetail_function(clientId, run_date, config_filename):
    # Import Config Details
    print("Loading configurations from: ", config_filename)
    config_object = ConfigParser()
    config_object.read(config_filename)
    importc2pdetail_config = config_object["IMPORTC2PDETAIL"]

    #hive_db = importc2pdetail_config['hive_db']
    hive_predictions_table = importc2pdetail_config['hive_predictions_table']
    hive_claim_prediction_detail_table = importc2pdetail_config['hive_claim_prediction_detail_table']
    HDFS_PATH = importc2pdetail_config['HDFS_PATH']
	
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

    spark = SparkSession \
    .builder \
    .appName("importc2pdetails") \
    .getOrCreate()


    #spark.sql('use ' + hive_db)
    #**pred_df = spark.sql("select * from " + hive_predictions_table + " where client_id='" + clientId + "' and run_date='" + run_date + "'")
	
    sql_read = "select * from " + hive_predictions_table + " where client_id='" + clientId + "' and run_date='" + run_date + "'"
    pred_df = spark.read.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("query",  sql_read).load()
	
    c2pDF = pred_df.select("transaction_id", "client_plan_id", "predicted_plan_id", "probability_score")
    c2pDF = c2pDF.withColumn("run_date", F.lit(run_date))
    c2pDF = c2pDF.withColumn("run_date", F.trim(F.col("run_date")).cast(StringType()))
    c2pDF = c2pDF.withColumn("transaction_id", F.trim(F.col("transaction_id")).cast(StringType()))
    c2pDF = c2pDF.withColumn("clientId", F.lit(clientId))
    c2pDF = c2pDF.withColumn("clientId", F.trim(F.col("clientId")).cast(StringType()))
    c2pDF = c2pDF.withColumn("client_plan_id", F.trim(F.col("client_plan_id")).cast(StringType()))
    c2pDF = c2pDF.withColumn("predicted_plan_id", F.trim(F.col("predicted_plan_id")).cast(StringType()))

    c2pDF.printSchema()
    print(c2pDF.show(5, False))
    path = HDFS_PATH + "/" + hive_claim_prediction_detail_table + "/client_id=" + clientId + "/run_date=" + run_date
    c2pDF.write.mode('overwrite').parquet(path)
    c2pDF = c2pDF.withColumn("CLIENT_ID", lit(clientId))
    c2pDF = c2pDF.withColumn("RUN_DATE", lit(run_date))
	
    try:
	    delete_query = "DELETE FROM " + hive_claim_prediction_detail_table + " client_id = '" + clientId + "' and run_date = '" + run_date + "'"
	    print ("Delete Query " + delete_query)
	    spark.sparkContext._jvm.net.snowflake.spark.snowflake.Utils.runQuery(sfOptions, delete_query)
    except Exception as e:
	    print("Failed to delete the existing records of the Snowflake table " + str(e))
		
    c2pDF.select("TRANSACTION_ID", "CLIENT_PLAN_ID", "PREDICTED_PLAN_ID", "PROBABILITY_SCORE", "CLIENT_ID", "RUN_DATE").write.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("dbtable", hive_claim_prediction_detail_table).mode('append').save()

    #**print('Adding Partitions...')
    #  c2pDF.write().mode("append").partitionBy("clientId", "run_date").saveAsTable(tablename)
    #**sql = "alter table " + hive_claim_prediction_detail_table + " add if not exists partition( client_id='" + clientId + "', run_date='" + run_date + "') location '" + path + "'"
    #**spark.sql(sql)
    #**spark.stop()
    print("importc2pdetails execution finished..")




