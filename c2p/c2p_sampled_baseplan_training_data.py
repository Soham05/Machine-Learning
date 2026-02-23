import pyspark
import sys
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import col
from pyspark.sql import functions as F
import datetime
from pyspark.sql import SQLContext
from pyspark.sql.functions import trim


spark = SparkSession     .builder.enableHiveSupport()     .appName("c2p_sampled_baseplan_training_data")  .getOrCreate()

sc=spark.sparkContext
sqlContext=SQLContext(sc)
sfOptions = {
    "sfURL" : "cvscdwprd.us-central1.gcp.snowflakecomputing.com",
    "sfUser" : "APP_PBM_SMRTCMP_PRD",
    "sfPassword" : "Fm$5#_9J",
    "sfDatabase" : "EDP_PBM_APPS_PROD",
    "sfSchema" : "APP_SMRTCMP",
    "sfRole" : "EDP_SMRTCMP_PROD_FUNC_ROLE",
    "sfWarehouse" : "WH_SMRTCMP_APP_PROD"
    }
SNOWFLAKE_SOURCE_NAME = "net.snowflake.spark.snowflake"
print('connection establised')



#now = datetime.datetime.now()
#Partition_Date=now.strftime("%d-%m-%Y")
Partition_Date='13-Oct-2021'


Hive_table='nov20_training'
sampled_hiveTable="c2p_sampled_baseplan_training_data"






#**spark.sql('use claims_cert_prod')
#**claim_df=sqlContext.sql("select * from {} where claim_status = 'P' ".format(Hive_table))
claim_df = spark.read.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("query" , "select * from nov20_training where claim_status = 'P'").load()

for c_name in claim_df.columns:
        claim_df = claim_df.withColumn(c_name, trim(F.col(c_name)))



claim_df=claim_df.filter(F.col('final_plan_code').isin(['INRX-CA01', 'INRX-CA02', 'INRX-DC01', 'INRX-DC02', 'INRX-GA01', 'INRX-GA02', 'INRX-IA01', 'INRX-IN01', 'INRX-IN02', 'INRX-KS01', 'INRX-KS02', 'INRX-KY01', 'INRX-KY02', 'INRX-LA01', 'INRX-MD', 'INRX-MD01', 'INRX-NJ01', 'INRX-NJ02', 'INRX-NJ03', 'INRX-NJ05', 'INRX-NJ06', 'INRX-NV01', 'INRX-NY01', 'INRX-FL01', 'INRX-SC01', 'INRX-TX01', 'INRX-TX02', 'INRX-TX03', 'INRX-VA01', 'INRX-VA02', 'INRX-WA01', 'INRXNYEP1', 'INRXNYEP2', 'IRX-WNY01']))






claim_df=claim_df.withColumn("load_date",F.lit(Partition_Date))

# Sampling code
claim_df.registerTempTable("temp_table")

brmd_baseplans=['INRX-CA01',
 'INRX-CA02',
 'INRX-DC01',
 'INRX-DC02',
 'INRX-GA01',
 'INRX-GA02',
 'INRX-IA01',
 'INRX-IN01',
 'INRX-IN02',
 'INRX-KS01',
 'INRX-KS02',
 'INRX-KY01',
 'INRX-KY02',
 'INRX-LA01',
 'INRX-MD',
 'INRX-MD01',
 'INRX-NJ01',
 'INRX-NJ02',
 'INRX-NJ03',
 'INRX-NJ05',
 'INRX-NJ06',
 'INRX-NV01',
 'INRX-NY01',
 'INRX-FL01',
 'INRX-SC01',
 'INRX-TX01',
 'INRX-TX02',
 'INRX-TX03',
 'INRX-VA01',
 'INRX-VA02',
 'INRX-WA01',
 'INRXNYEP1',
 'INRXNYEP2',
 'IRX-WNY01']



for plan in brmd_baseplans:
        hql_str = " select * FROM  temp_table where  final_plan_code ='"+ plan+"' limit 52500"
        sqlDF = sqlContext.sql(hql_str)
        print(sqlDF.printSchema())
        print(sqlDF.count())
        if(sqlDF.count() > 35000):
                #**sqlDF.write.mode("append").partitionBy("load_date").saveAsTable(sampled_hiveTable)
                sqlDF.write.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("dbtable","c2p_sampled_baseplan_training_data").mode("APPEND").save()



#**spark.sql(select * from sampled_hiveTable).show(2)
spark.read.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("query" , "select * from c2p_sampled_baseplan_training_data limit 2").load()
#**print("Count of training data after sampling: ", spark.sql("select count(*) from {0} where load_date ={1}".format(c2p_sampled_baseplan_training_data,Partition_Date)).count())
print("Count of training data after sampling: ", spark.read.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("query" , "select count(*) from c2p_sampled_baseplan_training_data where load_date = '12-11-2020' ").load())
