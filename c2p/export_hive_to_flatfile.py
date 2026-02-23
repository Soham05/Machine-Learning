# Run as: python export_hive_to_flatfile.py

#spark-submit --driver-memory 128g --num-executors 128 --executor-cores 5 --executor-memory 128g --conf spark.network.timeout=1000 --conf spark.debug.maxToStringFields=2000 --conf spark.dynamicAllocation.enabled=true --conf spark.rpc.message.maxSize=2047 --conf spark.shuffle.service.enabled=true --conf spark.driver.maxResultSize=0 export_hive_to_flatfile.py

#!/usr/bin/env python
# coding: utf-8


#------------------------------------------------------------------------------------------------------------------------------------------#

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
#from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import SparkContext, SparkConf, SQLContext
import pyspark
import sys, os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import isnan, when, count, col, countDistinct 
from pyspark.sql.types import StringType
#from pyspark.sql import SQLContext
#from pyspark.sql.types import * 
from pathlib import Path
from pyspark import SparkContext
import pandas as pd

import snowflake.connector

#**spark = SparkSession.builder.appName("Hive Export Script").config("spark.driver.memory", "64g").config("spark.driver.maxResultSize", "8g").enableHiveSupport().getOrCreate()

#spark = SparkSession.builder.appName("Snowflake Export Script").config("spark.driver.memory", "128g").config("spark.driver.maxResultSize", "0").config("num-executors", "128").config("executor-cores", "5").config("executor-memory", "128g").config("spark.network.timeout", "1000").config("spark.debug.maxToStringFields", "2000").config("spark.dynamicAllocation.enabled",True).config("spark.rpc.message.maxSize", "2047").config("spark.shuffle.service.enabled", True). getOrCreate()

#spark = SparkSession.builder.appName("Snowflake Export Script").getOrCreate()

#sc = spark.sparkContext
#sqlContext=SQLContext(sc)
#**sqlContext.sql('USE claims_cert_prod')


conn = snowflake.connector.connect( \
		url="cvscdwprd.us-central1.gcp.snowflakecomputing.com", \
                user="APP_PBM_SMRTCMP_PRD", \
                password="Fm$5#_9J", \
		account="cvscdwprd.us-central1.gcp", \
                warehouse="WH_SMRTCMP_APP_PROD", \
                database="EDP_PBM_APPS_PROD", \
		role="EDP_SMRTCMP_PROD_FUNC_ROLE", \
                schema="APP_SMRTCMP" \
                )

cur = conn.cursor()

print("Reading data from SNOWFLAKE table...")

#df = sqlContext.sql("SELECT * FROM c2p_sampled_featurelevel_training_data WHERE load_date = '25-12-2020'")
#**df = sqlContext.sql("SELECT * FROM nov20_training WHERE rundate = '12-11-2020' and claim_status = 'P'")

#**Smart Compare
sql_read = "SELECT * FROM nov20_training WHERE rundate = '12-11-2020' and claim_status = 'P'"
#df = spark.read.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("query",  sql_read).load()
cur.execute(sql_read)
dat = cur.fetchall()
col_list = [col.name for col in cur.description]
df_csv = pd.DataFrame(dat, columns=col_list)

#------------------------------------------------------------------------------------------------------------------------------------------#

#print(df.show(2, False))
print(df_csv.head(2))
print('debug:',len(df_csv))
#------------------------------------------------------------------------------------------------------------------------------------------#

filename = r'/claims_cert_prod/claimscert/claims2plan_featurelevel/training_data/c2p_full_claims_data.csv'
#filename = r'/claims_cert_prod/claimscert/claims2plan_featurelevel/training_data/c2p_sampled_featurelevel_training_data_25-12-2020.csv'
#df_csv = df.toPandas()
df_csv.to_csv(filename, header = True, index = False)

#df.coalesce(1).write.format('com.databricks.spark.csv').option("sep", "\t").mode('overwrite').save('/claims_cert_prod/claimscert/claims2plan_featurelevel/temp/training_data/', header = 'true')
#df.coalesce(1).write.format('com.databricks.spark.csv').mode('overwrite').save('/claims_cert_prod/claimscert/claims2plan_featurelevel/temp/training_data/', header = 'true')

#------------------------------------------------------------------------------------------------------------------------------------------#

#sc.stop()


#------------------------------------------------------------------------------------------------------------------------------------------#

# Change permission of the file created with chmod 777.
# command: cp /claims_cert_prod/claimscert/claims2plan_featurelevel/temp/training_data/part*.csv /claims_cert_prod/claimscert/claims2plan_featurelevel/training_data/c2p_full_claims_data_2.csv
# Command:  chmod -R 777 /claims_cert_prod/claimscert/claims2plan_featurelevel/*


#------------------------------------------------------------------------------------------------------------------------------------------#


