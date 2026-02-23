import pyspark 
import pyspark.sql.functions as F


import sys
from configparser import ConfigParser
import pyodbc

from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql import SparkSession


from pyspark.sql.types import StructField,IntegerType, StructType,StringType, DoubleType
from pyspark.sql.functions import trim
from pyspark.sql.functions import current_date
from pyspark.sql.functions import date_format, col
import datetime    

spark = SparkSession.builder.appName("PySpark - c2pClaimData load status").enableHiveSupport().getOrCreate()
sc = spark.sparkContext




config_path=sys.argv[1]
print("Loading configurations from: ",config_path)
config_object = ConfigParser()

config_object.read(config_path) # NOTE - Pass config file location as argument to Shell script
metadata_config = config_object["METADATA"] 

jars_path = metadata_config['jars_path']
sql_server = metadata_config['sql_server']
sql_db = metadata_config['sql_db']
sql_usr = metadata_config['sql_usr']
sql_pwd = metadata_config['sql_pwd']
sql_db_url=metadata_config['sql_db_url']
sql_metadata_table=metadata_config['sql_metadata_table']
hdfs_path=metadata_config['hdfs_path']
mapping_table=metadata_config['mapping_table']
#**hive_db=metadata_config['hive_db']
sf_for_prediction_tbl=metadata_config['sf_for_prediction_tbl']

snowflake_config = config_object["SNOWFLAKE_CRED"] 


	
sfOptions = {}
sfOptions["sfurl"] = snowflake_config['sf_url']
sfOptions["sfuser"] = snowflake_config['sf_user']
sfOptions["sfpassword"] = snowflake_config['sf_password']
sfOptions["sfdatabase"] = snowflake_config['sf_database']
sfOptions["sfschema"] = snowflake_config['sf_schema']
sfOptions["sfrole"] = snowflake_config['sf_role']
sfOptions["sfwarehouse"] = snowflake_config['sf_warehouse']

#**print(sfOptions)

SNOWFLAKE_SOURCE_NAME = snowflake_config['snowflake_source_name']

#**conn = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)
#**conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)

conn = pyodbc.connect('DRIVER=/opt/microsoft/msodbcsql17/lib64/libmsodbcsql-17.7.so.2.1;SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)
cursor = conn.cursor()
cursor.execute("select client_code,claims_data_filename,run_date from " + sql_metadata_table + " where data_load_status='0'") # NOTE - Use config
rows = cursor.fetchall()
client_code_list=[]
file_list=[]
rundate=[]
# Smart Comp
print("Check1")
print(rows)
for row in rows:
    client_code_list.append(row.client_code)
    file_list.append(row.claims_data_filename)
    rundate.append(row.run_date)
kv_pair= dict(zip(file_list, client_code_list))
kv_pair_rd=dict(zip(file_list, rundate))

# NOTE - Use config




for file in file_list:
    print("File name is: ", file)
    try:
	    file_df = spark.read.format("com.crealytics.spark.excel") \
			.option("useHeader", "true") \
			.option("treatEmptyValuesAsNulls", "true") \
			.option("inferSchema", "true") \
			.option("addColorColumns", "False") \
			.option("maxRowsInMemory", 200) \
			.load(hdfs_path + "/" + file)
	    print("Successfully read the file ")
    except Exception as e:
	    print("Failed to read the data " + str(e))
    except:
	    print("Unexpected error:", sys.exc_info()[0])
	    raise Exception ("Failed to read the file ")

		
    print("Processing File name is ",file)
    print(file_df.show(1, False))
    selectColumns = file_df.columns
    for col in selectColumns:
        file_df = file_df.withColumnRenamed(col,col.replace(" ", "_").replace("-","_").replace("/","_").replace("[^\\x00-\\x7F]", "").replace("[^ -~]","").replace(",",""))

    col_rep=file_df.columns
    for col in col_rep:
        file_df = file_df.withColumnRenamed(col, col.lower())
    col_low=file_df.columns
    
    for col_name in col_low:
        file_df = file_df.withColumn(col_name, file_df[col_name].cast(StringType()))

    table=mapping_table  # NOTE - Use config
  
    sqlContext=SQLContext(sc)

    mapping_df  = sqlContext.read.format("jdbc").options(url=sql_db_url,dbtable=table).load() # NOTE - Use config
    cl_cd=kv_pair[file]
    mapping_df=mapping_df.filter(mapping_df.client_code ==cl_cd )
    row_list = mapping_df.select('prediction_column_name').collect()
    prediction_cols = [row.prediction_column_name for row in row_list]
    print("Prediction cols: ", prediction_cols)

    file_df = file_df .select(prediction_cols)
    row_list = mapping_df.select('training_column_name').collect()
    training_cols = [row.training_column_name for row in row_list]



    for i in range(len(prediction_cols)):
        if prediction_cols[i] != training_cols[i]:
            print("Prediction Column name: ", prediction_cols[i])
            print("Training Column name: ", training_cols[i])
            file_df = file_df.withColumnRenamed(prediction_cols[i], training_cols[i])



    client_code=kv_pair[file]
    file_df = file_df.withColumn("client_code",F.lit(client_code))

    file_df = file_df.withColumn("rundate",F.lit(kv_pair_rd[file]))


    print(file_df.show(1, False))



    print('schema here',file_df.printSchema())
     
    write_path=metadata_config['write_path']
    path = write_path + "/client_code="+client_code+ "/rundate="+ kv_pair_rd[file] # NOTE - Use config
    file_df.write.mode("append").parquet(path)
    #**spark.sql('use  ' + hive_db) # NOTE - Use config
	
    file_df.write.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("dbtable", sf_for_prediction_tbl).mode('append').save()
	
    #**sql = "alter table " + hive_for_prediction_tbl + " add if not exists partition( client_code = '" + client_code + "', rundate='" + kv_pair_rd[file] + "') location '" + path + "'" # NOTE - Use config
    #**spark.sql(sql)
	
    #**df = spark.sql("select * from " + hive_for_prediction_tbl + " where rundate='"+kv_pair_rd[file]+"'  and client_code='" + client_code+"'" ) # NOTE - Use config 
    
	
    sql_read = "select * from " + sf_for_prediction_tbl + " where rundate='"+kv_pair_rd[file]+"'  and client_code='" + client_code+"'"
	
    df = spark.read.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("query",  sql_read).load()
		
    print(df.show(1, False))
    
    column_value=df.select('rxclaim_number').collect()
    check_null = [row.rxclaim_number for row in column_value]
    None_count = check_null.count(None)
    if None_count > 10 :
            print("The data load is not completed for..." ,file)
    
    else:
      
        cursor.execute("UPDATE "+sql_metadata_table +" SET data_load_status='1' WHERE client_code='" + kv_pair[file] + "' AND claims_data_filename='" + file + "'AND run_date='"+kv_pair_rd[file]+"'").commit()  
        print("The data load is completed for....." , file)


       
conn.close()





