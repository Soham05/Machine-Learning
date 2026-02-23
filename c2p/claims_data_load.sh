# Run as: ./claims_data_load.sh /claims_cert_prod/claimscert/claims2plan_pyspark/claims_data_dir/ /data/prod/PBM/ARC/CERT/CLM/PATNT/ prod_config.ini

HDFS_JAR_LOCATION=/data/prod/PBM/ARC/CERT/CLM/PATNT/cvs
#kinit -kt /etc/security/keytabs/certbat.keytab certbat@TEST.CVSCAREMARK.COM

export ENVIRONMENT=prod
DEPLOY_LOCATION=/claims_cert_prod

timestamp=$(date +%s)

unset PYSPARK_DRIVER_PYTHON
unset PYSPARK_DRIVER_PYTHON_OPTS
PYSPARK_PYTHON=/home/certbat/.conda/envs/py36/bin/python

echo $1 # $DEPLOY_LOCATION/claimscert/claims2plan_pyspark/claims_data_dir/
echo $2 # /data/prod/PBM/ARC/CERT/CLM/PATNT/
#**hdfs dfs -put $1/* $2
cp -r $1 $2
path=$3 # prod_config.ini

log=$DEPLOY_LOCATION/claimscert/log/claims_data_load_$timestamp.log
#spark-submit --driver-class-path $DEPLOY_LOCATION/claimscert/jars/mssql-jdbc-7.0.0.jre8.jar --driver-memory 128g --num-executors 128 --executor-cores 5 --executor-memory 128g --conf spark.rpc.message.maxSize=1024 --conf spark.driver.extraJavaOptions=-Xms24g --conf spark.network.timeout=4000s  --packages com.crealytics:spark-excel_2.11:0.13.1 claims_data_load.py  $path  $1 $2>> $log 2>&1

#spark-submit --driver-class-path $DEPLOY_LOCATION/claimscert/jars/mssql-jdbc-7.0.0.jre8.jar,/claims_cert_prod/jars/com.crealytics_spark-excel_2.11-0.9.17.jar --driver-memory 128g --num-executors 128 --executor-cores 5 --executor-memory 128g --conf spark.rpc.message.maxSize=2047 --conf spark.driver.extraJavaOptions=-Xms24g --conf spark.network.timeout=4000s 	.py  $path  $1 $2>> $log 2>&1

spark-submit --driver-class-path $DEPLOY_LOCATION/claimscert/jars/mssql-jdbc-7.0.0.jre8.jar --driver-memory 128g --num-executors 128 --executor-cores 5 --executor-memory 128g --conf spark.network.timeout=1000 --conf spark.debug.maxToStringFields=2000 --conf spark.dynamicAllocation.enabled=true --conf spark.rpc.message.maxSize=2047 --conf spark.shuffle.service.enabled=true   --packages com.crealytics:spark-excel_2.11:0.9.17 claims_data_load.py  $path  $1 $2>> $log 2>&1



echo 'Data loading script finished execution'

chmod 777 $log




