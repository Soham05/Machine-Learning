

export ENVIRONMENT=prod
DEPLOY_LOCATION=/claims_cert_prod

timestamp=$(date +%s)


unset PYSPARK_DRIVER_PYTHON
unset PYSPARK_DRIVER_PYTHON_OPTS
DATE=`date +%Y-%b-%d`

#log=/claims_cert_prod/claimscert/log/c2p_sampled_featurelevel_training_data_$timestamp.log
#spark-submit    --driver-class-path /claims_cert_prod/claimscert/jars/mssql-jdbc-7.0.0.jre8.jar --driver-memory 260g --num-executors 180 --executor-cores 5 --executor-memory 150g --conf spark.rpc.message.maxSize=1024 --conf spark.driver.extraJavaOptions=-Xms30g   c2p_sampled_featurelevel_training_data.py >> $log 2>&1

log1=/claims_cert_prod/claimscert/log/c2p_sampled_featurelevel_training_data_1_$timestamp.log
spark-submit  --driver-class-path /claims_cert_prod/claimscert/jars/mssql-jdbc-7.0.0.jre8.jar  --driver-memory 128g --num-executors 128 --executor-cores 5 --executor-memory 128g --conf spark.network.timeout=1000 --conf spark.debug.maxToStringFields=2000 --conf spark.dynamicAllocation.enabled=true --conf spark.rpc.message.maxSize=2047 --conf spark.shuffle.service.enabled=true  c2p_sampled_featurelevel_training_data_1.py >> $log1 2>&1
chmod 777 $log1

log2=/claims_cert_prod/claimscert/log/c2p_sampled_featurelevel_training_data_2_$timestamp.log
spark-submit  --driver-class-path /claims_cert_prod/claimscert/jars/mssql-jdbc-7.0.0.jre8.jar  --driver-memory 128g --num-executors 128 --executor-cores 5 --executor-memory 128g --conf spark.network.timeout=1000 --conf spark.debug.maxToStringFields=2000 --conf spark.dynamicAllocation.enabled=true --conf spark.rpc.message.maxSize=2047 --conf spark.shuffle.service.enabled=true  c2p_sampled_featurelevel_training_data_2.py >> $log2 2>&1

chmod 777 $log2

log3=/claims_cert_prod/claimscert/log/c2p_sampled_featurelevel_training_data_3_$timestamp.log
spark-submit  --driver-class-path /claims_cert_prod/claimscert/jars/mssql-jdbc-7.0.0.jre8.jar  --driver-memory 128g --num-executors 128 --executor-cores 5 --executor-memory 128g --conf spark.network.timeout=1000 --conf spark.debug.maxToStringFields=2000 --conf spark.dynamicAllocation.enabled=true --conf spark.rpc.message.maxSize=2047 --conf spark.shuffle.service.enabled=true  c2p_sampled_featurelevel_training_data_3.py >> $log3 2>&1



chmod 777 $log3






