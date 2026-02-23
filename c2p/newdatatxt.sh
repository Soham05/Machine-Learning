

#kinit -kt /etc/security/keytabs/certbat.keytab certbat@TEST.CVSCAREMARK.COM
export ENVIRONMENT=prod
DEPLOY_LOCATION=/claims_cert_prod

timestamp=$(date +%s)


unset PYSPARK_DRIVER_PYTHON
unset PYSPARK_DRIVER_PYTHON_OPTS
DATE=`date +%Y-%b-%d`

log=/claims_cert_prod/claimscert/log/nov20_$timestamp.log
spark-submit --driver-class-path /claims_cert_prod/claimscert/jars/mssql-jdbc-7.0.0.jre8.jar --driver-memory 260g --num-executors 180 --executor-cores 5 --executor-memory 150g --conf spark.rpc.message.maxSize=1024 --conf spark.driver.extraJavaOptions=-Xms30g  --packages com.crealytics:spark-excel_2.11:0.13.1  newdatatxt.py >> $log 2>&1




chmod 777 $log






