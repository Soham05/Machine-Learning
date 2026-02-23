# Run as: ./predictPlan.sh prod_config.ini

#!/bin/bash

config_filename=$1

DEPLOY_LOCATION=/claims_cert_prod

echo 'Running Prediction script...'

PYSPARK_PYTHON=/home/certbat/.conda/envs/py36/bin/python
#kinit -kt /etc/security/keytabs/certbat.keytab certbat@TEST.CVSCAREMARK.COM


#kinit -kt /etc/security/keytabs/certbat.keytab certbat@TEST.CVSCAREMARK.COM
HDFS_JAR_LOCATION=/data/prod/PBM/ARC/CERT/CLM/PATNT/cvs
cp $DEPLOY_LOCATION/claimscert/jars/cvs-ml-batch-0.0.1-SNAPSHOT.jar $HDFS_JAR_LOCATION


timestamp=$(date +%s)
echo $timestamp
log=$DEPLOY_LOCATION/claimscert/log/prediction_starter_$timestamp.log
spark-submit --driver-class-path $DEPLOY_LOCATION/claimscert/jars/mssql-jdbc-7.0.0.jre8.jar --driver-memory 128g --num-executors 128 --executor-cores 5 --executor-memory 128g --conf spark.rpc.message.maxSize=1024 --conf spark.driver.extraJavaOptions=-Xms24g  prediction_starter.py $config_filename >> $log 2>&1
chmod 777 $log

echo $timestamp

echo "Executed finished for predictPlan.sh"

