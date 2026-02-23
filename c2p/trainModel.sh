# Run as: ./trainModel.sh prod_config.ini

#!/bin/bash
echo 'Running Model Training script...'


export PATH=$PATH:/usr/local/spark/bin
source ~/.bashrc


config_filename=$1
DEPLOY_LOCATION=/claims_cert_prod

#**PYSPARK_PYTHON=/home/certbat/.conda/envs/py36/bin/python
#**kinit -kt /etc/security/keytabs/certbat.keytab certbat@TEST.CVSCAREMARK.COM

timestamp=$(date +%s)
echo $timestamp
log=$DEPLOY_LOCATION/claimscert/log/training_model_$timestamp.log
spark-submit --driver-memory 128g --num-executors 128 --executor-cores 5 --executor-memory 128g training_model.py $config_filename >> $log 2>&1
chmod 777 $log
echo $timestamp

echo "Executed successfully."
echo ""
echo "Setting appropriate permissions for new files created..."

echo " "
#cd ../pickle/
echo " "

echo " "
echo $PWD
echo " "
 
chmod -R 777 $DEPLOY_LOCATION/claimscert/claims2plan_pyspark/pickle/model/
chmod -R 777 $DEPLOY_LOCATION/claimscert/claims2plan_pyspark/pickle/pipeline/

echo "Permissions updated."
echo ""
echo "Execution of trainModel.sh finished."


