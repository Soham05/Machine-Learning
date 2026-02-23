# Run as: ./trainModel_fl.sh prod_fl_config.ini


echo 'Running Feature Level Model Training script...'

config_filename=$1
DEPLOY_LOCATION=/claims_cert_prod
chmod -R 777 $DEPLOY_LOCATION/claimscert/claims2plan_featurelevel/*

source activate py36

timestamp=$(date +%s)
echo $timestamp

#log=$DEPLOY_LOCATION/claimscert/log/training_fl_$timestamp.log
#python training_fl.py $config_filename >> $log 2>&1
#chmod 777 $log

python training_fl.py $config_filename
source deactivate py36
timestamp=$(date +%s)
echo $timestamp


echo "Executed successfully."
echo ""
echo "Setting appropriate permissions for new files created..."

echo " "

echo " "
echo $PWD
echo " "
 
chmod -R 777 $DEPLOY_LOCATION/claimscert/claims2plan_featurelevel/*

echo "Permissions updated."
echo ""
echo "Execution of trainModel_fl.sh finished."

