import pyspark
import numpy as np

import sys

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import * 
from pyspark.sql.functions import col
import pyspark.sql.functions as F

from pyspark.sql.functions import trim
import datetime
from pyspark.sql import SQLContext

import pandas as pd

spark = SparkSession     .builder.enableHiveSupport()     .appName("nov20_training")  .getOrCreate()

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
snowflake_source_name = "net.snowflake.spark.snowflake"


now = datetime.datetime.now()
formattedDate=now.strftime("%d-%m-%Y")

Date='12-11-2020'

#file_df = spark.read.format("com.crealytics.spark.excel").option("Header", "true").option("dataAddress", "Sheet2!").option("inferSchema", "true").load(r'/data/prod/PBM/ARC/CERT/CLM/PATNT/training_data_dir/PlanPredicitions_Medicaid_claims_F.xlsx')
file_df=spark.read.option("delimiter", "^").csv(r'/data/prod/PBM/ARC/CERT/CLM/PATNT/training_data_dir/PLAN_FILC.TXT')

u=['RxClaim_Number','Claim_Seq','Claim_Status','Submit_Date','Pharmacy_Id4','RXNumber','Fill_Date','Fill_Number','COB_Indicator','BIN','PCN','RXGROUP','Submitted_Cardholder_ID','Submitted_Person_Code','Submitted_Date_of_Birth','Submitted_Patient_Gender','Submitted_Patient_Rel','Submitted_Other_Coverage_Code','Submitted_Patient_Location','Quantity_Dispensed','Days_Supply','Compound_Code','Product_Id/NDC','GPI','DAW/PSC_Code','Written_Date','RX_Origin_Code','Submitted_Calarification_Code','Submitted_Usual_and_Customary_Amount','Submitted_Diagnosis_Qual','Submitted_Diagnosis_Code','Sumitted_Prescriber_ID','Prior_Auth_Reason_Code','Prior_Auth_Number','Network','Member_Id','Carrier_ID','Account_Id','Group_Id','BPG_Carrier','Final_Plan_Code','Final_Drug_Status','Submitted_Ingredient_Cost','Submitted_Gross_Amount_Due','Reimburesement_Flag','Claim_Origination_Flag','Account_Name','Member_Last_Name','Member_First_Name','Member_Family_Type','Member_Relationship_Code','Local_Message','Accum_detail','PA_LAYER','SETTLEMENT','Pharmacy_Id55','PHA_PharmacyThru_Date','PHA_PharmacyName,_Full','PHA_PharmacyFrom_Date','Medicaid_Rebate','Max_Amount_Basis','Max_Amount_Basis_Ind','NPI_ID','DESI_Code','Maintenance_Drug_Ind','DEA_Code','Drug_Strength_','Metric_Strength','GCN','Multi_Source_Ind','Route_of_Admin','RX/OTC_Indicator','3rdPartyExceptionCode','Product_Description','GOLIVE','Medicare_Contract_Id','DUR_Counter_consolidated','DUR_Reason_consolidated','DUR_Significance_consolidated','Reversal_Date','LOB_Carrier_Description','LISTDETAIL','SPLTY_FLG','Ingredient_Cost_Client','Remaining_OOP_Amount','Incentive_Fee_','Amount_Exceeded_Per_benefit','Amount_Attr_to_Sales_Tax','Remaining_Deductible_Amount','Amount_Applied_Per_Deductible89','Individual_OOP_PTD_','Family_OOP_PTD_','Individual_DED_PTD','Family_DED_PTD','Copay_Amount','Amount_Applied_Per_Deductible95','Patient_Pay_Amount','Total_Amount_Due','Dispensing_Fee','LICS_Subsidy_Amount','TROOP_Amount_this_claim','RxClaim_GDCA_Amount','Drug_Spend_Pat_Pay_Amt','OOP_Gap_Pat_Pay_Amt','CAT_Copay_With_OPAR','Salex_Tax_Perc_Paid','Salex_Tax_Basis_Paid','Client_Cost_Type','Client_Patient_Schedule_Name','Client_Price_Schedule_Name','Other_Payer_Amount_Paid','Total_Sales_Tax_Amt','Preferred_NDC_List_ID_Primary','Preferred_GPI_List_ID_Primary','Preferred_NDC_List_ID_Secondary','Preferred_GPI_List_ID_Secondary','DTD_Amt_Applied_Per_Dedu','Drugspend_Patient_Pay_Amount','GAP_Patient_Pay_Amount','CAT_Patient_Pay_Amount','TF_Tag','TF_Letter_Tag','TF_Combo_Edit_Tag','EGWP_Claim_Indicator','EGWP_Plan_Indicator','Benefit_Beginning_Phase_Ind','Benefit_End_Phase_Ind','Contingent_Therapy_Flag','Contingent_Therapy_Status','Zero_balance_Dollar_Amount','Zero_balance_Dollar_Indicator','Smart_PA_Indicator','Dispensing_Fee_Count_for_multiplier_logic','MedB_Claim_Indicator','Govt_Claim_Type','Primary_Edit_Flag','Med_D_Drug_Indicator','Adjudication_Upofront_Indicator','Coverage_Gap_Amount','TPM_Ignore_PA_Status','TPM_PA_Current_Drug_Status','Product_Selection_Penalty_Amt','Drug_Price_Tier','HRA_Amount','Except_Override_Tag','LTC_Override_Ind','Skip_Deductible_Flag','Admin_Fee_Type','Number_of_Mchoice_Claims_allowd','ADS_SCP_Tag','REJECT1','REJECT2','REJECT3','XREF_Plan_Code','RUNDATE','RUNTIME','PATRES','PHARSRVTYP','FORMULARY','DAYSUP1','THRESH1','DAYSUP2','THRESH2','DAYSUP3','THRESH3','CERTID','pa_ind','rflallwd'
]

x=file_df.columns
for i in range(len(u)):
    file_df = file_df.withColumnRenamed(x[i], u[i])
selectColumns = file_df.columns

for col in selectColumns:
    file_df = file_df.withColumnRenamed(col,col.replace(" ", "_").replace("-","_").replace("/","_").replace("[^\\x00-\\x7F]", "").replace("[^ -~]","").replace(",",""))

col_rep=file_df.columns
for col in col_rep:
    file_df = file_df.withColumnRenamed(col, col.lower())



#claim_df = file_df.join(brmd,"final_plan_code","inner" )
#claim_df = file_df.join(brmd, file_df.Final_Plan_Code == brmd.plan_id)
file_df= file_df.withColumn("rundate",F.lit(Date)) 

for c_name in file_df.columns:
        file_df = file_df.withColumn(c_name, trim(F.col(c_name)))



path = "/data/prod/PBM/ARC/CERT/CLM/PATNT/nov20_training" +  "/rundate="+ Date
file_df.write.mode("append").parquet(path)
file_df.write.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("dbtable","nov20_training").mode("APPEND").save()

              
print(file_df.show())
print("Success")
