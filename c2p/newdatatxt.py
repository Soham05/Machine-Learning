import pyspark
import numpy as np

import sys

from pyspark.sql import sparksession
from pyspark.sql import sqlcontext
from pyspark.sql.types import *
from pyspark.sql.functions import col
import pyspark.sql.functions as f

from pyspark.sql.functions import trim
import datetime
from pyspark.sql import sqlcontext

import pandas as pd

spark = sparksession     .builder.enablehivesupport()     .appname("newdatatxt_training")  .getorcreate()

sc=spark.sparkcontext
sqlcontext=sqlcontext(sc)
sfoptions = {
    "sfurl" : "cvscdwprd.us-central1.gcp.snowflakecomputing.com",
    "sfuser" : "APP_PBM_SMRTCMP_PRD",
    "sfpassword" : "Fm$5#_9J",
    "sfdatabase" : "EDP_PBM_APPS_PROD",
    "sfschema" : "APP_SMRTCMP",
    "sfwarehouse" : "WH_SMRTCMP_APP_PROD"
    }
snowflake_source_name = "net.snowflake.spark.snowflake"


now = datetime.datetime.now()
formatteddate=now.strftime("%d-%m-%y")

date='12-11-2020'

file_name='plan_file'
path='/data/prod/pbm/arc/cert/clm/patnt/training_data_dir/'+file_name+ '.' +'txt'

file_df=spark.read.option("delimiter", "^").csv(path)

u=['rxclaim_number','claim_seq','claim_status','submit_date','pharmacy_id4','rxnumber','fill_date','fill_number','cob_indicator','bin','pcn','rxgroup','submitted_cardholder_id','submitted_person_code','submitted_date_of_birth','submitted_patient_gender','submitted_patient_rel','submitted_other_coverage_code','submitted_patient_location','quantity_dispensed','days_supply','compound_code','product_id/ndc','gpi','daw/psc_code','written_date','rx_origin_code','submitted_calarification_code','submitted_usual_and_customary_amount','submitted_diagnosis_qual','submitted_diagnosis_code','sumitted_prescriber_id','prior_auth_reason_code','prior_auth_number','network','member_id','carrier_id','account_id','group_id','bpg_carrier','final_plan_code','final_drug_status','submitted_ingredient_cost','submitted_gross_amount_due','reimburesement_flag','claim_origination_flag','account_name','member_last_name','member_first_name','member_family_type','member_relationship_code','local_message','accum_detail','pa_layer','settlement','pharmacy_id55','pha_pharmacythru_date','pha_pharmacyname,_full','pha_pharmacyfrom_date','medicaid_rebate','max_amount_basis','max_amount_basis_ind','npi_id','desi_code','maintenance_drug_ind','dea_code','drug_strength_','metric_strength','gcn','multi_source_ind','route_of_admin','rx/otc_indicator','3rdpartyexceptioncode','product_description','golive','medicare_contract_id','dur_counter_consolidated','dur_reason_consolidated','dur_significance_consolidated','reversal_date','lob_carrier_description','listdetail','splty_flg','ingredient_cost_client','remaining_oop_amount','incentive_fee_','amount_exceeded_per_benefit','amount_attr_to_sales_tax','remaining_deductible_amount','amount_applied_per_deductible89','individual_oop_ptd_','family_oop_ptd_','individual_ded_ptd','family_ded_ptd','copay_amount','amount_applied_per_deductible95','patient_pay_amount','total_amount_due','dispensing_fee','lics_subsidy_amount','troop_amount_this_claim','rxclaim_gdca_amount','drug_spend_pat_pay_amt','oop_gap_pat_pay_amt','cat_copay_with_opar','salex_tax_perc_paid','salex_tax_basis_paid','client_cost_type','client_patient_schedule_name','client_price_schedule_name','other_payer_amount_paid','total_sales_tax_amt','preferred_ndc_list_id_primary','preferred_gpi_list_id_primary','preferred_ndc_list_id_secondary','preferred_gpi_list_id_secondary','dtd_amt_applied_per_dedu','drugspend_patient_pay_amount','gap_patient_pay_amount','cat_patient_pay_amount','tf_tag','tf_letter_tag','tf_combo_edit_tag','egwp_claim_indicator','egwp_plan_indicator','benefit_beginning_phase_ind','benefit_end_phase_ind','contingent_therapy_flag','contingent_therapy_status','zero_balance_dollar_amount','zero_balance_dollar_indicator','smart_pa_indicator','dispensing_fee_count_for_multiplier_logic','medb_claim_indicator','govt_claim_type','primary_edit_flag','med_d_drug_indicator','adjudication_upofront_indicator','coverage_gap_amount','tpm_ignore_pa_status','tpm_pa_current_drug_status','product_selection_penalty_amt','drug_price_tier','hra_amount','except_override_tag','ltc_override_ind','skip_deductible_flag','admin_fee_type','number_of_mchoice_claims_allowd','ads_scp_tag','reject1','reject2','reject3','xref_plan_code','rundate','runtime','patres','pharsrvtyp','formulary','daysup1','thresh1','daysup2','thresh2','daysup3','thresh3','certid','pa_ind','rflallwd'
]


x=file_df.columns
for i in range(len(u)):
    file_df = file_df.withcolumnrenamed(x[i], u[i])
selectcolumns = file_df.columns

for col in selectcolumns:
    file_df = file_df.withcolumnrenamed(col,col.replace(" ", "_").replace("-","_").replace("/","_").replace("[^\\x00-\\x7f]", "").replace("[^ -~]","").replace(",",""))

col_rep=file_df.columns
for col in col_rep:
    file_df = file_df.withcolumnrenamed(col, col.lower())




file_df= file_df.withcolumn("rundate",f.lit(date))

for c_name in file_df.columns:
        file_df = file_df.withcolumn(c_name, trim(f.col(c_name)))



path = "/data/prod/pbm/arc/cert/clm/patnt/nov20_training" +  "/rundate="+ date
file_df.write.mode("append").parquet(path)

file_df.write.format(snowflake_source_name).options(**sfoptions).option("dbtable","nov20_training").mode('append').save()

print(file_df.show())
print(file_df.count())
print("success")


