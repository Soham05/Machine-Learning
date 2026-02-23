import pyspark
import sys
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import * 
from pyspark.sql.functions import col
from pyspark.sql import functions as F
import datetime
from pyspark.sql import SQLContext
from pyspark.sql.functions import trim


spark = SparkSession     .builder  .appName("c2p_sampled_featurelevel_training_data")  .getOrCreate()

sc=spark.sparkContext
sqlContext=SQLContext(sc)

print('connection establised')



now = datetime.datetime.now()
Partition_Date=now.strftime("%d-%m-%Y")
#Partition_Date='25-12-2020'
training_rundate='12-11-2020'
#brmd_table='c2p_prod.brmd_master'
#**Smart compare changes
brmd_table='c2p_prod.brmd_master'
#**Hive_table='claims_cert_prod.nov20_training'
#**sampled_hiveTable='claims_cert_prod.c2p_sampled_featurelevel_training_data'

Hive_table='nov20_training'
sampled_hiveTable='c2p_sampled_featurelevel_training_data'

sfOptions = {
    "sfurl" : "cvscdwprd.us-central1.gcp.snowflakecomputing.com",
    "sfuser" : "APP_PBM_SMRTCMP_PRD",
    "sfpassword" : "Fm$5#_9J",
    "sfdatabase" : "EDP_PBM_APPS_PROD",
    "sfschema" : "APP_SMRTCMP",
    "sfrole" : "EDP_SMRTCMP_PROD_FUNC_ROLE",
    "sfwarehouse" : "WH_SMRTCMP_APP_PROD"
    }
SNOWFLAKE_SOURCE_NAME = "net.snowflake.spark.snowflake"

table='(select * from {}) as brmd'.format(brmd_table)
#**brmd1 = sqlContext.read.format("jdbc").options(url='jdbc:sqlserver://PAZ1ABSDPW1V;databaseName=CLAIMS_CERT_PROD;user=bdusr;password=Welcome#123',dbtable=table).load()

#**Smart compare changes
brmd1 = sqlContext.read.format("jdbc").options(url='jdbc:sqlserver://10.124.68.111;databaseName=CLAIMS_CERT_PROD;user=bdusr;password=Welcome#123',dbtable=table).load()
brmd1=brmd1.filter(brmd1.status=='A')



brmd = brmd1.withColumnRenamed("plan_id","XREF_Plan_Code")


tf=['base_plan_id','XREF_Plan_Code','copay_waiver_drug_specific_copays_tab_0',
 'pharmacy_participation_340b',
 'allow_unbreakable_packages_for_emergency_supply',
 'allow_unbreakable_packages_for_transition_fill',
 'allow_which_occ_other_coverage_codes',
 'apply_cms_labeler_rules',
 'apply_ndc5_standard_rebate_exception_list',
 'bulk_chemicals_covered',
 'compound_post_max_dollar_limit_reject_messaging',
 'compounds_max_dollar_limit',
 'controlled_substances_cii_ciii_civ_cv_refill_threshold_',
 'copay_exceptions',
 'courtesy_grace_fill_limit',
 'courtesy_grace_fills_if_exclusive',
 'cover_multi_ingredient_compounds',
 'custom_post_oop_met_copay',
 'custom_specialty_copay',
 'cvs_health_vaccine_program_applies',
 'desi_exclusions',
 'do_specialty_copays_mirror_retail',
 'does_emergency_fill_claims_history_count_towards_clinical_rule_lookbacks',
 'does_split_fill_apply',
 'does_the_plan_have_prescriber_gold_carding',
 'does_transition_fill_claims_history_count_towards_clinical_rule_lookbacks',
 'dur_services',
 'emergency_fill_lookback_gpi',
 'emergency_fill_max__of_fills',
 'emergency_fill_max_day_supply',
 'emergency_fill_allowed',
 'emergency_fill_applies_to_age_gender_rejects_',
 'emergency_fill_applies_to_exclusions',
 'emergency_fill_applies_to_pa_rejects',
 'emergency_fill_applies_to_qty_limit_rejects',
 'emergency_fill_applies_to_reverse_step_rejects',
 'emergency_fill_applies_to_smart_edit_rejects_',
 'emergency_fill_applies_to_step_therapy_rejects',
 'emergency_fill_applies_to_what_delivery_system',
 'emergency_fill_applies_to_which_members',
 'emergency_fill_copay',
 'emergency_fill_history_review_lookback_window_days',
 'emergency_fill_limit_messaging',
 'exclusive_or_open_specialty',
 'how_does_the_split_fill_copay_apply_',
 'if_split_fill_applies_what_drug_categories',
 'is_daw5_allowed_and_if_so_at_what_copay',
 'is_there_a_copay_amount_after_mab_is_met',
 'is_there_a_copay_amount_after_oop_is_met',
 'mail_claim_cost_0_to_10',
 'mail_claim_cost_10_01_to_25',
 'mail_claim_cost_25_01_to_50',
 'mail_claim_cost_50_01_or_more',
 'mail_covered',
 'mail_generic_copayment',
 'mail_non_preferred_brand_copayment',
 'mail_order_refill_threshold_',
 'mail_preferred_brand_copayment',
 'max__of_fills',
 'max_dollar_mail_non_specialty',
 'max_dollar_retail_paper_claims_non_specialty',
 'max_dollarspecialty',
 'msb_drugs_psc_exlcusions',
 'msb_drugs_psc_exlcusions_reject_messaging',
 'multi_ingredient_compounds_prior_authorization',
 'natural_disaster_recovery_scc_13',
 'non_specialty_post_max_dollar_limit_reject_messaging',
 'oop_4th_quarter_carryover',
 'oop_amount',
 'oop_type',
 'ophthalmic_refill_threshold_',
 'other_drugs_not_eliglible_for_emergency_fill',
 'other_drugs_not_eliglible_for_transition_fill',
 'paper_claim_refill_threshold_',
 'rebate_supplemental_list_applies',
 'reject_claim_after_mab_met',
 'retail_claim_cost_0_to_10',
 'retail_claim_cost_10_01_to_25',
 'retail_claim_cost_25_01_50',
 'retail_claim_cost_50_01_or_more',
 'retail_claim_refill_threshold_',
 'retail_generic_copayment',
 'retail_non_preferred_brand_copayment',
 'retail_preferred_brand_copayment',
 'scriptsync_cvs_medication_synchronization',
 'specialty_max_ds',
 'specialty_network',
 'specilaty_refill_threshold',
 'split_fill_for_first_15ds_and_30ds',
 'tpe_exclusions',
 'transition_fill_lookback_gpi',
 'transition_fill_max_cumulative_day_supply',
 'transition_fill_allowed',
 'transition_fill_applies_to_age_gender_rejects',
 'transition_fill_applies_to_exclusions',
 'transition_fill_applies_to_pa_rejects',
 'transition_fill_applies_to_qty_limit_rejects',
 'transition_fill_applies_to_reverse_step_rejects',
 'transition_fill_applies_to_step_therapy_rejects',
 'transition_fill_applies_to_what_delivery_system',
 'transition_fill_applies_to_which_members',
 'transition_fill_history_review_lookback_window_days',
 'transition_fill_limit_messaging_']



brmd =brmd.select(tf)




for c_name in brmd.columns:
        brmd = brmd.withColumn(c_name, trim(F.col(c_name)))


#**spark.sql('use claims_cert_prod')
#**claim_df=sqlContext.sql('select * from {0} where claim_status = "P" and rundate= "{1}"'.format(Hive_table,training_rundate))

sql_read = "select * from {0} where claim_status = 'P' and rundate= '{1}'".format(Hive_table,training_rundate)
claim_df = spark.read.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("query",  sql_read).load()

claim_df=claim_df.filter(F.col('final_plan_code').isin(['INRX-CA01', 'INRX-CA02', 'INRX-DC01', 'INRX-DC02', 'INRX-GA01', 'INRX-GA02', 'INRX-IA01', 'INRX-IN01', 'INRX-IN02', 'INRX-KS01', 'INRX-KS02', 'INRX-KY01', 'INRX-KY02', 'INRX-LA01', 'INRX-MD', 'INRX-MD01', 'INRX-NJ01', 'INRX-NJ02', 'INRX-NJ03', 'INRX-NJ05', 'INRX-NJ06', 'INRX-NV01', 'INRX-NY01', 'INRX-FL01', 'INRX-SC01', 'INRX-TX01', 'INRX-TX02', 'INRX-TX03', 'INRX-VA01', 'INRX-VA02', 'INRX-WA01', 'INRXNYEP1', 'INRXNYEP2', 'IRX-WNY01']))



df_hive = claim_df.join(brmd,"XREF_Plan_Code","inner" )

df_hive=df_hive.withColumn("load_date",F.lit(Partition_Date))

df_hive=df_hive.withColumnRenamed("DRUG_STRENGTH_", "DRUG_STRENGTH")
df_hive=df_hive.withColumnRenamed("INCENTIVE_FEE_", "INCENTIVE_FEE")
df_hive=df_hive.withColumnRenamed("INDIVIDUAL_OOP_PTD_", "INDIVIDUAL_OOP_PTD")
df_hive=df_hive.withColumnRenamed("FAMILY_OOP_PTD_", "FAMILY_OOP_PTD")
df_hive=df_hive.withColumnRenamed("controlled_substances_cii_ciii_civ_cv_refill_threshold_", "controlled_substances_cii_ciii_civ_cv_refill_threshold")
df_hive=df_hive.withColumnRenamed("emergency_fill_applies_to_age_gender_rejects_", "emergency_fill_applies_to_age_gender_rejects")
df_hive=df_hive.withColumnRenamed("emergency_fill_applies_to_smart_edit_rejects_", "emergency_fill_applies_to_smart_edit_rejects")
df_hive=df_hive.withColumnRenamed("how_does_the_split_fill_copay_apply_", "how_does_the_split_fill_copay_apply")
df_hive=df_hive.withColumnRenamed("ophthalmic_refill_threshold_", "ophthalmic_refill_threshold")
df_hive=df_hive.withColumnRenamed("retail_claim_refill_threshold_", "retail_claim_refill_threshold")
df_hive=df_hive.withColumnRenamed("transition_fill_limit_messaging_", "transition_fill_limit_messaging")
df_hive=df_hive.withColumnRenamed("mail_order_refill_threshold_", "mail_order_refill_threshold")
df_hive=df_hive.withColumnRenamed("paper_claim_refill_threshold_", "paper_claim_refill_threshold")

# Sampling code
df_hive.registerTempTable("temp_table")

brmd_baseplans=['INRX-CA01',
 'INRX-CA02',
 'INRX-DC01',
 'INRX-DC02',
 'INRX-GA01',
 'INRX-GA02',
 'INRX-IA01',
 'INRX-IN01',
 'INRX-IN02']
 
"""'INRX-KS01',
 'INRX-KS02',
 'INRX-KY01',
 'INRX-KY02',
 'INRX-LA01',
 'INRX-MD',
 'INRX-MD01',
 'INRX-NJ01',
 'INRX-NJ02',
 'INRX-NJ03',
 'INRX-NJ05',
 'INRX-NJ06',
 'INRX-NV01',
 'INRX-NY01',
 'INRX-FL01',
 'INRX-SC01',
 'INRX-TX01',
 'INRX-TX02',
 'INRX-TX03',
 'INRX-VA01',
 'INRX-VA02',
 'INRX-WA01',
 'INRXNYEP1',
 'INRXNYEP2',
 'IRX-WNY01']"""


final_cols = ["xref_plan_code",
"rxclaim_number",
"claim_seq",
"claim_status",
"submit_date",
"pharmacy_id4",
"rxnumber",
"fill_date",
"fill_number",
"cob_indicator",
"bin",
"pcn",
"rxgroup",
"submitted_cardholder_id",
"submitted_person_code",
"submitted_date_of_birth",
"submitted_patient_gender",
"submitted_patient_rel",
"submitted_other_coverage_code",
"submitted_patient_location",
"quantity_dispensed",
"days_supply",
"compound_code",
"product_id_ndc",
"gpi",
"daw_psc_code",
"written_date",
"rx_origin_code",
"submitted_calarification_code",
"submitted_usual_and_customary_amount",
"submitted_diagnosis_qual",
"submitted_diagnosis_code",
"sumitted_prescriber_id",
"prior_auth_reason_code",
"prior_auth_number",
"network",
"member_id",
"carrier_id",
"account_id",
"group_id",
"bpg_carrier",
"final_plan_code",
"final_drug_status",
"submitted_ingredient_cost",
"submitted_gross_amount_due",
"reimburesement_flag",
"claim_origination_flag",
"account_name",
"member_last_name",
"member_first_name",
"member_family_type",
"member_relationship_code",
"local_message",
"accum_detail",
"pa_layer",
"settlement",
"pharmacy_id55",
"pha_pharmacythru_date",
"pha_pharmacyname_full",
"pha_pharmacyfrom_date",
"medicaid_rebate",
"max_amount_basis",
"max_amount_basis_ind",
"npi_id",
"desi_code",
"maintenance_drug_ind",
"dea_code",
"drug_strength",
"metric_strength",
"gcn",
"multi_source_ind",
"route_of_admin",
"rx_otc_indicator",
"ARDPARTYEXCEPTIONCODE",
"product_description",
"golive",
"medicare_contract_id",
"dur_counter_consolidated",
"dur_reason_consolidated",
"dur_significance_consolidated",
"reversal_date",
"lob_carrier_description",
"listdetail",
"splty_flg",
"ingredient_cost_client",
"remaining_oop_amount",
"incentive_fee",
"amount_exceeded_per_benefit",
"amount_attr_to_sales_tax",
"remaining_deductible_amount",
"amount_applied_per_deductible89",
"individual_oop_ptd",
"family_oop_ptd",
"individual_ded_ptd",
"family_ded_ptd",
"copay_amount",
"amount_applied_per_deductible95",
"patient_pay_amount",
"total_amount_due",
"dispensing_fee",
"lics_subsidy_amount",
"troop_amount_this_claim",
"rxclaim_gdca_amount",
"drug_spend_pat_pay_amt",
"oop_gap_pat_pay_amt",
"cat_copay_with_opar",
"salex_tax_perc_paid",
"salex_tax_basis_paid",
"client_cost_type",
"client_patient_schedule_name",
"client_price_schedule_name",
"other_payer_amount_paid",
"total_sales_tax_amt",
"preferred_ndc_list_id_primary",
"preferred_gpi_list_id_primary",
"preferred_ndc_list_id_secondary",
"preferred_gpi_list_id_secondary",
"dtd_amt_applied_per_dedu",
"drugspend_patient_pay_amount",
"gap_patient_pay_amount",
"cat_patient_pay_amount",
"tf_tag",
"tf_letter_tag",
"tf_combo_edit_tag",
"egwp_claim_indicator",
"egwp_plan_indicator",
"benefit_beginning_phase_ind",
"benefit_end_phase_ind",
"contingent_therapy_flag",
"contingent_therapy_status",
"zero_balance_dollar_amount",
"zero_balance_dollar_indicator",
"smart_pa_indicator",
"dispensing_fee_count_for_multiplier_logic",
"medb_claim_indicator",
"govt_claim_type",
"primary_edit_flag",
"med_d_drug_indicator",
"adjudication_upofront_indicator",
"coverage_gap_amount",
"tpm_ignore_pa_status",
"tpm_pa_current_drug_status",
"product_selection_penalty_amt",
"drug_price_tier",
"hra_amount",
"except_override_tag",
"ltc_override_ind",
"skip_deductible_flag",
"admin_fee_type",
"number_of_mchoice_claims_allowd",
"ads_scp_tag",
"reject1",
"reject2",
"reject3",
"runtime",
"patres",
"pharsrvtyp",
"formulary",
"daysup1",
"thresh1",
"daysup2",
"thresh2",
"daysup3",
"thresh3",
"certid",
"pa_ind",
"rflallwd",
"rundate",
"base_plan_id",
"copay_waiver_drug_specific_copays_tab_0",
"pharmacy_participation_340b",
"allow_unbreakable_packages_for_emergency_supply",
"allow_unbreakable_packages_for_transition_fill",
"allow_which_occ_other_coverage_codes",
"apply_cms_labeler_rules",
"apply_ndc5_standard_rebate_exception_list",
"bulk_chemicals_covered",
"compound_post_max_dollar_limit_reject_messaging",
"compounds_max_dollar_limit",
"controlled_substances_cii_ciii_civ_cv_refill_threshold",
"copay_exceptions",
"courtesy_grace_fill_limit",
"courtesy_grace_fills_if_exclusive",
"cover_multi_ingredient_compounds",
"custom_post_oop_met_copay",
"custom_specialty_copay",
"cvs_health_vaccine_program_applies",
"desi_exclusions",
"do_specialty_copays_mirror_retail",
"does_emergency_fill_claims_history_count_towards_clinical_rule_lookbacks",
"does_split_fill_apply",
"does_the_plan_have_prescriber_gold_carding",
"does_transition_fill_claims_history_count_towards_clinical_rule_lookbacks",
"dur_services",
"emergency_fill_lookback_gpi",
"emergency_fill_max__of_fills",
"emergency_fill_max_day_supply",
"emergency_fill_allowed",
"emergency_fill_applies_to_age_gender_rejects",
"emergency_fill_applies_to_exclusions",
"emergency_fill_applies_to_pa_rejects",
"emergency_fill_applies_to_qty_limit_rejects",
"emergency_fill_applies_to_reverse_step_rejects",
"emergency_fill_applies_to_smart_edit_rejects",
"emergency_fill_applies_to_step_therapy_rejects",
"emergency_fill_applies_to_what_delivery_system",
"emergency_fill_applies_to_which_members",
"emergency_fill_copay",
"emergency_fill_history_review_lookback_window_days",
"emergency_fill_limit_messaging",
"exclusive_or_open_specialty",
"how_does_the_split_fill_copay_apply",
"if_split_fill_applies_what_drug_categories",
"is_daw5_allowed_and_if_so_at_what_copay",
"is_there_a_copay_amount_after_mab_is_met",
"is_there_a_copay_amount_after_oop_is_met",
"mail_claim_cost_0_to_10",
"mail_claim_cost_10_01_to_25",
"mail_claim_cost_25_01_to_50",
"mail_claim_cost_50_01_or_more",
"mail_covered",
"mail_generic_copayment",
"mail_non_preferred_brand_copayment",
"mail_order_refill_threshold",
"mail_preferred_brand_copayment",
"max__of_fills",
"max_dollar_mail_non_specialty",
"max_dollar_retail_paper_claims_non_specialty",
"max_dollarspecialty",
"msb_drugs_psc_exlcusions",
"msb_drugs_psc_exlcusions_reject_messaging",
"multi_ingredient_compounds_prior_authorization",
"natural_disaster_recovery_scc_13",
"non_specialty_post_max_dollar_limit_reject_messaging",
"oop_4th_quarter_carryover",
"oop_amount",
"oop_type",
"ophthalmic_refill_threshold",
"other_drugs_not_eliglible_for_emergency_fill",
"other_drugs_not_eliglible_for_transition_fill",
"paper_claim_refill_threshold",
"rebate_supplemental_list_applies",
"reject_claim_after_mab_met",
"retail_claim_cost_0_to_10",
"retail_claim_cost_10_01_to_25",
"retail_claim_cost_25_01_50",
"retail_claim_cost_50_01_or_more",
"retail_claim_refill_threshold",
"retail_generic_copayment",
"retail_non_preferred_brand_copayment",
"retail_preferred_brand_copayment",
"scriptsync_cvs_medication_synchronization",
"specialty_max_ds",
"specialty_network",
"specilaty_refill_threshold",
"split_fill_for_first_15ds_and_30ds",
"tpe_exclusions",
"transition_fill_lookback_gpi",
"transition_fill_max_cumulative_day_supply",
"transition_fill_allowed",
"transition_fill_applies_to_age_gender_rejects",
"transition_fill_applies_to_exclusions",
"transition_fill_applies_to_pa_rejects",
"transition_fill_applies_to_qty_limit_rejects",
"transition_fill_applies_to_reverse_step_rejects",
"transition_fill_applies_to_step_therapy_rejects",
"transition_fill_applies_to_what_delivery_system",
"transition_fill_applies_to_which_members",
"transition_fill_history_review_lookback_window_days",
"transition_fill_limit_messaging",
"load_date"]

for plan in brmd_baseplans: 
	hql_str = " select * FROM  temp_table where final_plan_code='"+ plan+"' limit 52500"
	sqlDF = sqlContext.sql(hql_str)
	print('Plan_code: ' + plan)
	print(sqlDF.printSchema())
	if(sqlDF.count() > 30000):
        	#**sqlDF.write.mode("append").partitionBy("load_date").saveAsTable(sampled_hiveTable)
		sqlDF.select(final_cols).repartition(150).write.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("dbtable", sampled_hiveTable).mode('append').save()



# Show the results using SELECT
#**spark.sql("select * from {}".format(sampled_hiveTable)).show(2)
qry = "select * from {}".format(sampled_hiveTable)
spark.read.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("query",  qry).load().show(2)
sql_count_qry = "select count(*) from {0} where load_date = '{1}'".format(sampled_hiveTable,Partition_Date)
df = spark.read.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("query",  sql_count_qry).load()
print("Count of training data after sampling: ", df.count())


