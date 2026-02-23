#!/usr/bin/env python
# coding: utf-8

# Run code as: python data_sampling.py prod_fl_config.ini


import os
import sys
from configparser import ConfigParser
import pyodbc
import pandas as pd
import traceback
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500) 
pd.set_option('display.max_rows', 500) 



#------------------------------------------------------------------------------------------------------------------------------------------#


config_filename = str(sys.argv[1])
print("Reading config details from: ", config_filename)

config_object = ConfigParser()
config_object.read(config_filename)
datasampling_config = config_object["DATASAMPLING"]

date = datasampling_config['date']
full_training_claims_data_filename = datasampling_config['full_training_claims_data_filename']

sql_server = datasampling_config['sql_server']
sql_db = datasampling_config['sql_db']
sql_usr = datasampling_config['sql_usr']
sql_pwd = datasampling_config['sql_pwd']
sql_brmd_master_table = datasampling_config['sql_brmd_master_table']

working_dir = datasampling_config['working_dir']
training_data_filename = datasampling_config['training_data_filename']

pickle_dir = datasampling_config['pickle_dir']

target_fields_list_name = datasampling_config['target_fields_list_name']
target_fields_list_name = pickle_dir + 'lists/' + target_fields_list_name + '_' + date


#------------------------------------------------------------------------------------------------------------------------------------------#


# BRMD_MASTER data
#conn = pyodbc.connect('DRIVER={ODBC Driver 11 for SQL Server};SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)

conn = pyodbc.connect('DRIVER=/opt/microsoft/msodbcsql17/lib64/libmsodbcsql-17.7.so.2.1;SERVER='+sql_server+';DATABASE='+sql_db+';UID='+sql_usr+';PWD='+ sql_pwd)
cursor = conn.cursor()

select_query = "SELECT * FROM " + sql_brmd_master_table + " WHERE status = 'A'"
print("Selecting active plans using query: ", select_query)
brmd_master_df = pd.read_sql(select_query, conn)

conn.close()

print(brmd_master_df.head(1))


#------------------------------------------------------------------------------------------------------------------------------------------#


# Hard coded for now.
def clean_data(data):
    print("Cleaning target data...")
    
    # Convert datatype to string
    data = data.astype(str)
    
    # Replace similar values with a single value. This will always be hard-coded. 
    
    data['allow_which_occ_other_coverage_codes'].replace({'02.03,04':'02 and 03 and 04' },inplace=True)
    data['allow_which_occ_other_coverage_codes'].replace({'02, 03 ,04':'02 and 03 and 04' },inplace=True)
    data['allow_which_occ_other_coverage_codes'].replace({'2,3,4':'02 and 03 and 04' },inplace=True)
    data['allow_which_occ_other_coverage_codes'].replace({'2.03.04':'02 and 03 and 04' },inplace=True)
    data['allow_which_occ_other_coverage_codes'].replace({'02.03.04':'02 and 03 and 04' },inplace=True)
    data['allow_which_occ_other_coverage_codes'].replace({'03,08':'02 and 03 and 04' },inplace=True)
    data['allow_which_occ_other_coverage_codes'].replace({'02,03,89':'02 and 03 and 04' },inplace=True)
    
    data['emergency_fill_lookback_gpi'].replace({'14-Match Full GPI Number':'GPI14' },inplace=True)
    data['emergency_fill_lookback_gpi'].replace({'GPI12':'GPI14' },inplace=True)
    
    data['transition_fill_lookback_gpi'].replace({'14-Match Full GPI Number':'GPI14' },inplace=True)
    data['transition_fill_lookback_gpi'].replace({'GPI12':'GPI14' },inplace=True)
    
    data['bulk_chemicals_covered'].replace({'Formulary Driven':'Yes - Formulary Driven' },inplace=True)
    data['bulk_chemicals_covered'].replace({'No':'No - Not Covered' },inplace=True)
    
    data['transition_fill_applies_to_age_gender_rejects'].replace({'NO except for Budesonide, ADHD, and Insulin pens':'No, (except Budesonide, ADHD and Insulin Pens)' },inplace=True)
    data['transition_fill_applies_to_age_gender_rejects'].replace({'Age/Gender Rejects (except Budesonide, ADHD and Insulin Pens)':'Age/Gender Rejects (except Budesonid and Insulin Pens *No ADHD b/c state limits*)' },inplace=True)
    
    data['msb_drugs_psc_exlcusions'].replace({'Reject all pcs 0-4 and 6-9':'Reject PSCs 0-4 & 6-9'},inplace=True)
    data['msb_drugs_psc_exlcusions'].replace({'PSC 0-4 & PSC 6-9 should reject with R70':'Reject PSCs 0-4 & 6-9'},inplace=True)
    data['msb_drugs_psc_exlcusions'].replace({'PSC 0,3,4,6,7,9 should reject with R22|PSC 1,2,8 Should reject with R70':'PSC 0,3,4,6,9 should reject with R22  PSC 1,2,8 Should reject with R70 PSC 5,7 Should allow'},inplace=True)
    data['msb_drugs_psc_exlcusions'].replace({'Reject 0-4,6,8,9':'Reject PSCs 0-4 & 6-9'},inplace=True)
    
    data['transition_fill_limit_messaging_'].replace({'Mbr Exhausted Transition Fill':'MEMBER EXHAUSTED TRANSITION FILL' },inplace=True)
    data['transition_fill_limit_messaging_'].replace({'Transition Fill Limit exceeded':'MEMBER EXHAUSTED TRANSITION FILL' },inplace=True)
    data['transition_fill_limit_messaging_'].replace({'Member Exhausted Transition Fill':'MEMBER EXHAUSTED TRANSITION FILL' },inplace=True)
    
    data['emergency_fill_applies_to_which_members'].replace({'All Members Anytime':'All members anytime'},inplace=True)
    data['emergency_fill_applies_to_which_members'].replace({'All members - with informed consent exclusions':'All members anytime'},inplace=True)
    
    data['does_emergency_fill_claims_history_count_towards_clinical_rule_lookbacks'].replace({'No specific requirements':'No' },inplace=True)
    
    data['transition_fill_max_cumulative_day_supply'].replace({'62':'60' },inplace=True)
    data['transition_fill_max_cumulative_day_supply'].replace({'31':'30' },inplace=True)
    
    data['does_split_fill_apply'].replace({'Yes':'YES - Retail NO - Mail' },inplace=True)
    data['does_split_fill_apply'].replace({'YES - Retail\nNO - Mail':'YES - Retail NO - Mail' },inplace=True)
    
    data['specialty_max_ds'].replace({'34':'30' },inplace=True)
    data['specialty_max_ds'].replace({'31':'30' },inplace=True)
    
    data['retail_maximum_days_supply'].replace({'29':'30' },inplace=True)
    data['retail_maximum_days_supply'].replace({'34':'30' },inplace=True)
    data['retail_maximum_days_supply'].replace({'31':'30' },inplace=True)

    data['state_carve_outs'].replace({'Hemophilia, Spinraza, Exondys 51 to FFS.? The hemophilia drugs are carved out of the capitation payment to managed care enrollees, and are paid fee-for-service.|The Comprehensive Statewide Hemophilia Disease Management Program (DMOH assignment plan) will continue to exist under MMA, and these drugs are carved out from the pharmacy benefit provided by managed care plans with exception of Healthy Kids - no carve outs':'Hemophilia - The hemophilia drugs are carved out of the capitation payment to managed care enrollees, and are paid fee-for-service. The Comprehensive Statewide Hemophilia Disease Management Program (DMOH assignment plan) will continue to exist under MMA, and these drugs are carved out from the pharmacy benefit provided by managed care plans with exception of Healthy Kids - no carve outs' },inplace=True)
    data['state_carve_outs'].replace({'Hemophilia, Spinraza, Exondys 51 to FFS.• The hemophilia drugs are carved out of the capitation payment to managed care enrollees, and are paid fee-for-service. The Comprehensive Statewide Hemophilia Disease Management Program (DMOH assignment plan) will continue to exist under MMA, and these drugs are carved out from the pharmacy benefit provided by managed care plans with exception of Healthy Kids - no carve outs':'Hemophilia - The hemophilia drugs are carved out of the capitation payment to managed care enrollees, and are paid fee-for-service. The Comprehensive Statewide Hemophilia Disease Management Program (DMOH assignment plan) will continue to exist under MMA, and these drugs are carved out from the pharmacy benefit provided by managed care plans with exception of Healthy Kids - no carve outs' },inplace=True)
    data['state_carve_outs'].replace({'Yes - HIV drugs •Exceptions are  ?Viread used for treatment of Hep B  ?Truvada used for pre-exposure prophylaxis (precaution) ':'Yes' },inplace=True)
    
    data['mail_covered'].replace({'Yes, VAB OTC drugs only':'Yes' },inplace=True)
    
    data['non_specialty_post_max_dollar_limit_reject_messaging'].replace({'Cost Exceeds Maximum $5,000 - PA required                                   For Mail Plans: Cost Exceeds Maximum $10,000 - PA required':'COSTS EXCEEDS MAXIMUM' },inplace=True)
    data['non_specialty_post_max_dollar_limit_reject_messaging'].replace({'COST EXCEEDS MAX; CALL PHARMACY HELP DESK 1-833-296-5037':'COSTS EXCEEDS MAXIMUM' },inplace=True)
    data['non_specialty_post_max_dollar_limit_reject_messaging'].replace({'Retail-Cost Exceeds Maximum $5000 - PA required Mail - Cost Exceeds Maximum $10000 - PA required':'COSTS EXCEEDS MAXIMUM' },inplace=True)
    data['non_specialty_post_max_dollar_limit_reject_messaging'].replace({'COST EXCEEDS MAX; CALL PHARMACY HELP DESK 1-833-296-5037':'COSTS EXCEEDS MAXIMUM' },inplace=True)
    data['non_specialty_post_max_dollar_limit_reject_messaging'].replace({'Retail-Cost Exceeds Maximum $5000 - PA required\nMail - Cost Exceeds Maximum $10000 - PA required':'COSTS EXCEEDS MAXIMUM' },inplace=True)
    data['non_specialty_post_max_dollar_limit_reject_messaging'].replace({'COST EXCEEDS MAX; CALL PHARMACY HELP\nDESK 1-833-296-5037':'COSTS EXCEEDS MAXIMUM' },inplace=True)
    
    data['copay_waiver_drug_specific_copays_tab_0'].replace({'Yes':'Yes- Naloxone products' },inplace=True)
    data['copay_waiver_drug_specific_copays_tab_0'].replace({'Yes - Nalaxone':'Yes- Naloxone products' },inplace=True)
    data['copay_waiver_drug_specific_copays_tab_0'].replace({'FDA-approved drugs to treat tuberculosis, Contraceptive Meds and Devices, Smoking Deterrents, OTCs, Diabetic Supplies Insulin Needles Syringes, Pyscotherapeutic agents, Vaccines Pure B only, Specialty Mandate list, PDL, Naloxone, Antidiabetics, CHEMO.':'Yes- Naloxone products' },inplace=True)
    data['copay_waiver_drug_specific_copays_tab_0'].replace({'Naloxone products':'Yes- Naloxone products' },inplace=True)
    data['copay_waiver_drug_specific_copays_tab_0'].replace({'Yes - Naloxone products':'Yes- Naloxone products' },inplace=True)
    data['copay_waiver_drug_specific_copays_tab_0'].replace({'FDA-approved drugs to treat tuberculosis, Contraceptive Meds and Devices, Smoking Deterrents, OTCs, Diabetic Supplies Insulin Needles Syringes, Pyscotherapeutic agents, Vaccines Pure B only, Specialty Mandate list, PDL, Naloxone, Antidiabetics, CHEMO, CHIP-DME.':'Yes- Naloxone products' },inplace=True)
    data['copay_waiver_drug_specific_copays_tab_0'].replace({'N/A- No copays':'NA' },inplace=True)
    
    data['paper_claim_timely_filing_limit_days'].replace({'95':'90' },inplace=True)
    data['paper_claim_timely_filing_limit_days'].replace({'180 PAR':'180' },inplace=True)
    data['paper_claim_timely_filing_limit_days'].replace({'90 days for Contracted Providers|180 days for Non-Contracted Providers':'90 days for Contracted Providers 180 days for Non-Contracted Providers' },inplace=True)

    data['how_does_the_split_fill_copay_apply_'].replace({'Allow first fill of 15 days to waive copay, 2nd Fill full copay':'Allow first fill of 15 days to waive copay' },inplace=True)
    data['how_does_the_split_fill_copay_apply_'].replace({'$0 for any fill':'No copay/$0' },inplace=True)
    data['how_does_the_split_fill_copay_apply_'].replace({'N/A(Zero copay applies to plan)':'No copay/$0' },inplace=True)
    data['how_does_the_split_fill_copay_apply_'].replace({'N/A- No Copay':'No copay/$0' },inplace=True)
    data['how_does_the_split_fill_copay_apply_'].replace({'No Plan Copay':'No copay/$0' },inplace=True)
    
    data['specialty_network'].replace({'recommend PBM- FL member can choose':'CVS Specialty plus the additional pharmacies in Specialty Network Tab' },inplace=True)
    data['specialty_network'].replace({'CVS Speciality':'CVS Specialty plus the additional pharmacies in Specialty Network Tab' },inplace=True)
    data['specialty_network'].replace({'CVS Specialty +          See specialty Network tab':'CVS Specialty plus the additional pharmacies in Specialty Network Tab' },inplace=True)
    data['specialty_network'].replace({'CVS Specialty':'CVS Specialty plus the additional pharmacies in Specialty Network Tab' },inplace=True)
    data['specialty_network'].replace({'CVS Speciality':'CVS Specialty plus the additional pharmacies in Specialty Network Tab' },inplace=True)
    data['specialty_network'].replace({'CVS Specialty & Duncan' 'CVS  Specialty':'CVS Specialty plus the additional pharmacies in Specialty Network Tab' },inplace=True)
    data['specialty_network'].replace({'*2 additional Empire specialty pharmacy(1528342557)  and giannotto specialty pharmacy (1497776454 )along with the CVS Specialty pharmacy network.':'CVS Specialty plus the additional pharmacies in Specialty Network Tab' },inplace=True)
    data['specialty_network'].replace({'CVS Specialty + Pronetics' 'CVS SPECIALTY':'CVS Specialty plus the additional pharmacies in Specialty Network Tab' },inplace=True)          
                        
    data['msb_drugs_psc_exlcusions_reject_messaging'].replace({'R70 (PSC 1,2,8) - GENERICS PREFERED. BRAND REQUIRES PA   R70 (PSC 0,3,4,6,7,9)  - GENRIC SUBST. REQUIRED FOR PAYMENT.':'R70 (PSC 1,2,8) - GENERICS PREFERED. BRAND REQUIRES PA   R70 (PSC 0,3,4,6,7,9)  - GENERIC SUBST. REQUIRED FOR PAYMENT.' },inplace=True)
    data['msb_drugs_psc_exlcusions_reject_messaging'].replace({'PSC 5: R70 - Non-Formulary Drug, Contact Prescriber PSC 0-4 & 6-9: R75  - GENRIC SUBST. REQUIRED FOR PAYMENT.':'PSC 5: R70 - Non-Formulary Drug, Contact Prescriber PSC 0-4 & 6-9: R70  - GENRIC SUBST. REQUIRED FOR PAYMENT.' },inplace=True)
    data['msb_drugs_psc_exlcusions_reject_messaging'].replace({'R70 (PSC 1,2,8) - GENERICS PREFERED. BRAND REQUIRES PA R70 (PSC 0,3,4,6,7,9)  - GENRIC SUBST. REQUIRED FOR PAYMENT.':'R70 (PSC 1,2,8) - GENERICS PREFERED. BRAND REQUIRES PA   R70 (PSC 0,3,4,6,7,9)  - GENERIC SUBST. REQUIRED FOR PAYMENT.' },inplace=True)
    data['msb_drugs_psc_exlcusions_reject_messaging'].replace({'R70 (PSC 1,2,8) - GENERICS PREFERED. BRAND REQUIRES PA R70 (PSC 0,3,4,6,9)  - GENRIC SUBST. REQUIRED FOR PAYMENT.':'R70 (PSC 1,2,8) - GENERICS PREFERED. BRAND REQUIRES PA   R70 (PSC 0,3,4,6,7,9)  - GENERIC SUBST. REQUIRED FOR PAYMENT.' },inplace=True)
    data['msb_drugs_psc_exlcusions_reject_messaging'].replace({'BRAND NOT COVERED - DISPENSE GENERIC':'BRAND NC/FILL GX: 8004543730' },inplace=True)
    data['msb_drugs_psc_exlcusions_reject_messaging'].replace({'BRAND NOT COVER-DISPENSE GX - CALL 8004543730':'BRAND NC/FILL GX: 8004543730' },inplace=True)
    
    data['msb_drugs_psc_exlcusions'].replace({'PSC 0,3,4,6,9 should reject with R22 \nPSC 1,2,8 Should reject with R70\nPSC 5,7 Should allow':'Reject PSCs 0-4 & 6-9' },inplace=True)
    data['msb_drugs_psc_exlcusions'].replace({'PSC 0,3,4,6,9 should reject with R22 \nPSC 1,2,8 Should reject with R70\nPSC 5 & 7 allow':'Reject PSCs 0-4 & 6-9' },inplace=True)
    data['msb_drugs_psc_exlcusions'].replace({'PSC 0,3,4,6,9 should reject with R22  PSC 1,2,8 Should reject with R70 PSC 5,7 Should allow':'Reject PSCs 0-4 & 6-9' },inplace=True)
    data['msb_drugs_psc_exlcusions'].replace({'Formulary Driven.\nExceptions:\n1. Mail - Certain MSB drugs will override the formulary with PSC 5.\n2. COB - If primary paid the claim (OCC 2), override/ignore the MSB exclusions.\n3. PSC 5 and 7 are allowed to pay as generic if the generic is covered in the formulary.':'Formulary Driven.\nExceptions:\n1. Mail - Certain MSB drugs will override the formulary with PSC 5.\n2. COB - If primary paid the claim (OCC 2), override/ignore the MSB exclusions.\n3. PSC 5 and 7 are allowed to pay as generic if the generic is covered in the formulary.' },inplace=True)

    data['custom_specialty_copay'].replace({'Mirrors retail':'Same as retail' },inplace=True)
    
    data['generic_drug_rules'].replace({'=-No DAW Penalty':'No DAW Penalty' },inplace=True)
    
    data['emergency_fill_max__of_fills'].replace({'2':'2 fills per 30 days' },inplace=True)
    data['emergency_fill_max__of_fills'].replace({'1 per 30 days':'1 fill per 30 days' },inplace=True)    
    data['emergency_fill_max__of_fills'].replace({'1':'1 fill per 30 days' },inplace=True)
    data['emergency_fill_max__of_fills'].replace({'2 fill per 30 days':'2 fills per 30 days' },inplace=True)
    data['emergency_fill_max__of_fills'].replace({'1 fill up tor 3 days (except Suboxone 1 fill up to 7 days)':'1 fill per 30 days' },inplace=True)
    data['emergency_fill_max__of_fills'].replace({'1 fill per gcn every 30 days  - 12 fills':'1 fill per 30 days' },inplace=True)
    
    data['is_there_a_copay_amount_after_oop_is_met'].replace({'No':'No( Zero Dollar Copay)' },inplace=True)
    
    data['mail_order_maximum_days_supply'].replace({'93':'90' },inplace=True)
    data['mail_order_maximum_days_supply'].replace({'34':'30' },inplace=True)
    data['mail_order_maximum_days_supply'].replace({'31':'30' },inplace=True)
    
    data['transition_fill_history_review_lookback_window_days'].replace({'90 days':'90' },inplace=True)
    data['transition_fill_history_review_lookback_window_days'].replace({'31 standard /60 behavioral health':'90' },inplace=True)
    
    data['electronic_claim_timely_filing_limit'].replace({'95':'90' },inplace=True)
    data['electronic_claim_timely_filing_limit'].replace({'90 days for Contracted Providers\n180 days for Non-Contracted Providers':'90 days for Contracted Providers|180 days for Non-Contracted Providers' },inplace=True) 
    
    data['emergency_fill_history_review_lookback_window_days'].replace({'90 days':'90' },inplace=True)
    data['emergency_fill_history_review_lookback_window_days'].replace({'30 days':'30' },inplace=True)
    data['emergency_fill_history_review_lookback_window_days'].replace({'30 Days':'30' },inplace=True)
    
    data['does_transition_fill_claims_history_count_towards_clinical_rule_lookbacks'].replace({'Only if stated im the auto PA criteria':'Only if stated in the auto PA criteria' },inplace=True)    
    
    data['emergency_fill_history_review_lookback_window_days'].astype(str).replace({'30 days':'30'},inplace=True)

    data['tpe_exclusions'].replace({'c - cosmetic alteration drugs\r\nv - impotence agents\r\n7 - fertility drugs\r\n8 - anorexic, anti-obesity\r\n3- surgical supply/medical device/ostomy (medical benefit only. refer to medical)\r\n5- diagnostic agent (medical benefit only. refer to medical)':
    'c - cosmetic alteration drugs\r\nv - impotence agents\r\n7 - fertility drugs\r\n8 - anorexic, anti-obesity\r\n5 - diagnostic agent (medical benefit only. refer to medical)\r\n3 - surgical supply/medical device/ostomy (medical benefit only. refer to medical)'},inplace=True)
    data['tpe_exclusions'].replace({'c - cosmetic alteration drugs\r\nv - impotence agents                                                                                                                   \r\n3 - surgical supply/medical device/ostomy - reject message: medical benefit only. refer to medical\r\n5 - diagnostic agent - reject message: medical benefit only. refer to medical.\r\n7 - fertility drugs\r\n8 - anorexic, anti-obesity':
    'c - cosmetic alteration drugs\r\nv - impotence agents\r\n7 - fertility drugs\r\n8 - anorexic, anti-obesity\r\n5 - diagnostic agent (medical benefit only. refer to medical)\r\n3 - surgical supply/medical device/ostomy (medical benefit only. refer to medical)'},inplace=True)
    
    data['is_there_a_copay_amount_after_oop_is_met'].replace({'no( zero dollar copay)':'no'},inplace=True)
    
    data['if_split_fill_applies_what_drug_categories'].replace({'*hep b\r\n*hep c - with sovaldi and olysio\r\n*oncology':'*hep b|*hep c - with sovaldi and olysio|*oncology'},inplace=True)
    
    data['allow_unbreakable_packages_for_emergency_supply'].replace({'No except Insulins/Asthma Inh':'No'},inplace=True)
    
    data['dur_services'].replace({'cvs standard dur services':'cvs standard'},inplace=True)
    
    data['desi_exclusions'].replace({'yes 5 and 6':'5 and 6'},inplace=True)
    data['desi_exclusions'].replace({'5 & 6':'5 and 6'},inplace=True)
    data['desi_exclusions'].replace({'5, 6':'5 and 6'},inplace=True)
    data['desi_exclusions'].replace({'Yes 5 and 6':'5 and 6'},inplace=True)

    data['max_dollar_retail_paper_claims_non_specialty'].replace({'Yes 5 and 6':'5 and 6' },inplace=True)
    
    data['multi_ingredient_compounds_prior_authorization'].replace({'Yes - Drug specific - any ingredient that requires a PA will be subject to a PA.':'Yes' },inplace=True)
    
    data['cover_multi_ingredient_compounds'].replace({'Yes':'Yes - Formulary Driven'},inplace=True)
    data['cover_multi_ingredient_compounds'].replace({'No':'No - Not Covered'},inplace=True)

    data.replace(to_replace=['4999.99','4999.0','5000.0','9999.99','9999.0','10000.0','99999.0', '99999'], value=['5000','5000','5000','10000','10000','10000','100000','100000'], inplace=True)
    data.replace(to_replace=[5000.0,10000.0], value=['5000','10000'], inplace=True)
    data.replace(to_replace=['0.0','99.0'], value=['0','99'], inplace=True)
    data.replace(to_replace=[0.0,99.0], value=['0','99'], inplace=True)
    
    data['compound_post_max_dollar_limit_reject_messaging'].replace({'CALL POS HELP DESK 8004543730':'COMPOUND CLAIM REVIEW REQUIRED' },inplace=True)
    data['compound_post_max_dollar_limit_reject_messaging'].replace({'Compound Claim Review Required':'COMPOUND CLAIM REVIEW REQUIRED'},inplace=True)
    data['compound_post_max_dollar_limit_reject_messaging'].replace({'COMPOUND CLAIM REVIEW REQ 8004543730':'COMPOUND CLAIM REVIEW REQUIRED' },inplace=True)
    data['compound_post_max_dollar_limit_reject_messaging'].replace({'COMPOUND CLAIM REVIEW REQ  \n 855-661-2028':'COMPOUND CLAIM REVIEW REQUIRED' },inplace=True)
    data['compound_post_max_dollar_limit_reject_messaging'].replace({'COMPOUND CLAIM REVIEW REQ 8004543730 \nPROVIDE HERNANDEZ PAMPHLET TO MBR FOR \nTHIS REJ.GIVE EMERG FILL IF APPROPRIATE':'COMPOUND CLAIM REVIEW REQUIRED' },inplace=True)
    data['compound_post_max_dollar_limit_reject_messaging'].replace({'COSTS EXCEEDS MAXIMUM; CALL POS HELP DESK 8004543730':'COSTS EXCEEDS MAXIMUM' },inplace=True)
    data['compound_post_max_dollar_limit_reject_messaging'].replace({'COST EXCEEDS MAXIMUM $200 - PA REQUIRED':'COSTS EXCEEDS MAXIMUM' },inplace=True)
    data['compound_post_max_dollar_limit_reject_messaging'].replace({'Cost Exceeds Maximum $200 - PA required':'COSTS EXCEEDS MAXIMUM' },inplace=True)
    data['compound_post_max_dollar_limit_reject_messaging'].replace({'CALL POS HELP DESK 8004543730':'COMPOUND CLAIM REVIEW REQUIRED' },inplace=True)
    
    # Make Yes/No values consistent
    data.replace(to_replace=['YES','yes','NO','no','None'], value=['Yes','Yes','No','No','NA'], inplace=True)
    
    # Replace garbage values
    data.replace(to_replace=['WW','AAAA','ghjgh'], value=['NA','NA','NA'], inplace=True)
    
    print("Data cleaned successfully!")
    
    return data


#------------------------------------------------------------------------------------------------------------------------------------------#


# Clean some values in brmd_master table
brmd_master_df = clean_data(brmd_master_df) 


#------------------------------------------------------------------------------------------------------------------------------------------#


# Filtering of columns in BRMD_MASTER
# Drop duplicate columns
brmd_master_df = brmd_master_df.loc[:,~brmd_master_df.T.duplicated(keep='first')]
brmd_master_df = brmd_master_df[brmd_master_df.columns.drop(list(brmd_master_df.filter(regex='timestamp')))]
brmd_master_df = brmd_master_df[brmd_master_df.columns.drop(list(brmd_master_df.filter(regex='date')))]
brmd_master_df = brmd_master_df[brmd_master_df.columns.drop(list(brmd_master_df.filter(regex='modified')))]
brmd_master_df = brmd_master_df[brmd_master_df.columns.drop(list(brmd_master_df.filter(regex='created')))]
brmd_master_df = brmd_master_df[brmd_master_df.columns.drop(list(brmd_master_df.filter(regex='Deleted')))]
brmd_master_df = brmd_master_df[brmd_master_df.columns.drop(list(brmd_master_df.filter(regex='message')))]
brmd_master_df = brmd_master_df[brmd_master_df.columns.drop(list(brmd_master_df.filter(regex='header')))]
brmd_master_df = brmd_master_df[brmd_master_df.columns.drop(list(brmd_master_df.filter(regex='formulary')))]
brmd_master_df = brmd_master_df[brmd_master_df.columns.drop(list(brmd_master_df.filter(regex='description')))]
brmd_master_df = brmd_master_df[brmd_master_df.columns.drop(list(brmd_master_df.filter(regex='um_')))]

# Drop non-required columns
#drop_cols_list = ['status','id','state','client_code',
#                  'new_or_existing_plan_change','plan_type','cvsh_carrier_id', 'isDeleted', 'created_by', 
#				  'isCurrentVersion', 'versionId', 'versionPlanId', 'selected_prediction_mode']
drop_cols_list = ['status','id','state','client_code',
                  'new_or_existing_plan_change','plan_type','cvsh_carrier_id', 
				  'isCurrentVersion', 'versionId']

for col in drop_cols_list:
    brmd_master_df = brmd_master_df.drop(col, axis = 1)
    
# Data cleaning for brmd. Not using columns with 30+ unique values.
cols_to_exclude = ['xref_plan_code','final_plan_code','plan_id','base_plan_id']
cols_dropped = []
for col in brmd_master_df.columns:
    if (col not in cols_to_exclude):
        if(brmd_master_df[col].nunique() > 30):
            cols_dropped.append(col)
            brmd_master_df.drop([col], axis=1, inplace=True)
            
print("Columns dropped from brmd data: ", cols_dropped)
print("Number of columns left in brmd_master: ", len(brmd_master_df.columns.values))



#------------------------------------------------------------------------------------------------------------------------------------------#


print("List of unique values for each field: ")
for col in brmd_master_df.columns:
    print("\nUnique values for {0}: {1}".format(col, brmd_master_df[col].unique()))
    
print('\n\nDone printing list of all unique values for all columns.\n\n')



#------------------------------------------------------------------------------------------------------------------------------------------#


# Get plans list. Renaming to xref_plan_code (plan_id (brmd_master) = xref_plan_code (claims_data))
brmd_master_df.rename(columns={'plan_id':'xref_plan_code'}, inplace=True)
plan_id_list = brmd_master_df['xref_plan_code'].unique()
print("List of plan_id present in brmd_master: ", plan_id_list)



#------------------------------------------------------------------------------------------------------------------------------------------#


# Full Training Claims data (non-sampled)
#full_training_claims_data = pd.read_csv(full_training_claims_data_filename)
full_training_claims_data = pd.read_csv(full_training_claims_data_filename)

full_training_claims_data.columns = map(str.lower, full_training_claims_data.columns)

# Filter out non-paid claims. Only consider paid claims.
full_training_claims_data = full_training_claims_data[full_training_claims_data.claim_status.eq('P')]

#full_training_claims_data = full_training_claims_data.filter("claim_status = 'P'")

# Drop duplicates
full_training_claims_data.drop_duplicates(keep = 'first', inplace = True)

print("Count of full paid claims data: ", len(full_training_claims_data))
print(full_training_claims_data.head(1))



#------------------------------------------------------------------------------------------------------------------------------------------#


filtered_df = full_training_claims_data[full_training_claims_data['xref_plan_code'].isin(plan_id_list)]
print("Intermediate Count: ", len(filtered_df))



#------------------------------------------------------------------------------------------------------------------------------------------#


# Sample data
sample_df = pd.DataFrame()
for plan in plan_id_list:
    temp_df = filtered_df[filtered_df.xref_plan_code == plan]
    
    if(len(temp_df) > 12500):
        temp_df = temp_df.sample(n = 12500, random_state = 123)
        sample_df = sample_df.append(temp_df, ignore_index=True)
    elif(len(temp_df) > 5000):
        #temp_df = temp_df.sample(n = 5000, random_state = 123)
        sample_df = sample_df.append(temp_df, ignore_index=True)
    else:
        continue
        
print("Count of claims data after sampling is: ", sample_df.shape)
print('Count of plan_ids in sampled data: ', sample_df['xref_plan_code'].nunique())
print("Count of claims per plan_id: ", sample_df['xref_plan_code'].value_counts())


#------------------------------------------------------------------------------------------------------------------------------------------#


def data_cleaning(train_data):
    cols = []
    cat_data = train_data.select_dtypes(include='object')
    num_data = train_data.select_dtypes(exclude='object')
    
    cols_to_exclude = ['xref_plan_code','final_plan_code','plan_id','base_plan_id']
    
    # Remove columns with over 30 values for categorical values
    for col in cat_data.columns:
        if (col not in cols_to_exclude):
            if (cat_data[col].nunique() > 30) or (cat_data[col].nunique()<=1):
                cols.append(col)
                cat_data.drop([col], axis=1, inplace=True)

    for col in num_data.columns:
        if (num_data[col].nunique()<=1):
            cols.append(col)
            num_data.drop([col], axis=1, inplace=True)

    # Missing value imputation
    cat_data.fillna('NA',inplace=True)
    num_data.fillna(0,inplace=True)

    # Creating the cleaned training sample
    train_data= pd.concat([cat_data,num_data],axis=1)
    
    return train_data, cols



#------------------------------------------------------------------------------------------------------------------------------------------#


# Filtering of columns in sampled claims_data
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='timestamp')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='date')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='modified')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='created')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='Deleted')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='message')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='header')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='formulary')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='_id')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='name')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='description')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='golive')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='runtime')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='location')))]
sample_df = sample_df[sample_df.columns.drop(list(sample_df.filter(regex='patres')))]

# Drop columns with unique value
sample_df.drop(columns=sample_df.columns[sample_df.nunique()==1], inplace=True)

# Clean data
sample_df, cols = data_cleaning(sample_df)

drop_cols_list = ['rxnumber']
for col in drop_cols_list:
    sample_df = sample_df.drop(col, axis = 1)
    
print("Number of columns left in sampled claims data: ", len(sample_df.columns.values))



#------------------------------------------------------------------------------------------------------------------------------------------#


joined_df = pd.merge(sample_df, brmd_master_df, on='xref_plan_code',how='inner')
print("Shape of data after joining with brmd data: ", joined_df.shape)

# Save a list of fields from brmd_master
target_fields_list = list(set(brmd_master_df.columns.values.tolist()) - set(['xref_plan_code','final_plan_code','plan_id','base_plan_id']))
print("Count of target fields: ", len(target_fields_list))
print("List of target fields: ", target_fields_list)

os.makedirs(os.path.dirname(pickle_dir + 'lists/'), exist_ok=True)

open_file = open(target_fields_list_name, "wb")
pickle.dump(brmd_master_df.columns.values.tolist(), open_file)
open_file.close()

print("\nSaved the list of all target fields.")
print("Number of unique values: ", joined_df.nunique())
print("Count of original training sample data: ", len(joined_df)) 

test_df = joined_df.sample(n = 50000, random_state = 123)
print("Count of test sample data: ", len(test_df)) 

joined_df = joined_df.drop(test_df.index)
print("Count of new training sample data: ", len(joined_df)) 


#------------------------------------------------------------------------------------------------------------------------------------------#


print("Writing as CSV files..")
test_df.to_csv(working_dir + 'training_data/Test_Data_' + date + '.csv', header = True, index = False)
joined_df.to_csv(training_data_filename + '_' + date + '.csv', header = True, index = False)

print("Sampling code execution finished!")


#------------------------------------------------------------------------------------------------------------------------------------------#


