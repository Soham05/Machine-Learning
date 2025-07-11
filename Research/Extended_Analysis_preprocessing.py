#!/usr/bin/env python3
"""
Dataset Inspection: Check Column Names and Structure
Before running extended analysis
"""

import pandas as pd
import numpy as np

print("="*80)
print("DATASET INSPECTION FOR EXTENDED ANALYSIS")
print("="*80)

# Load the dataset
df = pd.read_csv('hierarchical_model_cleaned_dataset.csv')

print("\n1. BASIC DATASET INFORMATION")
print("-" * 50)
print(f"Dataset shape: {df.shape}")
print(f"Number of hospitals: {df.shape[0]}")
print(f"Number of variables: {df.shape[1]}")

print("\n2. ALL COLUMN NAMES")
print("-" * 50)
print("Column names in your dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n3. DATA TYPES")
print("-" * 50)
print(df.dtypes)

print("\n4. IDENTIFY KEY VARIABLES")
print("-" * 50)

# Look for outcome variable
outcome_candidates = [col for col in df.columns if 'ERR' in col.upper() or 'READM' in col.upper()]
print(f"Potential outcome variables: {outcome_candidates}")

# Look for county grouping variable
county_candidates = [col for col in df.columns if 'county' in col.lower()]
print(f"Potential county grouping variables: {county_candidates}")

# Look for ownership variables
ownership_candidates = [col for col in df.columns if 'ownership' in col.lower()]
print(f"Ownership-related variables: {ownership_candidates}")

# Look for hospital rating variables
rating_candidates = [col for col in df.columns if 'rating' in col.lower()]
print(f"Hospital rating variables: {rating_candidates}")

# Look for social determinant variables
social_candidates = [col for col in df.columns if any(term in col.lower() for term in 
                    ['income', 'poverty', 'uninsured', 'physician', 'rural', 'white', 'college'])]
print(f"Social determinant variables: {social_candidates}")

print("\n5. MISSING VALUES CHECK")
print("-" * 50)
missing_summary = df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0]
if len(missing_summary) > 0:
    print("Variables with missing values:")
    for var, count in missing_summary.items():
        pct = (count / len(df)) * 100
        print(f"  - {var}: {count} ({pct:.1f}%)")
else:
    print("No missing values found")

print("\n6. CATEGORICAL VARIABLES")
print("-" * 50)
categorical_vars = df.select_dtypes(include=['object', 'category']).columns
if len(categorical_vars) > 0:
    print("Categorical variables:")
    for var in categorical_vars:
        unique_vals = df[var].unique()
        print(f"  - {var}: {len(unique_vals)} unique values")
        print(f"    Values: {unique_vals}")
else:
    print("No categorical variables found")

print("\n7. SAMPLE DATA (FIRST 5 ROWS)")
print("-" * 50)
print(df.head())

print("\n8. VARIABLE PATTERNS")
print("-" * 50)
print("Variables ending with '_std' (standardized):")
std_vars = [col for col in df.columns if col.endswith('_std')]
for var in std_vars:
    print(f"  - {var}")

print("\nVariables ending with '_raw_value':")
raw_vars = [col for col in df.columns if 'raw_value' in col]
for var in raw_vars:
    print(f"  - {var}")

print("\n9. SUMMARY STATISTICS FOR KEY VARIABLES")
print("-" * 50)
if outcome_candidates:
    print(f"Outcome variable statistics:")
    for var in outcome_candidates:
        print(f"  {var}:")
        print(f"    Mean: {df[var].mean():.4f}")
        print(f"    Std: {df[var].std():.4f}")
        print(f"    Min: {df[var].min():.4f}")
        print(f"    Max: {df[var].max():.4f}")

print("\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)
print("""
NEXT STEPS:
1. Review the column names above
2. Identify which variables correspond to:
   - Outcome variable (ERR)
   - County grouping variable
   - Hospital ownership categories
   - Hospital rating variable
   - Social determinant variables
3. Let me know the exact column names to use in the extended analysis
""")




Reults:


================================================================================
DATASET INSPECTION FOR EXTENDED ANALYSIS
================================================================================

1. BASIC DATASET INFORMATION
--------------------------------------------------
Dataset shape: (2152, 22)
Number of hospitals: 2152
Number of variables: 22

2. ALL COLUMN NAMES
--------------------------------------------------
Column names in your dataset:
 1. Ownership_Category_Clean
 2. Hospital_Rating_Numeric_Imputed_std
 3. Rating_Missing
 4. median_household_income_raw_value_std
 5. children_in_poverty_raw_value_std
 6. uninsured_adults_raw_value_std
 7. ratio_of_population_to_primary_care_physicians_std
 8. %_rural_raw_value_std
 9. %_non_hispanic_white_raw_value_std
10. some_college_raw_value_std
11. county_group
12. Excess Readmission Ratio
13. Facility ID
14. Facility Name
15. State
16. ERR
17. county_id
18. pct_rural_raw_value_std
19. pct_non_hispanic_white_raw_value_std
20. Excess_Readmission_Ratio
21. Facility_ID
22. Facility_Name

3. DATA TYPES
--------------------------------------------------
Ownership_Category_Clean                               object
Hospital_Rating_Numeric_Imputed_std                   float64
Rating_Missing                                          int64
median_household_income_raw_value_std                 float64
children_in_poverty_raw_value_std                     float64
uninsured_adults_raw_value_std                        float64
ratio_of_population_to_primary_care_physicians_std    float64
%_rural_raw_value_std                                 float64
%_non_hispanic_white_raw_value_std                    float64
some_college_raw_value_std                            float64
county_group                                          float64
Excess Readmission Ratio                              float64
Facility ID                                             int64
Facility Name                                          object
State                                                  object
ERR                                                   float64
county_id                                               int64
pct_rural_raw_value_std                               float64
pct_non_hispanic_white_raw_value_std                  float64
Excess_Readmission_Ratio                              float64
Facility_ID                                             int64
Facility_Name                                          object
dtype: object

4. IDENTIFY KEY VARIABLES
--------------------------------------------------
Potential outcome variables: ['Excess Readmission Ratio', 'ERR', 'Excess_Readmission_Ratio']
Potential county grouping variables: ['county_group', 'county_id']
Ownership-related variables: ['Ownership_Category_Clean']
Hospital rating variables: ['Hospital_Rating_Numeric_Imputed_std', 'Rating_Missing']
Social determinant variables: ['median_household_income_raw_value_std', 'children_in_poverty_raw_value_std', 'uninsured_adults_raw_value_std', 'ratio_of_population_to_primary_care_physicians_std', '%_rural_raw_value_std', '%_non_hispanic_white_raw_value_std', 'some_college_raw_value_std', 'pct_rural_raw_value_std', 'pct_non_hispanic_white_raw_value_std']

5. MISSING VALUES CHECK
--------------------------------------------------
No missing values found

6. CATEGORICAL VARIABLES
--------------------------------------------------
Categorical variables:
  - Ownership_Category_Clean: 3 unique values
    Values: ['Private' 'Public' 'Non-Profit']
  - Facility Name: 2110 unique values
    Values: ['SHANDS JACKSONVILLE' 'BETHESDA  HOSPITAL INC' 'ORLANDO HEALTH' ...
 'TEXAS HEALTH HOSPITAL FRISCO' 'METHODIST MIDLOTHIAN MEDICAL CENTER'
 'TEXAS HEALTH HOSPITAL MANSFIELD']
  - State: 42 unique values
    Values: ['FL' 'GA' 'HI' 'ID' 'IL' 'IN' 'IA' 'KS' 'KY' 'LA' 'ME' 'MD' 'MA' 'MI'
 'MN' 'MS' 'MO' 'MT' 'NE' 'NV' 'NH' 'NJ' 'NM' 'NY' 'NC' 'ND' 'OH' 'OK'
 'OR' 'PA' 'RI' 'SC' 'SD' 'TN' 'TX' 'UT' 'VT' 'VA' 'WA' 'WV' 'WI' 'WY']
  - Facility_Name: 2110 unique values
    Values: ['SHANDS JACKSONVILLE' 'BETHESDA  HOSPITAL INC' 'ORLANDO HEALTH' ...
 'TEXAS HEALTH HOSPITAL FRISCO' 'METHODIST MIDLOTHIAN MEDICAL CENTER'
 'TEXAS HEALTH HOSPITAL MANSFIELD']

7. SAMPLE DATA (FIRST 5 ROWS)
--------------------------------------------------
  Ownership_Category_Clean  Hospital_Rating_Numeric_Imputed_std  \
0                  Private                            -0.097283   
1                  Private                            -1.001196   
2                  Private                            -1.001196   
3                  Private                            -0.097283   
4                  Private                             0.806629   

   Rating_Missing  median_household_income_raw_value_std  \
0               0                              -0.558233   
1               0                               0.205422   
2               0                               0.256771   
3               0                               0.256771   
4               0                              -0.319877   

   children_in_poverty_raw_value_std  uninsured_adults_raw_value_std  \
0                           0.616602                        0.582992   
1                          -0.056164                        1.201987   
2                           0.012486                        0.774791   
3                           0.012486                        0.774791   
4                           0.451843                        1.519053   

   ratio_of_population_to_primary_care_physicians_std  %_rural_raw_value_std  \
0                                          -0.483788               -0.918804   
1                                          -0.404047               -0.995930   
2                                          -0.594037               -0.954680   
3                                          -0.594037               -0.954680   
4                                          -0.458539               -1.022265   

   %_non_hispanic_white_raw_value_std  some_college_raw_value_std  ...  \
0                           -0.764350                    0.088643  ...   
1                           -0.669817                   -0.030125  ...   
2                           -1.357963                    0.436139  ...   
3                           -1.357963                    0.436139  ...   
4                           -2.594412                   -0.179802  ...   

   Facility ID              Facility Name  State     ERR county_id  \
0       100001        SHANDS JACKSONVILLE     FL  1.0077         0   
1       100002     BETHESDA  HOSPITAL INC     FL  0.9386         1   
2       100006             ORLANDO HEALTH     FL  1.0074         2   
3       100007       ADVENTHEALTH ORLANDO     FL  1.1283         2   
4       100008  BAPTIST HOSPITAL OF MIAMI     FL  1.0561         3   

   pct_rural_raw_value_std  pct_non_hispanic_white_raw_value_std  \
0                -0.918804                             -0.764350   
1                -0.995930                             -0.669817   
2                -0.954680                             -1.357963   
3                -0.954680                             -1.357963   
4                -1.022265                             -2.594412   

   Excess_Readmission_Ratio  Facility_ID              Facility_Name  
0                    1.0077       100001        SHANDS JACKSONVILLE  
1                    0.9386       100002     BETHESDA  HOSPITAL INC  
2                    1.0074       100006             ORLANDO HEALTH  
3                    1.1283       100007       ADVENTHEALTH ORLANDO  
4                    1.0561       100008  BAPTIST HOSPITAL OF MIAMI  

[5 rows x 22 columns]

8. VARIABLE PATTERNS
--------------------------------------------------
Variables ending with '_std' (standardized):
  - Hospital_Rating_Numeric_Imputed_std
  - median_household_income_raw_value_std
  - children_in_poverty_raw_value_std
  - uninsured_adults_raw_value_std
  - ratio_of_population_to_primary_care_physicians_std
  - %_rural_raw_value_std
  - %_non_hispanic_white_raw_value_std
  - some_college_raw_value_std
  - pct_rural_raw_value_std
  - pct_non_hispanic_white_raw_value_std

Variables ending with '_raw_value':
  - median_household_income_raw_value_std
  - children_in_poverty_raw_value_std
  - uninsured_adults_raw_value_std
  - %_rural_raw_value_std
  - %_non_hispanic_white_raw_value_std
  - some_college_raw_value_std
  - pct_rural_raw_value_std
  - pct_non_hispanic_white_raw_value_std

9. SUMMARY STATISTICS FOR KEY VARIABLES
--------------------------------------------------
Outcome variable statistics:
  Excess Readmission Ratio:
    Mean: 0.9992
    Std: 0.0640
    Min: 0.7803
    Max: 1.2871
  ERR:
    Mean: 0.9992
    Std: 0.0640
    Min: 0.7803
    Max: 1.2871
  Excess_Readmission_Ratio:
    Mean: 0.9992
    Std: 0.0640
    Min: 0.7803
    Max: 1.2871

================================================================================
INSPECTION COMPLETE
================================================================================
