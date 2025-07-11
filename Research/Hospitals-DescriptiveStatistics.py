import pandas as pd
import numpy as np

# Load your merged dataset
df = pd.read_csv('merged_social_determinants_analysis.csv')

print("=== STEP 1A: HOSPITAL-LEVEL VARIABLES ANALYSIS ===\n")

# Task 1A.1: Frequency distributions for categorical variables
print("1. CATEGORICAL VARIABLES - FREQUENCY DISTRIBUTIONS")
print("="*50)

print("\nHospital_Type_Category:")
print(df['Hospital_Type_Category'].value_counts())
print(f"Missing values: {df['Hospital_Type_Category'].isna().sum()}")

print("\nOwnership_Category:")
print(df['Ownership_Category'].value_counts())
print(f"Missing values: {df['Ownership_Category'].isna().sum()}")

print("\nHas_Emergency_Services:")
print(df['Has_Emergency_Services'].value_counts())
print(f"Missing values: {df['Has_Emergency_Services'].isna().sum()}")

# Task 1A.1: Descriptive statistics for continuous variables
print("\n\n2. CONTINUOUS VARIABLES - DESCRIPTIVE STATISTICS")
print("="*50)

print("\nHospital_Rating_Numeric:")
print(df['Hospital_Rating_Numeric'].describe())
print(f"Missing values: {df['Hospital_Rating_Numeric'].isna().sum()}")

# Task 1A.1: Cross-tabulations
print("\n\n3. CROSS-TABULATIONS")
print("="*50)

print("\nOwnership_Category vs Has_Emergency_Services:")
crosstab1 = pd.crosstab(df['Ownership_Category'], df['Has_Emergency_Services'], margins=True)
print(crosstab1)

print("\nHospital_Type_Category vs Ownership_Category:")
crosstab2 = pd.crosstab(df['Hospital_Type_Category'], df['Ownership_Category'], margins=True)
print(crosstab2)

# Task 1A.2: Primary outcome distribution
print("\n\n4. PRIMARY OUTCOME VARIABLE")
print("="*50)

print("\nExcess Readmission Ratio:")
print(df['Excess Readmission Ratio'].describe())
print(f"Missing values: {df['Excess Readmission Ratio'].isna().sum()}")

# Check for extreme outliers
print(f"\nOutlier check (values beyond 3 standard deviations):")
mean_err = df['Excess Readmission Ratio'].mean()
std_err = df['Excess Readmission Ratio'].std()
outliers = df[(df['Excess Readmission Ratio'] < mean_err - 3*std_err) | 
              (df['Excess Readmission Ratio'] > mean_err + 3*std_err)]
print(f"Number of potential outliers: {len(outliers)}")

# Additional check for the range mentioned in your summary
print(f"\nExpected range check (your summary mentioned 0.780-1.287):")
print(f"Actual min: {df['Excess Readmission Ratio'].min()}")
print(f"Actual max: {df['Excess Readmission Ratio'].max()}")

print("\n\n5. SAMPLE SIZE CONFIRMATION")
print("="*50)
print(f"Total hospitals in dataset: {len(df)}")
print(f"Complete cases for analysis: {len(df.dropna(subset=['Hospital_Type_Category', 'Ownership_Category', 'Has_Emergency_Services', 'Hospital_Rating_Numeric', 'Excess Readmission Ratio']))}")

# Verify the Measure Name filter
print(f"\nMeasure Name verification:")
print(f"Unique values in Measure Name: {df['Measure Name'].unique()}")
print(f"Count of READM-30-PN-HRRP records: {(df['Measure Name'] == 'READM-30-PN-HRRP').sum()}")


**************************************************************************************************
import pandas as pd
import numpy as np

# Load your merged dataset
df = pd.read_csv('merged_social_determinants_analysis.csv')

print("=== STEP 1C: DATA CLEANING FOR LEVEL 1 VARIABLES ===\n")

print("BEFORE CLEANING:")
print("="*30)
print(f"Total sample size: {len(df)}")
print(f"Ownership_Category distribution:")
print(df['Ownership_Category'].value_counts())
print(f"Hospital_Rating_Numeric missing: {df['Hospital_Rating_Numeric'].isna().sum()}")

# Task 1C.1: Handle "Other" ownership category
print("\n\n1. HANDLING 'OTHER' OWNERSHIP CATEGORY")
print("="*50)

# Combine "Other" with "Private" (most common category)
df['Ownership_Category_Clean'] = df['Ownership_Category'].replace('Other', 'Private')

print("After combining 'Other' with 'Private':")
print(df['Ownership_Category_Clean'].value_counts())

# Task 1C.2: Handle missing Hospital_Rating_Numeric values
print("\n\n2. HANDLING MISSING HOSPITAL_RATING_NUMERIC VALUES")
print("="*50)

# Option 1: Create indicator variable for missing ratings
df['Rating_Missing'] = df['Hospital_Rating_Numeric'].isna().astype(int)

# Option 2: Mean imputation (we'll also create this version)
mean_rating = df['Hospital_Rating_Numeric'].mean()
df['Hospital_Rating_Numeric_Imputed'] = df['Hospital_Rating_Numeric'].fillna(mean_rating)

print(f"Created missing indicator: {df['Rating_Missing'].sum()} hospitals with missing ratings")
print(f"Mean imputation value: {mean_rating:.2f}")

# Task 1C.3: Create final Level 1 variables
print("\n\n3. FINAL LEVEL 1 VARIABLES SUMMARY")
print("="*50)

# Final variable selection
level1_vars = ['Ownership_Category_Clean', 'Hospital_Rating_Numeric_Imputed', 'Rating_Missing']

print("Selected Level 1 (Hospital) variables:")
for var in level1_vars:
    print(f"- {var}")

# Check final distributions
print(f"\nFinal Ownership_Category_Clean distribution:")
print(df['Ownership_Category_Clean'].value_counts())

print(f"\nFinal Hospital_Rating_Numeric_Imputed distribution:")
print(df['Hospital_Rating_Numeric_Imputed'].describe())

print(f"\nRating_Missing distribution:")
print(df['Rating_Missing'].value_counts())

# Task 1C.4: Check complete cases after cleaning
print("\n\n4. COMPLETE CASES CHECK")
print("="*50)

complete_cases = df[level1_vars + ['Excess Readmission Ratio']].dropna()
print(f"Complete cases for Level 1 analysis: {len(complete_cases)}")
print(f"Percentage of complete cases: {len(complete_cases)/len(df)*100:.1f}%")

# Task 1C.5: Save cleaned dataset (optional)
print("\n\n5. SAVE CLEANED DATASET")
print("="*50)

# Save the cleaned dataset
df.to_csv('merged_social_determinants_analysis_level1_cleaned.csv', index=False)
print("Cleaned dataset saved as: merged_social_determinants_analysis_level1_cleaned.csv")

print("\n\nSTEP 1C COMPLETED SUCCESSFULLY!")
print("="*50)
print("Ready for Step 2: Level 2 (County) Variable Selection")

*****************************************************************************************************************************************************
import pandas as pd
import numpy as np

# Load the cleaned dataset
df = pd.read_csv('merged_social_determinants_analysis_level1_cleaned.csv')

print("=== STEP 2A: COUNTY-LEVEL VARIABLES ANALYSIS ===\n")

# Define county-level variables by category
economic_vars = [
    'median_household_income_raw_value',
    'children_in_poverty_raw_value', 
    'unemployment_raw_value',
    'income_inequality_raw_value'
]

healthcare_access_vars = [
    'uninsured_adults_raw_value',
    'primary_care_physicians_raw_value',
    'mental_health_providers_raw_value'
]

demographic_vars = [
    '%_65_and_older_raw_value',
    '%_non_hispanic_black_raw_value',
    '%_hispanic_raw_value',
    '%_non_hispanic_white_raw_value',
    '%_rural_raw_value'
]

education_vars = [
    'some_college_raw_value',
    'high_school_graduation_raw_value'
]

print("1. ECONOMIC VARIABLES")
print("="*50)
for var in economic_vars:
    print(f"\n{var}:")
    print(f"  Count: {df[var].count()}")
    print(f"  Mean: {df[var].mean():.2f}")
    print(f"  Std: {df[var].std():.2f}")
    print(f"  Min: {df[var].min():.2f}")
    print(f"  Max: {df[var].max():.2f}")
    print(f"  Missing: {df[var].isna().sum()}")

print("\n\n2. HEALTHCARE ACCESS VARIABLES")
print("="*50)
for var in healthcare_access_vars:
    print(f"\n{var}:")
    print(f"  Count: {df[var].count()}")
    print(f"  Mean: {df[var].mean():.2f}")
    print(f"  Std: {df[var].std():.2f}")
    print(f"  Min: {df[var].min():.2f}")
    print(f"  Max: {df[var].max():.2f}")
    print(f"  Missing: {df[var].isna().sum()}")

print("\n\n3. DEMOGRAPHIC VARIABLES")
print("="*50)
for var in demographic_vars:
    print(f"\n{var}:")
    print(f"  Count: {df[var].count()}")
    print(f"  Mean: {df[var].mean():.2f}")
    print(f"  Std: {df[var].std():.2f}")
    print(f"  Min: {df[var].min():.2f}")
    print(f"  Max: {df[var].max():.2f}")
    print(f"  Missing: {df[var].isna().sum()}")

print("\n\n4. EDUCATION VARIABLES")
print("="*50)
for var in education_vars:
    print(f"\n{var}:")
    print(f"  Count: {df[var].count()}")
    print(f"  Mean: {df[var].mean():.2f}")
    print(f"  Std: {df[var].std():.2f}")
    print(f"  Min: {df[var].min():.2f}")
    print(f"  Max: {df[var].max():.2f}")
    print(f"  Missing: {df[var].isna().sum()}")

print("\n\n5. COUNTY CLUSTERING CHECK")
print("="*50)
print(f"Number of unique counties: {df['county_fips_code'].nunique()}")
print(f"Hospitals per county (mean): {len(df) / df['county_fips_code'].nunique():.2f}")
print(f"Counties with multiple hospitals: {(df['county_fips_code'].value_counts() > 1).sum()}")


**********************************************************************************************
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv('merged_social_determinants_analysis_level1_cleaned.csv')

print("=== STEP 2C: HEALTHCARE ACCESS RATIO VARIABLES CHECK ===\n")

# Check if ratio variables exist
ratio_vars = [
    'ratio_of_population_to_primary_care_physicians',
    'ratio_of_population_to_mental_health_providers'
]

for var in ratio_vars:
    if var in df.columns:
        print(f"\n{var}:")
        print(f"  Count: {df[var].count()}")
        print(f"  Mean: {df[var].mean():.2f}")
        print(f"  Std: {df[var].std():.2f}")
        print(f"  Min: {df[var].min():.2f}")
        print(f"  Max: {df[var].max():.2f}")
        print(f"  Missing: {df[var].isna().sum()}")
    else:
        print(f"\n{var}: NOT FOUND in dataset")

# Also check what other healthcare-related variables we have
print("\n\nALL AVAILABLE HEALTHCARE-RELATED VARIABLES:")
print("="*50)
healthcare_keywords = ['care', 'physician', 'provider', 'health', 'uninsured']
healthcare_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in healthcare_keywords)]
for col in healthcare_cols:
    print(f"- {col}")




******************************************************************************************************************************************************
import pandas as pd
import numpy as np

# Load the cleaned dataset
df = pd.read_csv('merged_social_determinants_analysis_level1_cleaned.csv')

print("=== STEP 2D: FINAL LEVEL 2 VARIABLE SELECTION & CLEANING ===\n")

# Define final Level 2 (County) variables
level2_vars = [
    # Economic (2 variables)
    'median_household_income_raw_value',
    'children_in_poverty_raw_value',
    
    # Healthcare Access (2 variables)
    'uninsured_adults_raw_value',
    'ratio_of_population_to_primary_care_physicians',
    
    # Demographics (2 variables)
    '%_rural_raw_value',
    '%_non_hispanic_white_raw_value',
    
    # Education (1 variable)
    'some_college_raw_value'
]

print("FINAL LEVEL 2 (COUNTY) VARIABLES:")
print("="*50)
for i, var in enumerate(level2_vars, 1):
    print(f"{i}. {var}")

# Check missing data pattern
print("\n\nMISSING DATA ANALYSIS:")
print("="*50)
missing_summary = df[level2_vars].isnull().sum()
print(missing_summary)

# Check if missing data is from same counties
print(f"\nHospitals with any missing Level 2 data: {df[level2_vars].isnull().any(axis=1).sum()}")

# Task 2D.1: Handle missing data
print("\n\n1. HANDLING MISSING DATA")
print("="*50)

# Option 1: Complete case analysis (recommended for multilevel models)
df_complete = df.dropna(subset=level2_vars + ['Excess Readmission Ratio'])

print(f"Original sample: {len(df)} hospitals")
print(f"Complete cases: {len(df_complete)} hospitals")
print(f"Percentage retained: {len(df_complete)/len(df)*100:.1f}%")

# Task 2D.2: Create county grouping variable
print("\n\n2. COUNTY GROUPING VARIABLE")
print("="*50)

# Use county_fips_code as the grouping variable
df_complete['county_group'] = df_complete['county_fips_code']

print(f"Number of counties in final sample: {df_complete['county_group'].nunique()}")
print(f"Average hospitals per county: {len(df_complete) / df_complete['county_group'].nunique():.2f}")

# Check county distribution
county_counts = df_complete['county_group'].value_counts()
print(f"Counties with 1 hospital: {(county_counts == 1).sum()}")
print(f"Counties with 2-5 hospitals: {((county_counts >= 2) & (county_counts <= 5)).sum()}")
print(f"Counties with 6+ hospitals: {(county_counts >= 6).sum()}")

# Task 2D.3: Standardize continuous variables
print("\n\n3. STANDARDIZING CONTINUOUS VARIABLES")
print("="*50)

# Standardize Level 2 variables (z-score)
level2_vars_std = []
for var in level2_vars:
    var_std = var + '_std'
    df_complete[var_std] = (df_complete[var] - df_complete[var].mean()) / df_complete[var].std()
    level2_vars_std.append(var_std)
    print(f"Standardized {var} -> {var_std}")

# Also standardize Level 1 continuous variable
df_complete['Hospital_Rating_Numeric_Imputed_std'] = (
    df_complete['Hospital_Rating_Numeric_Imputed'] - df_complete['Hospital_Rating_Numeric_Imputed'].mean()
) / df_complete['Hospital_Rating_Numeric_Imputed'].std()

# Task 2D.4: Final variable summary
print("\n\n4. FINAL HIERARCHICAL MODEL VARIABLES")
print("="*50)

level1_vars_final = [
    'Ownership_Category_Clean',
    'Hospital_Rating_Numeric_Imputed_std',
    'Rating_Missing'
]

print("LEVEL 1 (HOSPITAL) VARIABLES:")
for var in level1_vars_final:
    print(f"  - {var}")

print("\nLEVEL 2 (COUNTY) VARIABLES:")
for var in level2_vars_std:
    print(f"  - {var}")

print(f"\nCOUNTY GROUPING VARIABLE: county_group")
print(f"PRIMARY OUTCOME: Excess Readmission Ratio")

# Task 2D.5: Save final dataset
print("\n\n5. SAVE FINAL DATASET")
print("="*50)

final_vars = level1_vars_final + level2_vars_std + ['county_group', 'Excess Readmission Ratio', 'Facility ID', 'Facility Name', 'State']
df_final = df_complete[final_vars].copy()

df_final.to_csv('hierarchical_model_final_dataset.csv', index=False)
print(f"Final dataset saved: hierarchical_model_final_dataset.csv")
print(f"Final sample size: {len(df_final)} hospitals")
print(f"Final county count: {df_final['county_group'].nunique()} counties")

print("\n\nSTEP 2D COMPLETED - READY FOR HIERARCHICAL MODELING!")
