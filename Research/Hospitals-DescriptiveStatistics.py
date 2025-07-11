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
