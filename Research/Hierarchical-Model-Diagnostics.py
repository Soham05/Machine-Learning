#!/usr/bin/env python3
"""
Data Structure Diagnostic Script
Identifies issues preventing hierarchical model fitting
"""

import pandas as pd
import numpy as np
from statsmodels.formula.api import mixedlm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DATA STRUCTURE DIAGNOSTIC")
print("="*80)

# Load data
df = pd.read_csv('hierarchical_model_final_dataset.csv')

print("\n1. BASIC DATA STRUCTURE")
print("-" * 50)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check for problematic column names
problematic_cols = [col for col in df.columns if ' ' in col or '-' in col or col.startswith('%')]
print(f"\nProblematic column names (spaces/symbols): {problematic_cols}")

print("\n2. OUTCOME VARIABLE ANALYSIS")
print("-" * 50)
outcome_col = 'Excess Readmission Ratio'
if outcome_col in df.columns:
    print(f"Original outcome column: '{outcome_col}'")
    print(f"Data type: {df[outcome_col].dtype}")
    print(f"Missing values: {df[outcome_col].isnull().sum()}")
    print(f"Unique values: {df[outcome_col].nunique()}")
    print(f"Sample values: {df[outcome_col].head().tolist()}")
    
    # Create clean version
    df['ERR'] = df[outcome_col]
    print(f"\nCreated clean column 'ERR'")
    print(f"ERR data type: {df['ERR'].dtype}")
    print(f"ERR missing values: {df['ERR'].isnull().sum()}")

print("\n3. GROUPING VARIABLE ANALYSIS")
print("-" * 50)
group_col = 'county_group'
if group_col in df.columns:
    print(f"Grouping column: '{group_col}'")
    print(f"Data type: {df[group_col].dtype}")
    print(f"Missing values: {df[group_col].isnull().sum()}")
    print(f"Unique groups: {df[group_col].nunique()}")
    print(f"Sample values: {df[group_col].head().tolist()}")
    
    # Check for problematic group values
    if df[group_col].dtype == 'object':
        print(f"Group values contain strings - converting to numeric")
        df['county_group_numeric'] = pd.factorize(df[group_col])[0]
        print(f"Created county_group_numeric")
else:
    print(f"❌ Grouping column '{group_col}' not found!")

print("\n4. PREDICTOR VARIABLES ANALYSIS")
print("-" * 50)

# Level 1 variables
level1_vars = ['Ownership_Category_Clean', 'Hospital_Rating_Numeric_Imputed_std', 'Rating_Missing']
print("Level 1 (Hospital) Variables:")
for var in level1_vars:
    if var in df.columns:
        print(f"  ✅ {var}: {df[var].dtype}, {df[var].isnull().sum()} missing")
        if df[var].dtype == 'object':
            print(f"      Categories: {df[var].value_counts().to_dict()}")
    else:
        print(f"  ❌ {var}: NOT FOUND")
        # Try to find similar columns
        similar = [col for col in df.columns if any(word in col.lower() for word in var.lower().split('_'))]
        if similar:
            print(f"      Similar columns: {similar}")

# Level 2 variables
level2_vars = [
    'median_household_income_raw_value_std',
    'children_in_poverty_raw_value_std', 
    'uninsured_adults_raw_value_std',
    'ratio_of_population_to_primary_care_physicians_std',
    '%_rural_raw_value_std',
    '%_non_hispanic_white_raw_value_std',
    'some_college_raw_value_std'
]

print("\nLevel 2 (County) Variables:")
for var in level2_vars:
    if var in df.columns:
        print(f"  ✅ {var}: {df[var].dtype}, {df[var].isnull().sum()} missing")
    else:
        print(f"  ❌ {var}: NOT FOUND")

print("\n5. CLEAN COLUMN NAMES FOR MODELING")
print("-" * 50)

# Create clean column names
df_clean = df.copy()

# Clean outcome
if 'Excess Readmission Ratio' in df_clean.columns:
    df_clean['ERR'] = df_clean['Excess Readmission Ratio']

# Clean group variable
if 'county_group' in df_clean.columns:
    df_clean['county_id'] = pd.factorize(df_clean['county_group'])[0]

# Clean predictor variables
column_mapping = {}
for col in df_clean.columns:
    clean_col = col.replace(' ', '_').replace('%', 'pct').replace('-', '_')
    if clean_col != col:
        df_clean[clean_col] = df_clean[col]
        column_mapping[col] = clean_col

print(f"Column mapping applied: {column_mapping}")

print("\n6. TEST SIMPLE MODEL")
print("-" * 50)

try:
    # Test with simplest possible model
    if 'ERR' in df_clean.columns and 'county_id' in df_clean.columns:
        # Remove any rows with missing data
        test_df = df_clean[['ERR', 'county_id']].dropna()
        print(f"Test dataset shape: {test_df.shape}")
        
        # Test null model
        null_model = mixedlm("ERR ~ 1", data=test_df, groups=test_df['county_id'])
        null_results = null_model.fit()
        print("✅ Null model fitted successfully!")
        print(f"   - Log-likelihood: {null_results.llf:.2f}")
        print(f"   - AIC: {null_results.aic:.2f}")
        
        # Calculate ICC
        sigma2_u = null_results.cov_re.iloc[0, 0]
        sigma2_e = null_results.scale
        icc = sigma2_u / (sigma2_u + sigma2_e)
        print(f"   - ICC: {icc:.4f} ({icc*100:.1f}% county-level variance)")
        
except Exception as e:
    print(f"❌ Error in test model: {str(e)}")
    print("   Detailed error information:")
    import traceback
    traceback.print_exc()

print("\n7. RECOMMENDED FIXES")
print("-" * 50)

# Save cleaned dataset
df_clean.to_csv('hierarchical_model_cleaned_dataset.csv', index=False)
print("✅ Saved cleaned dataset: 'hierarchical_model_cleaned_dataset.csv'")

print("\nRecommended variable names for modeling:")
print("- Outcome: 'ERR'")
print("- Grouping: 'county_id'")
print("- Level 1 vars: Check availability in cleaned dataset")
print("- Level 2 vars: Check availability in cleaned dataset")

print("\n8. FINAL DATA SUMMARY")
print("-" * 50)
print(f"Final cleaned dataset shape: {df_clean.shape}")
print(f"Complete cases for ERR and county_id: {df_clean[['ERR', 'county_id']].dropna().shape[0]}")
print(f"Counties in final dataset: {df_clean['county_id'].nunique()}")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
