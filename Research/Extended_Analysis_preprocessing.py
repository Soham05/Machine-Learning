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
