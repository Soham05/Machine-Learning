import pandas as pd
import numpy as np
import re

# STEP 1: LOAD AND CLEAN EACH YEAR'S DATA

# Replace these with your actual file paths
file_2020 = 'county_health_2020.csv'
file_2021 = 'county_health_2021.csv'
file_2022 = 'county_health_2022.csv'
file_2023 = 'county_health_2023.csv'
file_2024 = 'county_health_2024.csv'

# Load 2020 data
df_2020 = pd.read_csv(file_2020)
# Use first row as column names
df_2020.columns = df_2020.iloc[0]
df_2020 = df_2020.drop(df_2020.index[0]).reset_index(drop=True)

# Clean column names for 2020
cleaned_columns = []
for col in df_2020.columns:
    if pd.isna(col):
        cleaned_columns.append('unnamed_column')
    else:
        clean_col = str(col).lower()
        clean_col = re.sub(r'[/\.&\-\s\(\)\,\%\#\+\=\[\]]+', '_', clean_col)
        clean_col = re.sub(r'_+', '_', clean_col)
        clean_col = clean_col.strip('_')
        cleaned_columns.append(clean_col)

df_2020.columns = cleaned_columns
df_2020['data_year'] = 2020
print(f"2020 data shape: {df_2020.shape}")

# Load 2021 data
df_2021 = pd.read_csv(file_2021)
df_2021.columns = df_2021.iloc[0]
df_2021 = df_2021.drop(df_2021.index[0]).reset_index(drop=True)

# Clean column names for 2021
cleaned_columns = []
for col in df_2021.columns:
    if pd.isna(col):
        cleaned_columns.append('unnamed_column')
    else:
        clean_col = str(col).lower()
        clean_col = re.sub(r'[/\.&\-\s\(\)\,\%\#\+\=\[\]]+', '_', clean_col)
        clean_col = re.sub(r'_+', '_', clean_col)
        clean_col = clean_col.strip('_')
        cleaned_columns.append(clean_col)

df_2021.columns = cleaned_columns
df_2021['data_year'] = 2021
print(f"2021 data shape: {df_2021.shape}")

# Load 2022 data
df_2022 = pd.read_csv(file_2022)
df_2022.columns = df_2022.iloc[0]
df_2022 = df_2022.drop(df_2022.index[0]).reset_index(drop=True)

# Clean column names for 2022
cleaned_columns = []
for col in df_2022.columns:
    if pd.isna(col):
        cleaned_columns.append('unnamed_column')
    else:
        clean_col = str(col).lower()
        clean_col = re.sub(r'[/\.&\-\s\(\)\,\%\#\+\=\[\]]+', '_', clean_col)
        clean_col = re.sub(r'_+', '_', clean_col)
        clean_col = clean_col.strip('_')
        cleaned_columns.append(clean_col)

df_2022.columns = cleaned_columns
df_2022['data_year'] = 2022
print(f"2022 data shape: {df_2022.shape}")

# Load 2023 data
df_2023 = pd.read_csv(file_2023)
df_2023.columns = df_2023.iloc[0]
df_2023 = df_2023.drop(df_2023.index[0]).reset_index(drop=True)

# Clean column names for 2023
cleaned_columns = []
for col in df_2023.columns:
    if pd.isna(col):
        cleaned_columns.append('unnamed_column')
    else:
        clean_col = str(col).lower()
        clean_col = re.sub(r'[/\.&\-\s\(\)\,\%\#\+\=\[\]]+', '_', clean_col)
        clean_col = re.sub(r'_+', '_', clean_col)
        clean_col = clean_col.strip('_')
        cleaned_columns.append(clean_col)

df_2023.columns = cleaned_columns
df_2023['data_year'] = 2023
print(f"2023 data shape: {df_2023.shape}")

# Load 2024 data
df_2024 = pd.read_csv(file_2024)
df_2024.columns = df_2024.iloc[0]
df_2024 = df_2024.drop(df_2024.index[0]).reset_index(drop=True)

# Clean column names for 2024
cleaned_columns = []
for col in df_2024.columns:
    if pd.isna(col):
        cleaned_columns.append('unnamed_column')
    else:
        clean_col = str(col).lower()
        clean_col = re.sub(r'[/\.&\-\s\(\)\,\%\#\+\=\[\]]+', '_', clean_col)
        clean_col = re.sub(r'_+', '_', clean_col)
        clean_col = clean_col.strip('_')
        cleaned_columns.append(clean_col)

df_2024.columns = cleaned_columns
df_2024['data_year'] = 2024
print(f"2024 data shape: {df_2024.shape}")

# STEP 2: COMBINE ALL YEARS
combined_df = pd.concat([df_2020, df_2021, df_2022, df_2023, df_2024], ignore_index=True)
print(f"\nCombined data shape: {combined_df.shape}")

# STEP 3: EXTRACT REQUIRED COLUMNS
required_columns = [
    # Identification columns
    'statecode', 'countycode', 'fipscode', 'state', 'county', 'year', 'data_year',
    
    # Primary phase variables
    'v063_rawvalue', 'v063_cilow', 'v063_cihigh',  # Median Household Income
    'v024_rawvalue', 'v024_cilow', 'v024_cihigh',  # Children in Poverty
    'v023_rawvalue',  # Unemployment
    'v044_rawvalue',  # Income Inequality
    'v069_rawvalue',  # Some College
    'v168_rawvalue',  # High School Completion
    'v085_rawvalue',  # Uninsured
    'v004_rawvalue', 'v004_other_data_1',  # Primary Care Physicians
    'v062_rawvalue', 'v062_other_data_1',  # Mental Health Providers
    'v136_rawvalue',  # Severe Housing Problems
    'v140_rawvalue',  # Social Associations
    'v141_rawvalue',  # Residential Segregation
    'v053_rawvalue',  # % 65 and Older
    'v054_rawvalue',  # % Non-Hispanic Black
    'v056_rawvalue',  # % Hispanic
    'v058_rawvalue',  # % Rural
    'v051_rawvalue',  # Population
    
    # Secondary phase variables
    'v009_rawvalue',  # Adult Smoking
    'v011_rawvalue',  # Adult Obesity
    'v049_rawvalue',  # Excessive Drinking
    'v070_rawvalue',  # Physical Inactivity
    'v002_rawvalue',  # Poor or Fair Health
    'v042_rawvalue',  # Poor Mental Health Days
    'v036_rawvalue',  # Poor Physical Health Days
    'v082_rawvalue',  # Children in Single-Parent Households
    'v139_rawvalue',  # Food Insecurity
    'v153_rawvalue',  # Homeownership
    'v059_rawvalue',  # % Not Proficient in English
    'v003_rawvalue',  # Uninsured Adults
    'v122_rawvalue',  # Uninsured Children
    'v131_rawvalue',  # Other Primary Care Providers
]

# Check which columns exist
available_columns = []
missing_columns = []

for col in required_columns:
    if col in combined_df.columns:
        available_columns.append(col)
    else:
        missing_columns.append(col)

print(f"\nAvailable columns: {len(available_columns)}")
print(f"Missing columns: {len(missing_columns)}")

if missing_columns:
    print("\nMissing columns:")
    for col in missing_columns:
        print(f"  - {col}")

# Extract available columns
final_df = combined_df[available_columns].copy()
print(f"\nFinal dataset shape: {final_df.shape}")

# STEP 4: DATA QUALITY ANALYSIS

print("\n" + "="*50)
print("DATA QUALITY ANALYSIS")
print("="*50)

# Basic information
print(f"\n1. BASIC INFORMATION:")
print(f"   Shape: {final_df.shape}")
print(f"   Memory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Year distribution
print(f"\n2. YEAR DISTRIBUTION:")
if 'data_year' in final_df.columns:
    year_counts = final_df['data_year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"   {year}: {count:,} counties")

# Geographic coverage
print(f"\n3. GEOGRAPHIC COVERAGE:")
if 'state' in final_df.columns:
    print(f"   Unique states: {final_df['state'].nunique()}")
if 'county' in final_df.columns:
    print(f"   Unique counties: {final_df['county'].nunique()}")
if 'fipscode' in final_df.columns:
    print(f"   Unique FIPS codes: {final_df['fipscode'].nunique()}")

# Missing data analysis
print(f"\n4. MISSING DATA ANALYSIS:")
missing_data = final_df.isnull().sum()
missing_percent = (missing_data / len(final_df)) * 100

missing_summary = pd.DataFrame({
    'Missing_Count': missing_data,
    'Missing_Percent': missing_percent
}).sort_values('Missing_Percent', ascending=False)

# Show variables with missing data
variables_with_missing = missing_summary[missing_summary['Missing_Percent'] > 0]
print(f"   Variables with missing data: {len(variables_with_missing)}")
print("\n   Top 15 variables with most missing data:")
for idx, row in variables_with_missing.head(15).iterrows():
    print(f"   {idx}: {row['Missing_Count']:,} ({row['Missing_Percent']:.1f}%)")

# Data completeness by year
print(f"\n5. DATA COMPLETENESS BY YEAR:")
if 'data_year' in final_df.columns:
    for year in sorted(final_df['data_year'].unique()):
        year_data = final_df[final_df['data_year'] == year]
        total_cells = year_data.shape[0] * year_data.shape[1]
        missing_cells = year_data.isnull().sum().sum()
        completeness = (1 - missing_cells / total_cells) * 100
        print(f"   {year}: {completeness:.1f}% complete ({missing_cells:,} missing cells)")

# Key variables analysis
print(f"\n6. KEY VARIABLES ANALYSIS:")

# Economic variables
economic_vars = ['v063_rawvalue', 'v024_rawvalue', 'v023_rawvalue', 'v044_rawvalue']
print("\n   ECONOMIC VARIABLES:")
for var in economic_vars:
    if var in final_df.columns:
        missing_pct = (final_df[var].isnull().sum() / len(final_df)) * 100
        print(f"   {var}: {missing_pct:.1f}% missing")
        if final_df[var].dtype in ['int64', 'float64']:
            try:
                print(f"      Mean: {final_df[var].mean():.2f}, Std: {final_df[var].std():.2f}")
            except:
                print(f"      Cannot calculate statistics")

# Healthcare variables
healthcare_vars = ['v085_rawvalue', 'v004_rawvalue', 'v062_rawvalue']
print("\n   HEALTHCARE ACCESS VARIABLES:")
for var in healthcare_vars:
    if var in final_df.columns:
        missing_pct = (final_df[var].isnull().sum() / len(final_df)) * 100
        print(f"   {var}: {missing_pct:.1f}% missing")
        if final_df[var].dtype in ['int64', 'float64']:
            try:
                print(f"      Mean: {final_df[var].mean():.2f}, Std: {final_df[var].std():.2f}")
            except:
                print(f"      Cannot calculate statistics")

# Education variables
education_vars = ['v069_rawvalue', 'v168_rawvalue']
print("\n   EDUCATION VARIABLES:")
for var in education_vars:
    if var in final_df.columns:
        missing_pct = (final_df[var].isnull().sum() / len(final_df)) * 100
        print(f"   {var}: {missing_pct:.1f}% missing")
        if final_df[var].dtype in ['int64', 'float64']:
            try:
                print(f"      Mean: {final_df[var].mean():.2f}, Std: {final_df[var].std():.2f}")
            except:
                print(f"      Cannot calculate statistics")

# Demographics variables
demo_vars = ['v053_rawvalue', 'v054_rawvalue', 'v056_rawvalue', 'v058_rawvalue']
print("\n   DEMOGRAPHIC VARIABLES:")
for var in demo_vars:
    if var in final_df.columns:
        missing_pct = (final_df[var].isnull().sum() / len(final_df)) * 100
        print(f"   {var}: {missing_pct:.1f}% missing")
        if final_df[var].dtype in ['int64', 'float64']:
            try:
                print(f"      Mean: {final_df[var].mean():.2f}, Std: {final_df[var].std():.2f}")
            except:
                print(f"      Cannot calculate statistics")

print(f"\n7. SAMPLE FIRST 5 ROWS:")
print(final_df.head())

print(f"\n8. COLUMN NAMES:")
print("Available columns in final dataset:")
for i, col in enumerate(final_df.columns):
    print(f"   {i+1}. {col}")

# Save the processed data
final_df.to_csv('county_health_combined_processed.csv', index=False)
print(f"\nProcessed data saved to: county_health_combined_processed.csv")
