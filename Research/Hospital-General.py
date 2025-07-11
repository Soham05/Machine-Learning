# Hospital General Information Dataset: Preprocessing & Analysis Plan

## Phase 1: Data Quality Assessment

### 1.1 Missing Data Analysis
```python
# Check missing data patterns
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
print("Missing Data Summary:")
print(pd.DataFrame({'Missing_Count': missing_data, 'Missing_Percentage': missing_percentage}))
```

### 1.2 Key Variables Inspection
**Priority Variables for Analysis:**
- `Facility ID` (Primary key for merging with readmissions data)
- `County/Parish` (Critical for linking to county health rankings)
- `State` (Geographic validation)
- `Hospital Type` (Control variable)
- `Hospital Ownership` (Control variable)
- `Emergency Services` (Control variable)
- `Hospital overall rating` (Control variable)

### 1.3 Data Type Validation
```python
# Check data types and unique values
for col in ['Facility ID', 'County/Parish', 'State', 'Hospital Type', 'Hospital Ownership']:
    print(f"\n{col}:")
    print(f"  Data type: {df[col].dtype}")
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Sample values: {df[col].value_counts().head(3)}")
```

## Phase 2: Geographic Data Standardization

### 2.1 County Name Standardization
**Critical Step:** County names must match between datasets for successful merging.

```python
# Standardize county names
def standardize_county_names(county_name):
    if pd.isna(county_name):
        return county_name
    
    # Convert to string and clean
    county_clean = str(county_name).strip().upper()
    
    # Remove common suffixes
    county_clean = county_clean.replace(' COUNTY', '')
    county_clean = county_clean.replace(' PARISH', '')
    county_clean = county_clean.replace(' BOROUGH', '')
    
    return county_clean

df['County_Standardized'] = df['County/Parish'].apply(standardize_county_names)
```

### 2.2 State Standardization
```python
# Ensure state codes are consistent
print("State distribution:")
print(df['State'].value_counts())

# Check for any non-standard state codes
standard_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC']
```

### 2.3 ZIP Code Analysis
```python
# Analyze ZIP codes for potential county mapping backup
print(f"ZIP Code range: {df['ZIP Code'].min()} - {df['ZIP Code'].max()}")
print(f"ZIP Code missing: {df['ZIP Code'].isnull().sum()}")
```

## Phase 3: Hospital Characteristics Processing

### 3.1 Hospital Type Categorization
```python
# Analyze hospital types
print("Hospital Type Distribution:")
print(df['Hospital Type'].value_counts())

# Create simplified categories if needed
def categorize_hospital_type(hospital_type):
    if pd.isna(hospital_type):
        return 'Unknown'
    
    hospital_type = str(hospital_type).upper()
    
    if 'ACUTE' in hospital_type:
        return 'Acute Care'
    elif 'CRITICAL' in hospital_type:
        return 'Critical Access'
    elif 'SPECIALTY' in hospital_type:
        return 'Specialty'
    else:
        return 'Other'

df['Hospital_Type_Category'] = df['Hospital Type'].apply(categorize_hospital_type)
```

### 3.2 Hospital Ownership Processing
```python
# Analyze ownership patterns
print("Hospital Ownership Distribution:")
print(df['Hospital Ownership'].value_counts())

# Create ownership categories
def categorize_ownership(ownership):
    if pd.isna(ownership):
        return 'Unknown'
    
    ownership = str(ownership).upper()
    
    if 'GOVERNMENT' in ownership or 'PUBLIC' in ownership:
        return 'Public'
    elif 'PROPRIETARY' in ownership or 'PRIVATE' in ownership:
        return 'Private'
    elif 'VOLUNTARY' in ownership or 'NON-PROFIT' in ownership:
        return 'Non-Profit'
    else:
        return 'Other'

df['Ownership_Category'] = df['Hospital Ownership'].apply(categorize_ownership)
```

### 3.3 Emergency Services Processing
```python
# Process emergency services
print("Emergency Services Distribution:")
print(df['Emergency Services'].value_counts())

# Convert to binary
df['Has_Emergency_Services'] = df['Emergency Services'].map({'Yes': 1, 'No': 0})
```

### 3.4 Hospital Rating Processing
```python
# Analyze hospital ratings
print("Hospital Rating Distribution:")
print(df['Hospital overall rating'].value_counts())

# Convert to numeric (handle 'Not Available' cases)
def convert_rating(rating):
    if pd.isna(rating) or rating == 'Not Available':
        return None
    try:
        return int(rating)
    except:
        return None

df['Hospital_Rating_Numeric'] = df['Hospital overall rating'].apply(convert_rating)
```

## Phase 4: Data Integration Preparation

### 4.1 Create Merge Keys
```python
# Create standardized merge key for county health rankings
df['State_County_Key'] = df['State'] + '_' + df['County_Standardized']

# Verify Facility ID format for readmissions merge
print(f"Facility ID format check:")
print(f"  Length range: {df['Facility ID'].str.len().min()} - {df['Facility ID'].str.len().max()}")
print(f"  Sample IDs: {df['Facility ID'].head()}")
```

### 4.2 Geographic Coverage Analysis
```python
# Analyze geographic coverage
print("Geographic Coverage Analysis:")
print(f"Total hospitals: {len(df)}")
print(f"Unique states: {df['State'].nunique()}")
print(f"Unique counties: {df['County_Standardized'].nunique()}")

# State-level distribution
state_counts = df['State'].value_counts()
print(f"\nTop 10 states by hospital count:")
print(state_counts.head(10))
```

### 4.3 Data Completeness Assessment
```python
# Assess completeness of key variables
key_vars = ['Facility ID', 'County_Standardized', 'State', 'Hospital_Type_Category', 
           'Ownership_Category', 'Has_Emergency_Services']

completeness = {}
for var in key_vars:
    completeness[var] = {
        'Complete': df[var].notna().sum(),
        'Missing': df[var].isna().sum(),
        'Completeness_Rate': (df[var].notna().sum() / len(df)) * 100
    }

completeness_df = pd.DataFrame(completeness).T
print("Data Completeness for Key Variables:")
print(completeness_df)
```

## Phase 5: Quality Checks & Validation

### 5.1 Duplicate Detection
```python
# Check for duplicate facilities
duplicates = df[df.duplicated(subset=['Facility ID'], keep=False)]
print(f"Duplicate Facility IDs: {len(duplicates)}")

if len(duplicates) > 0:
    print("Duplicate facilities found:")
    print(duplicates[['Facility ID', 'Facility Name', 'State', 'County_Standardized']])
```

### 5.2 Geographic Validation
```python
# Cross-validate state and county combinations
state_county_combos = df.groupby(['State', 'County_Standardized']).size().reset_index(name='Hospital_Count')
print(f"Unique state-county combinations: {len(state_county_combos)}")

# Look for potential data quality issues
suspicious_combos = state_county_combos[state_county_combos['Hospital_Count'] > 20]
print(f"Counties with >20 hospitals (potential data quality check needed):")
print(suspicious_combos)
```

## Phase 6: Final Dataset Creation

### 6.1 Create Analysis-Ready Dataset
```python
# Select final variables for analysis
analysis_vars = [
    'Facility ID',                    # Primary key
    'Facility Name',                  # Reference
    'State',                         # Geographic
    'County_Standardized',           # Geographic (for county health rankings merge)
    'State_County_Key',              # Merge key
    'Hospital_Type_Category',        # Control variable
    'Ownership_Category',            # Control variable
    'Has_Emergency_Services',        # Control variable
    'Hospital_Rating_Numeric',       # Control variable
    'ZIP Code'                       # Backup geographic identifier
]

hospital_analysis_df = df[analysis_vars].copy()
```

### 6.2 Export for Integration
```python
# Save preprocessed dataset
hospital_analysis_df.to_csv('hospital_general_preprocessed.csv', index=False)

# Create summary statistics
summary_stats = hospital_analysis_df.describe(include='all')
print("Final Dataset Summary:")
print(summary_stats)
```

## Expected Outcomes

After completing this preprocessing:

1. **Clean hospital characteristics data** ready for merging
2. **Standardized county names** for linking to county health rankings
3. **Validated Facility IDs** for linking to readmissions data
4. **Categorical variables** ready for hierarchical modeling
5. **Quality-checked dataset** with known completeness rates

## Critical Success Factors

1. **County name standardization** - This is crucial for your county health rankings merge
2. **Facility ID validation** - Must match your readmissions dataset exactly
3. **Missing data handling** - Document any systematic missingness patterns
4. **Geographic coverage** - Ensure adequate representation across states

## Next Steps After Preprocessing

1. **Test merge** with county health rankings data
2. **Test merge** with readmissions data
3. **Assess final sample size** after all merges
4. **Validate geographic distribution** of final analytical sample
