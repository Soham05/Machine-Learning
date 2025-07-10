import pandas as pd
import numpy as np

# Load the readmissions dataset
# Replace 'your_file_path.csv' with the actual path to your CSV file
df = pd.read_csv('FY_2025_HOSPITAL_READMISSIONS_REDUCTION_PROGRAM_HOSPITAL.CSV')

print("=" * 60)
print("READMISSIONS DATASET EXPLORATION")
print("=" * 60)

# 1. Basic dataset shape and info
print("\n1. DATASET OVERVIEW:")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {df.shape[1]}")
print(f"Dataset shape: {df.shape}")

# 2. Column names and data types
print("\n2. COLUMN INFORMATION:")
print(df.dtypes)

# 3. How many unique hospitals (Facility IDs)?
unique_hospitals = df['Facility ID'].nunique()
print(f"\n3. UNIQUE HOSPITALS:")
print(f"Number of unique hospitals: {unique_hospitals:,}")

# 4. What are ALL the unique "Measure Name" values?
print("\n4. ALL READMISSION MEASURES:")
unique_measures = df['Measure Name'].unique()
print(f"Number of different measures: {len(unique_measures)}")
print("List of all measures:")
for i, measure in enumerate(sorted(unique_measures), 1):
    measure_count = df[df['Measure Name'] == measure].shape[0]
    print(f"{i:2d}. {measure} (n={measure_count:,} hospitals)")

# 5. State distribution
print("\n5. GEOGRAPHIC DISTRIBUTION:")
state_counts = df['State'].value_counts()
print(f"Number of states: {len(state_counts)}")
print("Top 10 states by number of hospital-measure combinations:")
print(state_counts.head(10))

# 6. Missing values analysis
print("\n6. MISSING VALUES ANALYSIS:")
missing_summary = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_summary,
    'Missing Percentage': missing_percent.round(2)
})
print(missing_df[missing_df['Missing Count'] > 0])

# 7. Key numeric variables summary
print("\n7. KEY NUMERIC VARIABLES SUMMARY:")
numeric_cols = ['Number of Discharges', 'Excess Readmission Ratio', 
                'Predicted Readmission Rate', 'Expected Readmission Rate', 
                'Number of Readmissions']

for col in numeric_cols:
    if col in df.columns:
        # Convert to numeric, handling any non-numeric values
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.4f}")
        print(f"  Median: {df[col].median():.4f}")
        print(f"  Min: {df[col].min():.4f}")
        print(f"  Max: {df[col].max():.4f}")
        print(f"  Missing: {df[col].isnull().sum()}")

# 8. Sample of data for verification
print("\n8. SAMPLE DATA (First 5 rows):")
print(df.head())

# 9. Check for hospitals with multiple measures
print("\n9. HOSPITAL MEASURE COVERAGE:")
hospital_measure_counts = df.groupby('Facility ID')['Measure Name'].count()
print(f"Average measures per hospital: {hospital_measure_counts.mean():.2f}")
print(f"Min measures per hospital: {hospital_measure_counts.min()}")
print(f"Max measures per hospital: {hospital_measure_counts.max()}")

# 10. Time period verification
print("\n10. TIME PERIOD VERIFICATION:")
print(f"Start dates: {df['Start Date'].unique()}")
print(f"End dates: {df['End Date'].unique()}")

# 11. Check for any footnote patterns
print("\n11. FOOTNOTE ANALYSIS:")
if 'Footnote' in df.columns:
    footnote_counts = df['Footnote'].value_counts()
    print("Footnote distribution:")
    print(footnote_counts.head(10))

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE")
print("=" * 60)
