import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, List

def standardize_county_names(county_series: pd.Series) -> pd.Series:
    """
    Standardize county names by removing common suffixes and cleaning formatting
    """
    # Convert to string and handle missing values
    county_clean = county_series.astype(str).str.strip()
    
    # Remove common county suffixes
    suffixes_to_remove = [
        r'\s+County$', r'\s+Parish$', r'\s+Borough$', 
        r'\s+city$', r'\s+City$', r'\s+COUNTY$', 
        r'\s+PARISH$', r'\s+BOROUGH$'
    ]
    
    for suffix in suffixes_to_remove:
        county_clean = county_clean.str.replace(suffix, '', regex=True)
    
    # Additional cleaning
    county_clean = county_clean.str.strip()
    county_clean = county_clean.str.title()  # Standardize capitalization
    
    return county_clean

def load_and_prepare_hospital_general(file_path: str) -> pd.DataFrame:
    """
    Load and prepare Hospital General Information dataset
    """
    print("Loading Hospital General Information dataset...")
    
    # Load the dataset
    hospital_general = pd.read_csv(file_path)
    
    # Select relevant columns for analysis
    columns_to_keep = [
        'Facility ID',
        'Facility Name', 
        'State',
        'County Name',
        'Hospital Type',
        'Hospital Ownership',
        'Emergency Services',
        'Hospital overall rating'  # Keep even though 47.8% missing - may be useful
    ]
    
    # Keep only existing columns
    existing_columns = [col for col in columns_to_keep if col in hospital_general.columns]
    hospital_general = hospital_general[existing_columns].copy()
    
    # Clean and standardize
    hospital_general['County_Clean'] = standardize_county_names(hospital_general['County Name'])
    hospital_general['State_County'] = hospital_general['State'].astype(str) + '_' + hospital_general['County_Clean']
    
    # Clean Facility ID to ensure 6-character format
    hospital_general['Facility ID'] = hospital_general['Facility ID'].astype(str).str.strip()
    
    # Remove duplicates based on Facility ID
    hospital_general = hospital_general.drop_duplicates(subset=['Facility ID'])
    
    print(f"Hospital General dataset prepared: {len(hospital_general)} hospitals")
    print(f"Geographic coverage: {hospital_general['State'].nunique()} states, {hospital_general['County_Clean'].nunique()} counties")
    
    return hospital_general

def load_and_prepare_county_health(file_path: str) -> pd.DataFrame:
    """
    Load and prepare County Health Rankings dataset
    """
    print("Loading County Health Rankings dataset...")
    
    # Load the dataset
    county_health = pd.read_csv(file_path)
    
    # Filter to 2020-2022 to match readmissions period
    county_health = county_health[county_health['Year'].isin([2020, 2021, 2022])].copy()
    
    # Select core social determinant variables with low missing data
    core_variables = [
        'Year',
        'State',
        'County',
        'FIPS',
        # Economic factors
        'Median household income',
        'Children in poverty',
        'Unemployment',
        # Healthcare access
        'Uninsured adults',
        'Primary care physicians',
        'Mental health providers',
        # Demographics
        '% 65 and older',
        '% Rural',
        '% White',
        '% Black',
        '% Hispanic',
        '% Asian',
        # Education
        'Some college'
    ]
    
    # Keep only existing columns
    existing_columns = [col for col in core_variables if col in county_health.columns]
    county_health = county_health[existing_columns].copy()
    
    # Clean and standardize county names
    county_health['County_Clean'] = standardize_county_names(county_health['County'])
    county_health['State_County'] = county_health['State'].astype(str) + '_' + county_health['County_Clean']
    
    # For multiple years, take the most recent available data (2022 first, then 2021, then 2020)
    # This handles the decreasing completeness over time
    county_health = county_health.sort_values(['State_County', 'Year'], ascending=[True, False])
    county_health = county_health.groupby('State_County').first().reset_index()
    
    print(f"County Health dataset prepared: {len(county_health)} counties")
    print(f"Geographic coverage: {county_health['State'].nunique()} states")
    
    return county_health

def load_and_prepare_readmissions(file_path: str) -> pd.DataFrame:
    """
    Load and prepare Hospital Readmissions dataset
    """
    print("Loading Hospital Readmissions dataset...")
    
    # Load the dataset
    readmissions = pd.read_csv(file_path)
    
    # Filter to pneumonia readmissions (primary outcome)
    readmissions = readmissions[readmissions['Measure ID'] == 'READM-30-PN-HRRP'].copy()
    
    # Select relevant columns
    columns_to_keep = [
        'Facility ID',
        'Facility Name',
        'Measure ID',
        'Number of Discharges',
        'Footnote',
        'Excess Readmission Ratio',
        'Predicted Readmission Rate',
        'Expected Readmission Rate',
        'Number of Readmissions'
    ]
    
    # Keep only existing columns
    existing_columns = [col for col in columns_to_keep if col in readmissions.columns]
    readmissions = readmissions[existing_columns].copy()
    
    # Clean Facility ID
    readmissions['Facility ID'] = readmissions['Facility ID'].astype(str).str.strip()
    
    # Remove records with missing outcome data
    readmissions = readmissions.dropna(subset=['Excess Readmission Ratio'])
    
    # Remove duplicates
    readmissions = readmissions.drop_duplicates(subset=['Facility ID'])
    
    print(f"Readmissions dataset prepared: {len(readmissions)} hospitals with pneumonia data")
    
    return readmissions

def execute_three_way_merge(hospital_general: pd.DataFrame, 
                           county_health: pd.DataFrame, 
                           readmissions: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute the three-way merge and return merged dataset with statistics
    """
    print("\n=== EXECUTING THREE-WAY MERGE ===")
    
    # Step 1: Merge Hospital General with County Health Rankings
    print("Step 1: Merging Hospital General with County Health Rankings...")
    
    merge_stats = {}
    
    # First merge on State_County
    hospital_county_merge = pd.merge(
        hospital_general, 
        county_health, 
        on='State_County', 
        how='left',
        suffixes=('_hospital', '_county')
    )
    
    # Check merge success
    successful_county_merge = hospital_county_merge['Year'].notna().sum()
    merge_stats['hospital_county_successful'] = successful_county_merge
    merge_stats['hospital_county_total'] = len(hospital_general)
    merge_stats['hospital_county_rate'] = successful_county_merge / len(hospital_general) * 100
    
    print(f"Hospital-County merge: {successful_county_merge}/{len(hospital_general)} hospitals matched ({merge_stats['hospital_county_rate']:.1f}%)")
    
    # Step 2: Merge with Readmissions data
    print("Step 2: Merging with Readmissions data...")
    
    final_merged = pd.merge(
        hospital_county_merge,
        readmissions,
        on='Facility ID',
        how='inner',  # Only keep hospitals with readmissions data
        suffixes=('', '_readmissions')
    )
    
    # Check final merge success
    merge_stats['final_sample_size'] = len(final_merged)
    merge_stats['readmissions_total'] = len(readmissions)
    merge_stats['three_way_success_rate'] = len(final_merged) / len(readmissions) * 100
    
    print(f"Final three-way merge: {len(final_merged)} hospitals with complete data")
    print(f"Success rate from readmissions sample: {merge_stats['three_way_success_rate']:.1f}%")
    
    # Step 3: Clean final dataset
    print("Step 3: Cleaning final dataset...")
    
    # Remove duplicate columns and clean names
    final_merged = final_merged.loc[:, ~final_merged.columns.duplicated()]
    
    # Create final analysis variables
    final_merged['Has_Emergency_Services'] = final_merged['Emergency Services'].map({'Yes': 1, 'No': 0})
    
    # Convert numeric columns
    numeric_columns = [
        'Median household income', 'Children in poverty', 'Unemployment',
        'Uninsured adults', 'Primary care physicians', 'Mental health providers',
        '% 65 and older', '% Rural', '% White', '% Black', '% Hispanic', '% Asian',
        'Some college', 'Excess Readmission Ratio'
    ]
    
    for col in numeric_columns:
        if col in final_merged.columns:
            final_merged[col] = pd.to_numeric(final_merged[col], errors='coerce')
    
    # Calculate data completeness for key variables
    merge_stats['key_variable_completeness'] = {}
    for col in numeric_columns:
        if col in final_merged.columns:
            completeness = (1 - final_merged[col].isna().sum() / len(final_merged)) * 100
            merge_stats['key_variable_completeness'][col] = completeness
    
    print(f"Final dataset ready: {len(final_merged)} hospitals")
    
    return final_merged, merge_stats

def analyze_merge_results(merged_data: pd.DataFrame, merge_stats: Dict) -> None:
    """
    Analyze and display merge results
    """
    print("\n=== MERGE ANALYSIS RESULTS ===")
    
    print(f"\nFinal Sample Size: {merge_stats['final_sample_size']} hospitals")
    print(f"Geographic Coverage: {merged_data['State_hospital'].nunique()} states, {merged_data['County_Clean'].nunique()} counties")
    
    print(f"\nMerge Success Rates:")
    print(f"  Hospital-County merge: {merge_stats['hospital_county_rate']:.1f}%")
    print(f"  Three-way merge success: {merge_stats['three_way_success_rate']:.1f}%")
    
    print(f"\nHospital Characteristics Distribution:")
    print(f"  Hospital Type:\n{merged_data['Hospital Type'].value_counts()}")
    print(f"  Hospital Ownership:\n{merged_data['Hospital Ownership'].value_counts()}")
    print(f"  Emergency Services: {merged_data['Has_Emergency_Services'].mean()*100:.1f}% have emergency services")
    
    print(f"\nKey Variable Completeness:")
    for var, completeness in merge_stats['key_variable_completeness'].items():
        print(f"  {var}: {completeness:.1f}%")
    
    print(f"\nPrimary Outcome (Excess Readmission Ratio):")
    err_stats = merged_data['Excess Readmission Ratio'].describe()
    print(f"  Mean: {err_stats['mean']:.3f}")
    print(f"  Std: {err_stats['std']:.3f}")
    print(f"  Range: {err_stats['min']:.3f} - {err_stats['max']:.3f}")

# Main execution function
def main():
    """
    Main function to execute the three-way merge
    """
    print("=== SOCIAL DETERMINANTS ANALYSIS: THREE-WAY DATA MERGE ===\n")
    
    # File paths - UPDATE THESE WITH YOUR ACTUAL FILE PATHS
    hospital_general_path = "hospital_general_information.csv"  # Update this path
    county_health_path = "County Health Rankings Analytical Data.csv"  # Update this path
    readmissions_path = "FY_2025_HOSPITAL_READMISSIONS_REDUCTION_PROGRAM_HOSPITAL.CSV"  # Update this path
    
    try:
        # Load and prepare datasets
        hospital_general = load_and_prepare_hospital_general(hospital_general_path)
        county_health = load_and_prepare_county_health(county_health_path)
        readmissions = load_and_prepare_readmissions(readmissions_path)
        
        # Execute three-way merge
        merged_data, merge_stats = execute_three_way_merge(hospital_general, county_health, readmissions)
        
        # Analyze results
        analyze_merge_results(merged_data, merge_stats)
        
        # Save merged dataset
        output_path = "merged_social_determinants_analysis.csv"
        merged_data.to_csv(output_path, index=False)
        print(f"\nMerged dataset saved to: {output_path}")
        
        # Return for further analysis
        return merged_data, merge_stats
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file. Please update the file paths in the main() function.")
        print(f"Missing file: {e}")
        return None, None
    except Exception as e:
        print(f"Error during merge process: {e}")
        return None, None

if __name__ == "__main__":
    merged_data, merge_stats = main()
