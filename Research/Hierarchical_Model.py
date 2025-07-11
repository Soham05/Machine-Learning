#!/usr/bin/env python3
"""
Step 3A: Hierarchical Model Implementation
Social Determinants vs. Hospital Characteristics Analysis

This script implements the hierarchical modeling phase to test the hypothesis
that county-level social determinants explain >60% of variance in hospital
readmission performance outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import mixedlm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("="*80)
print("STEP 3A: HIERARCHICAL MODEL IMPLEMENTATION")
print("Social Determinants vs. Hospital Characteristics Analysis")
print("="*80)

# Load the final analytical dataset
print("\n1. LOADING FINAL ANALYTICAL DATASET")
print("-" * 50)

try:
    df = pd.read_csv('hierarchical_model_final_dataset.csv')
    print(f"✅ Dataset loaded successfully")
    print(f"   - Shape: {df.shape[0]} hospitals, {df.shape[1]} variables")
    print(f"   - Counties: {df['county_group'].nunique()}")
    print(f"   - Hospitals per county (avg): {df.groupby('county_group').size().mean():.1f}")
    
    # Display basic info about the outcome variable
    outcome_col = 'Excess Readmission Ratio'  # Adjust if column name differs
    if outcome_col in df.columns:
        print(f"\n   Primary Outcome: {outcome_col}")
        print(f"   - Mean: {df[outcome_col].mean():.3f}")
        print(f"   - Std: {df[outcome_col].std():.3f}")
        print(f"   - Range: {df[outcome_col].min():.3f} to {df[outcome_col].max():.3f}")
    else:
        print(f"\n   ⚠️  Need to identify outcome variable column name")
        print(f"   Available columns: {list(df.columns)}")
        
except FileNotFoundError:
    print("❌ File 'hierarchical_model_final_dataset.csv' not found")
    print("   Please ensure the final dataset is in the working directory")
    exit()

print("\n2. DESCRIPTIVE STATISTICS")
print("-" * 50)

# County-level clustering summary
county_summary = df.groupby('county_group').agg({
    outcome_col: ['count', 'mean', 'std'],
    'Ownership_Category_Clean': lambda x: x.mode()[0] if not x.empty else 'Unknown'
}).round(3)

print("County-level clustering summary:")
print(f"- Total counties: {len(county_summary)}")
print(f"- Hospitals per county: {county_summary[(outcome_col, 'count')].describe()}")
print(f"- Within-county outcome variation: {county_summary[(outcome_col, 'std')].mean():.3f}")

# Check for counties with single hospitals (problematic for hierarchical modeling)
single_hospital_counties = county_summary[county_summary[(outcome_col, 'count')] == 1]
print(f"- Counties with single hospitals: {len(single_hospital_counties)} ({len(single_hospital_counties)/len(county_summary)*100:.1f}%)")

print("\n3. MODEL 1: NULL MODEL (Baseline)")
print("-" * 50)

# Fit null model - only random intercept for counties
print("Fitting null model: ERR ~ 1 + (1|county_group)")

try:
    null_model = mixedlm(f"{outcome_col} ~ 1", 
                        data=df, 
                        groups=df['county_group'])
    null_results = null_model.fit()
    
    print("✅ Null model fitted successfully")
    print(f"   - Log-likelihood: {null_results.llf:.2f}")
    print(f"   - AIC: {null_results.aic:.2f}")
    print(f"   - BIC: {null_results.bic:.2f}")
    
    # Extract variance components
    sigma2_u = null_results.cov_re.iloc[0, 0]  # County-level variance
    sigma2_e = null_results.scale  # Hospital-level (residual) variance
    
    print(f"\n   Variance Components:")
    print(f"   - County-level variance (σ²_u): {sigma2_u:.6f}")
    print(f"   - Hospital-level variance (σ²_e): {sigma2_e:.6f}")
    print(f"   - Total variance: {sigma2_u + sigma2_e:.6f}")
    
    # Calculate Intraclass Correlation Coefficient (ICC)
    icc = sigma2_u / (sigma2_u + sigma2_e)
    print(f"\n   📊 INTRACLASS CORRELATION COEFFICIENT (ICC): {icc:.4f}")
    print(f"   - This means {icc*100:.1f}% of variance is between counties")
    print(f"   - And {(1-icc)*100:.1f}% of variance is within counties (between hospitals)")
    
    # ICC interpretation
    if icc < 0.05:
        icc_interpretation = "Very low clustering - hierarchical modeling may not be needed"
    elif icc < 0.10:
        icc_interpretation = "Low clustering - some county-level effects"
    elif icc < 0.25:
        icc_interpretation = "Moderate clustering - hierarchical modeling justified"
    else:
        icc_interpretation = "High clustering - strong county-level effects"
    
    print(f"   - Interpretation: {icc_interpretation}")
    
except Exception as e:
    print(f"❌ Error fitting null model: {str(e)}")
    print("   Check outcome variable name and data structure")

print("\n4. MODEL 2: LEVEL 1 MODEL (Hospital Controls)")
print("-" * 50)

# Fit model with hospital-level predictors
print("Fitting Level 1 model: ERR ~ hospital_vars + (1|county_group)")

# Define Level 1 variables
level1_vars = [
    'Ownership_Category_Clean',
    'Hospital_Rating_Numeric_Imputed_std',
    'Rating_Missing'
]

# Check if all Level 1 variables exist
missing_vars = [var for var in level1_vars if var not in df.columns]
if missing_vars:
    print(f"⚠️  Missing Level 1 variables: {missing_vars}")
    print("   Available columns:", [col for col in df.columns if any(x in col.lower() for x in ['ownership', 'rating', 'missing'])])
else:
    try:
        # Create formula for Level 1 model
        level1_formula = f"{outcome_col} ~ " + " + ".join(level1_vars)
        
        level1_model = mixedlm(level1_formula, 
                              data=df, 
                              groups=df['county_group'])
        level1_results = level1_model.fit()
        
        print("✅ Level 1 model fitted successfully")
        print(f"   - Log-likelihood: {level1_results.llf:.2f}")
        print(f"   - AIC: {level1_results.aic:.2f}")
        print(f"   - BIC: {level1_results.bic:.2f}")
        
        # Extract variance components
        sigma2_u_l1 = level1_results.cov_re.iloc[0, 0]  # County-level variance
        sigma2_e_l1 = level1_results.scale  # Hospital-level variance
        
        print(f"\n   Variance Components (after adding hospital controls):")
        print(f"   - County-level variance (σ²_u): {sigma2_u_l1:.6f}")
        print(f"   - Hospital-level variance (σ²_e): {sigma2_e_l1:.6f}")
        print(f"   - Total variance: {sigma2_u_l1 + sigma2_e_l1:.6f}")
        
        # Calculate variance explained by hospital characteristics
        total_var_null = sigma2_u + sigma2_e
        total_var_l1 = sigma2_u_l1 + sigma2_e_l1
        var_explained_hospital = (total_var_null - total_var_l1) / total_var_null
        
        print(f"\n   📊 VARIANCE EXPLAINED BY HOSPITAL CHARACTERISTICS:")
        print(f"   - Variance reduction: {var_explained_hospital*100:.1f}%")
        print(f"   - Remaining county variance: {sigma2_u_l1/total_var_l1*100:.1f}%")
        
        # Display coefficients
        print(f"\n   Fixed Effects (Hospital Characteristics):")
        print(level1_results.summary().tables[1])
        
    except Exception as e:
        print(f"❌ Error fitting Level 1 model: {str(e)}")

print("\n5. MODEL 3: FULL MODEL (Social Determinants + Hospital Controls)")
print("-" * 50)

# Define Level 2 variables (county social determinants)
level2_vars = [
    'median_household_income_raw_value_std',
    'children_in_poverty_raw_value_std', 
    'uninsured_adults_raw_value_std',
    'ratio_of_population_to_primary_care_physicians_std',
    '%_rural_raw_value_std',
    '%_non_hispanic_white_raw_value_std',
    'some_college_raw_value_std'
]

# Check if all Level 2 variables exist
missing_l2_vars = [var for var in level2_vars if var not in df.columns]
if missing_l2_vars:
    print(f"⚠️  Missing Level 2 variables: {missing_l2_vars}")
    print("   Available columns:", [col for col in df.columns if 'std' in col])
else:
    try:
        # Create formula for full model
        all_vars = level1_vars + level2_vars
        full_formula = f"{outcome_col} ~ " + " + ".join(all_vars)
        
        print("Fitting full model: ERR ~ hospital_vars + county_vars + (1|county_group)")
        
        full_model = mixedlm(full_formula, 
                            data=df, 
                            groups=df['county_group'])
        full_results = full_model.fit()
        
        print("✅ Full model fitted successfully")
        print(f"   - Log-likelihood: {full_results.llf:.2f}")
        print(f"   - AIC: {full_results.aic:.2f}")
        print(f"   - BIC: {full_results.bic:.2f}")
        
        # Extract variance components
        sigma2_u_full = full_results.cov_re.iloc[0, 0]  # County-level variance
        sigma2_e_full = full_results.scale  # Hospital-level variance
        
        print(f"\n   Variance Components (full model):")
        print(f"   - County-level variance (σ²_u): {sigma2_u_full:.6f}")
        print(f"   - Hospital-level variance (σ²_e): {sigma2_e_full:.6f}")
        print(f"   - Total variance: {sigma2_u_full + sigma2_e_full:.6f}")
        
    except Exception as e:
        print(f"❌ Error fitting full model: {str(e)}")
        print("   This might be due to multicollinearity or model complexity")

print("\n6. VARIANCE DECOMPOSITION ANALYSIS")
print("-" * 50)

try:
    # Calculate variance explained by each component
    total_var_null = sigma2_u + sigma2_e
    total_var_l1 = sigma2_u_l1 + sigma2_e_l1  
    total_var_full = sigma2_u_full + sigma2_e_full
    
    # Variance explained by hospital characteristics
    var_explained_hospital = (total_var_null - total_var_l1) / total_var_null
    
    # Variance explained by social determinants (county factors)
    var_explained_social = (total_var_l1 - total_var_full) / total_var_null
    
    # Remaining variance
    var_remaining = total_var_full / total_var_null
    
    print("📊 VARIANCE DECOMPOSITION RESULTS:")
    print(f"   - Hospital Characteristics: {var_explained_hospital*100:.1f}%")
    print(f"   - Social Determinants: {var_explained_social*100:.1f}%")
    print(f"   - Remaining Unexplained: {var_remaining*100:.1f}%")
    
    # Test primary hypothesis
    print(f"\n🎯 PRIMARY HYPOTHESIS TEST:")
    print(f"   - Hypothesis: County social determinants explain >60% of variance")
    print(f"   - Actual: Social determinants explain {var_explained_social*100:.1f}%")
    
    if var_explained_social > 0.60:
        print(f"   - Result: ✅ HYPOTHESIS SUPPORTED")
    else:
        print(f"   - Result: ❌ HYPOTHESIS NOT SUPPORTED")
        
    # Additional insights
    combined_predictors = var_explained_hospital + var_explained_social
    print(f"\n   📈 ADDITIONAL INSIGHTS:")
    print(f"   - Combined predictors explain: {combined_predictors*100:.1f}%")
    print(f"   - Social determinants vs Hospital characteristics ratio: {var_explained_social/var_explained_hospital:.1f}:1")
    
    # Create variance decomposition visualization
    plt.figure(figsize=(10, 6))
    
    # Pie chart
    plt.subplot(1, 2, 1)
    labels = ['Hospital\nCharacteristics', 'Social\nDeterminants', 'Unexplained']
    sizes = [var_explained_hospital*100, var_explained_social*100, var_remaining*100]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Variance Decomposition\n(Pneumonia Readmission Outcomes)')
    
    # Bar chart
    plt.subplot(1, 2, 2)
    components = ['Hospital\nCharacteristics', 'Social\nDeterminants', 'Unexplained']
    values = [var_explained_hospital*100, var_explained_social*100, var_remaining*100]
    
    bars = plt.bar(components, values, color=colors, alpha=0.7)
    plt.ylabel('Percent of Variance Explained')
    plt.title('Variance Components')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('variance_decomposition_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
except Exception as e:
    print(f"❌ Error in variance decomposition: {str(e)}")

print("\n7. MODEL DIAGNOSTICS")
print("-" * 50)

try:
    # Check multicollinearity for full model
    print("Checking multicollinearity (VIF scores):")
    
    # Prepare data for VIF calculation (only numeric variables)
    numeric_vars = []
    for var in all_vars:
        if var in df.columns:
            if df[var].dtype in ['int64', 'float64']:
                numeric_vars.append(var)
    
    if len(numeric_vars) > 1:
        # Calculate VIF for numeric variables
        vif_data = df[numeric_vars].dropna()
        if len(vif_data) > 0:
            vif_df = pd.DataFrame()
            vif_df["Variable"] = vif_data.columns
            vif_df["VIF"] = [variance_inflation_factor(vif_data.values, i) 
                           for i in range(len(vif_data.columns))]
            vif_df = vif_df.sort_values('VIF', ascending=False)
            
            print(vif_df)
            
            # Flag high VIF values
            high_vif = vif_df[vif_df['VIF'] > 10]
            if len(high_vif) > 0:
                print(f"\n⚠️  High multicollinearity detected (VIF > 10):")
                print(high_vif)
            else:
                print("\n✅ No concerning multicollinearity detected")
    
    # Model fit statistics comparison
    print(f"\n📊 MODEL FIT COMPARISON:")
    print(f"{'Model':<20} {'AIC':<10} {'BIC':<10} {'Log-Likelihood':<15}")
    print("-" * 55)
    print(f"{'Null Model':<20} {null_results.aic:<10.2f} {null_results.bic:<10.2f} {null_results.llf:<15.2f}")
    print(f"{'Level 1 Model':<20} {level1_results.aic:<10.2f} {level1_results.bic:<10.2f} {level1_results.llf:<15.2f}")
    print(f"{'Full Model':<20} {full_results.aic:<10.2f} {full_results.bic:<10.2f} {full_results.llf:<15.2f}")
    
    # Model improvement tests
    print(f"\n📈 MODEL IMPROVEMENT:")
    print(f"   - Level 1 vs Null: ΔAIC = {level1_results.aic - null_results.aic:.2f}")
    print(f"   - Full vs Level 1: ΔAIC = {full_results.aic - level1_results.aic:.2f}")
    print(f"   - Full vs Null: ΔAIC = {full_results.aic - null_results.aic:.2f}")
    
except Exception as e:
    print(f"❌ Error in model diagnostics: {str(e)}")

print("\n" + "="*80)
print("STEP 3A COMPLETED - HIERARCHICAL MODELING RESULTS SUMMARY")
print("="*80)

print(f"""
🎯 RESEARCH QUESTION ANSWERED:
   "What percentage of variance in hospital readmission performance 
   is explained by county social determinants vs. hospital characteristics?"

📊 KEY FINDINGS:
   • County social determinants explain: {var_explained_social*100:.1f}% of variance
   • Hospital characteristics explain: {var_explained_hospital*100:.1f}% of variance
   • Combined model explains: {(var_explained_social + var_explained_hospital)*100:.1f}% of variance
   
🔬 HYPOTHESIS TEST RESULT:
   • Hypothesis: County social determinants explain >60% of variance
   • Result: {'SUPPORTED' if var_explained_social > 0.60 else 'NOT SUPPORTED'}

📈 NEXT STEPS:
   1. Review coefficient significance in full model
   2. Conduct sensitivity analyses
   3. Prepare manuscript for JAMA Network Open
   4. Consider additional model specifications
""")

print("\n✅ Hierarchical modeling analysis complete!")
print("   Results saved: variance_decomposition_analysis.png")
print("   Ready for manuscript preparation phase.")
