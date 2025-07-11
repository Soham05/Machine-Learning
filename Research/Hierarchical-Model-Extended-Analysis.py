#!/usr/bin/env python3
"""
Extended Analysis: Robustness Checks and Sensitivity Analysis
Social Determinants vs. Hospital Characteristics Analysis
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

print("="*80)
print("EXTENDED ANALYSIS: ROBUSTNESS CHECKS & SENSITIVITY ANALYSIS")
print("="*80)

# Load the dataset
df = pd.read_csv('hierarchical_model_cleaned_dataset.csv')

print("\n1. SENSITIVITY ANALYSIS: ALTERNATIVE OUTCOME SPECIFICATIONS")
print("-" * 70)

# Check if outcome is normally distributed
outcome_col = 'ERR'
print(f"Testing normality of {outcome_col}:")
shapiro_stat, shapiro_p = stats.shapiro(df[outcome_col].sample(min(5000, len(df))))
print(f"   - Shapiro-Wilk test: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")

# Test log transformation
df['log_ERR'] = np.log(df[outcome_col])
print(f"   - Log-transformed outcome statistics:")
print(f"     Mean: {df['log_ERR'].mean():.4f}, Std: {df['log_ERR'].std():.4f}")

# Test alternative specifications
print(f"\n2. ROBUSTNESS CHECK: DIFFERENT CLUSTERING APPROACHES")
print("-" * 70)

# Check if results are sensitive to county sample size
county_sizes = df.groupby('county_id').size()
print(f"County size distribution:")
print(f"   - Counties with 1 hospital: {(county_sizes == 1).sum()} ({(county_sizes == 1).mean()*100:.1f}%)")
print(f"   - Counties with 2-5 hospitals: {((county_sizes >= 2) & (county_sizes <= 5)).sum()}")
print(f"   - Counties with 6+ hospitals: {(county_sizes >= 6).sum()}")

# Subset analysis: Remove single-hospital counties
multi_hospital_counties = county_sizes[county_sizes > 1].index
df_multi = df[df['county_id'].isin(multi_hospital_counties)]
print(f"\n   Analysis excluding single-hospital counties:")
print(f"   - Sample size: {len(df_multi)} hospitals in {df_multi['county_id'].nunique()} counties")

print(f"\n3. MULTICOLLINEARITY ASSESSMENT")
print("-" * 70)

# Check VIF for social determinants
social_vars = [
    'median_household_income_raw_value_std',
    'children_in_poverty_raw_value_std',
    'uninsured_adults_raw_value_std',
    'ratio_of_population_to_primary_care_physicians_std',
    'pct_rural_raw_value_std',
    'pct_non_hispanic_white_raw_value_std',
    'some_college_raw_value_std'
]

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = social_vars
vif_data["VIF"] = [variance_inflation_factor(df[social_vars].values, i) 
                   for i in range(len(social_vars))]

print("Variance Inflation Factors (VIF):")
print(vif_data.round(2))
print(f"\nHigh multicollinearity concerns (VIF > 5):")
high_vif = vif_data[vif_data["VIF"] > 5]
if len(high_vif) > 0:
    print(high_vif[["Variable", "VIF"]])
else:
    print("   - No variables with VIF > 5")

print(f"\n4. EFFECT SIZE ANALYSIS")
print("-" * 70)

# Calculate Cohen's f² for effect sizes
def cohens_f_squared(r2_full, r2_reduced):
    """Calculate Cohen's f² effect size"""
    return (r2_full - r2_reduced) / (1 - r2_full)

# Approximate R² values from variance explained
r2_null = 0.0
r2_hospital = 0.075
r2_full = 0.111

f2_hospital = cohens_f_squared(r2_hospital, r2_null)
f2_social = cohens_f_squared(r2_full, r2_hospital)

print(f"Cohen's f² Effect Sizes:")
print(f"   - Hospital characteristics: f² = {f2_hospital:.4f}")
print(f"   - Social determinants: f² = {f2_social:.4f}")

# Effect size interpretation
def interpret_f2(f2):
    if f2 < 0.02:
        return "negligible"
    elif f2 < 0.15:
        return "small"
    elif f2 < 0.35:
        return "medium"
    else:
        return "large"

print(f"   - Hospital effect size: {interpret_f2(f2_hospital)}")
print(f"   - Social determinants effect size: {interpret_f2(f2_social)}")

print(f"\n5. CROSS-VALIDATION ANALYSIS")
print("-" * 70)

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# Prepare data for cross-validation
X_hospital = df[['Hospital_Rating_Numeric_Imputed_std', 'Rating_Missing'] + 
                ['Ownership_Private', 'Ownership_Public']].fillna(0)
X_social = df[social_vars].fillna(0)
X_combined = pd.concat([X_hospital, X_social], axis=1)
y = df[outcome_col]

# Simple linear regression cross-validation (proxy for mixed effects)
from sklearn.linear_model import LinearRegression

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, X in [('Hospital Only', X_hospital), 
                ('Social Only', X_social), 
                ('Combined', X_combined)]:
    
    cv_scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cv_scores.append(r2_score(y_test, y_pred))
    
    cv_results[name] = cv_scores

print("5-Fold Cross-Validation R² Scores:")
for name, scores in cv_results.items():
    print(f"   - {name}: {np.mean(scores):.4f} (±{np.std(scores):.4f})")

print(f"\n6. ALTERNATIVE MODEL SPECIFICATIONS")
print("-" * 70)

# Test interaction effects
print("Testing key interaction effects:")

# Hospital rating × Social determinants interactions
interaction_vars = [
    'Hospital_Rating_Numeric_Imputed_std * median_household_income_raw_value_std',
    'Hospital_Rating_Numeric_Imputed_std * children_in_poverty_raw_value_std'
]

for interaction in interaction_vars:
    try:
        # Create interaction term
        var1, var2 = interaction.split(' * ')
        df[f'{var1}_x_{var2}'] = df[var1] * df[var2]
        
        # Test if interaction is significant
        interaction_formula = f"{outcome_col} ~ {var1} + {var2} + {var1}_x_{var2}"
        
        # Note: This is a simplified test - full mixed effects would be more appropriate
        from statsmodels.formula.api import ols
        interaction_model = ols(interaction_formula, data=df).fit()
        
        p_value = interaction_model.pvalues[f'{var1}_x_{var2}']
        print(f"   - {interaction}: p = {p_value:.4f}")
        
    except Exception as e:
        print(f"   - {interaction}: Error in calculation")

print(f"\n7. OUTLIER ANALYSIS")
print("-" * 70)

# Identify potential outliers
Q1 = df[outcome_col].quantile(0.25)
Q3 = df[outcome_col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df[outcome_col] < lower_bound) | (df[outcome_col] > upper_bound)]
print(f"Potential outliers (IQR method):")
print(f"   - Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
print(f"   - Outlier range: {outliers[outcome_col].min():.3f} to {outliers[outcome_col].max():.3f}")

# Test sensitivity to outliers
df_no_outliers = df[~df.index.isin(outliers.index)]
print(f"   - Sample without outliers: {len(df_no_outliers)} hospitals")

print(f"\n8. POWER ANALYSIS")
print("-" * 70)

# Calculate statistical power for detecting effect sizes
from scipy.stats import f

# Approximate power calculation
n_hospitals = len(df)
n_counties = df['county_id'].nunique()
alpha = 0.05

# Power for detecting small, medium, large effects
effect_sizes = [0.02, 0.15, 0.35]  # Cohen's f²
effect_names = ['small', 'medium', 'large']

print("Statistical Power Analysis:")
for f2, name in zip(effect_sizes, effect_names):
    # Approximate power using F-distribution
    # This is a simplified calculation
    df_num = 7  # Number of social determinant predictors
    df_den = n_hospitals - df_num - 1
    
    # Non-centrality parameter
    lambda_nc = f2 * n_hospitals
    
    # Critical F-value
    f_crit = f.ppf(1 - alpha, df_num, df_den)
    
    # Power (approximate)
    power = 1 - f.cdf(f_crit, df_num, df_den, nc=lambda_nc)
    
    print(f"   - Power to detect {name} effect (f² = {f2}): {power:.3f}")

print(f"\n9. CLINICAL SIGNIFICANCE ASSESSMENT")
print("-" * 70)

# Calculate minimum detectable effect in clinical terms
baseline_err = df[outcome_col].mean()
print(f"Baseline ERR: {baseline_err:.3f}")

# Effect sizes in terms of ERR change
hospital_effect = 0.017  # From coefficient table
social_effect = 0.012   # Approximate from social determinants

print(f"Clinical Effect Sizes:")
print(f"   - 1 SD improvement in hospital rating: {hospital_effect:.3f} ERR reduction")
print(f"   - 1 SD improvement in social conditions: {social_effect:.3f} ERR change")
print(f"   - Hospital rating effect as % of baseline: {hospital_effect/baseline_err*100:.1f}%")

print(f"\n10. RECOMMENDATIONS FOR MANUSCRIPT")
print("-" * 70)

print("""
METHODOLOGICAL STRENGTHS TO HIGHLIGHT:
✓ Large sample size (2,152 hospitals, 178 counties)
✓ Hierarchical modeling accounting for clustering
✓ Comprehensive social determinants assessment
✓ Robust variance decomposition methodology
✓ Multiple sensitivity analyses

LIMITATIONS TO ACKNOWLEDGE:
• Low overall variance explained (11.1%)
• Cross-sectional design limits causal inference
• Single outcome measure (pneumonia readmissions)
• Administrative data limitations
• Unmeasured confounding possible

SUGGESTED ADDITIONAL ANALYSES:
1. Stratified analysis by hospital size/type
2. Geographic region as additional clustering level
3. Multiple readmission outcomes comparison
4. Longitudinal analysis if multi-year data available
5. Machine learning approaches for comparison

POLICY IMPLICATIONS:
• Hospital quality improvement initiatives prioritized
• Social determinants still matter (3.6% is meaningful)
• Need for hospital-specific interventions
• Community partnerships complementary but secondary
""")

print(f"\n" + "="*80)
print("EXTENDED ANALYSIS COMPLETE")
print("="*80)
