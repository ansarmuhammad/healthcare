import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime

def modify_bad_debt_reasons(charges):
    """Modify the bad debt reasons to make 'late_charges' the most common reason."""
    # Create a copy to avoid modifying the original data
    modified_charges = charges.copy()
    
    # Get the current distribution
    reason_counts = modified_charges['bad_debt_reason'].value_counts(dropna=False)
    
    # If there are no bad debt reasons, we can't modify anything
    if len(reason_counts) <= 1 and pd.isna(reason_counts.index[0]):
        print("No bad debt reasons found to modify.")
        return modified_charges
    
    # Find the current top reason and its count
    if not pd.isna(reason_counts.index[0]):
        current_top_reason = reason_counts.index[0]
        current_max = reason_counts.iloc[0]
    else:
        current_top_reason = None
        current_max = 0
    
    print(f"Current top reason: {current_top_reason} ({current_max} records)")
    
    # Set target count to be higher than the current max to ensure 'late_charges' is top
    target_count = int(current_max * 2)  # Double the current max to ensure it's the top reason
    print(f"Target 'late_charges' count: {target_count} (doubled from current max)")
    
    # Get all record indices that aren't already 'late_charges'
    not_late_charges = ~modified_charges['bad_debt_reason'].str.contains('late_charges', na=True)
    
    # Calculate how many records we need to modify
    n_to_modify = min(target_count, not_late_charges.sum())
    print(f"Will modify {n_to_modify} records to be 'late_charges'")
    
    # Randomly select records to modify
    to_modify = modified_charges[not_late_charges].sample(n=n_to_modify, random_state=42).index
    
    # Set the selected records to 'late_charges'
    modified_charges.loc[to_modify, 'bad_debt_reason'] = 'late_charges'
    
    # Now handle any remaining null/empty records
    null_mask = modified_charges['bad_debt_reason'].isna() | (modified_charges['bad_debt_reason'] == 'None')
    if null_mask.sum() > 0:
        print(f"Setting {null_mask.sum()} null/empty records to 'late_charges'")
        modified_charges.loc[null_mask, 'bad_debt_reason'] = 'late_charges'
    
    # Verify the results
    new_reason_counts = modified_charges['bad_debt_reason'].value_counts(dropna=False)
    print("\nNew reason distribution:")
    print(new_reason_counts.head(10))
    
    # Double check that 'late_charges' is now the top reason
    if not new_reason_counts.empty and 'late_charges' in new_reason_counts.index:
        if new_reason_counts.idxmax() == 'late_charges':
            print("[SUCCESS] 'late_charges' is now the top reason for bad debt!")
        else:
            print(f"[WARNING] 'late_charges' is not the top reason. Top reason is: {new_reason_counts.idxmax()}")
    
    return modified_charges

def load_data():
    """Load and modify the enhanced dataset."""
    try:
        patients = pd.read_csv('D:/____________healthcare/old but it works for late charges/output_data/enhanced_patients.csv')
        visits = pd.read_csv('D:/____________healthcare/old but it works for late charges/output_data/enhanced_visits.csv')
        charges = pd.read_csv('D:/____________healthcare/old but it works for late charges/output_data/enhanced_charges.csv')
        
        # Modify the charges data to make 'late_charges' the most common reason
        charges = modify_bad_debt_reasons(charges)
        
        return patients, visits, charges
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please make sure to run the enhancement script first.")
        return None, None, None

def preprocess_data(patients, visits, charges):
    """Preprocess and merge the data for analysis."""
    # Convert date columns to datetime
    patients['date_of_birth'] = pd.to_datetime(patients['date_of_birth'])
    visits['start_date_of_inpatient_visit'] = pd.to_datetime(visits['start_date_of_inpatient_visit'])
    visits['discharge_date'] = pd.to_datetime(visits['discharge_date'])
    charges['charge_date'] = pd.to_datetime(charges['charge_date'])
    charges['charge_transfer_date_from_lab'] = pd.to_datetime(charges['charge_transfer_date_from_lab'])
    
    # Calculate age
    current_year = datetime.now().year
    patients['age'] = current_year - pd.to_datetime(patients['date_of_birth']).dt.year
    
    # Calculate length of stay in days
    visits['length_of_stay'] = (visits['discharge_date'] - visits['start_date_of_inpatient_visit']).dt.days
    
    # Merge data
    data = pd.merge(charges, visits, on='visit_id', how='left')
    data = pd.merge(data, patients, on='patient_id', how='left')
    
    # Calculate charge delay in days
    data['charge_delay_days'] = (data['charge_transfer_date_from_lab'] - data['charge_date']).dt.days
    
    return data

def analyze_bad_debt_reasons(data):
    """Analyze and calculate probabilities for bad debt reasons.
    
    This function ensures 'late_charges' is the most common reason but not 100%
    of cases, while maintaining a realistic distribution of other reasons.
    """
    # Create a copy to avoid modifying the original data
    data = data.copy()
    
    # Fill NA values
    data['bad_debt_reason'] = data['bad_debt_reason'].fillna('None')
    
    # Track individual reasons and their counts
    reason_counter = Counter()
    
    # Process each row's reasons
    for reasons_str in data['bad_debt_reason']:
        if pd.isna(reasons_str) or reasons_str == 'None':
            continue
            
        # Split by comma and clean up any whitespace
        reasons = [r.strip() for r in str(reasons_str).split(',')]
        
        # Update counts for each reason
        for reason in reasons:
            if reason != 'None':
                reason_counter[reason] += 1
    
    # Convert to DataFrame for easier manipulation
    reason_counts = pd.Series(reason_counter)
    
    # If no reasons found, return empty series
    if reason_counts.empty:
        return pd.Series(dtype=float)
    
    # Calculate current distribution
    total = reason_counts.sum()
    reason_probs = (reason_counts / total) * 100  # as percentage
    
    # If 'late_charges' exists but isn't the top reason, adjust the distribution
    if 'late_charges' in reason_probs.index and reason_probs.idxmax() != 'late_charges':
        # Get current top reason's probability
        current_top_prob = reason_probs.max()
        late_charges_prob = reason_probs['late_charges']
        
        # Calculate adjustment needed to make 'late_charges' the top reason
        # by reducing other reasons proportionally
        if late_charges_prob < current_top_prob:
            # Calculate scaling factor to make late_charges the top reason
            # but not more than 40% of total
            target_late_charges_prob = min(current_top_prob * 1.1, 40.0)  # Cap at 40%
            
            # Calculate remaining probability for other reasons
            remaining_prob = 100 - target_late_charges_prob
            
            # Calculate scaling factor for other reasons
            other_reasons_prob = 100 - late_charges_prob
            if other_reasons_prob > 0:
                scale_factor = remaining_prob / other_reasons_prob
                
                # Apply scaling to other reasons
                for reason in reason_probs.index:
                    if reason != 'late_charges':
                        reason_probs[reason] *= scale_factor
                
                # Set late_charges to target probability
                reason_probs['late_charges'] = target_late_charges_prob
    
    # Ensure 'late_charges' is first in the series
    if 'late_charges' in reason_probs.index:
        late_charges_prob = reason_probs['late_charges']
        reason_probs = reason_probs.drop('late_charges')
        reason_probs = pd.concat([
            pd.Series({'late_charges': late_charges_prob}),
            reason_probs
        ])
        
        # Ensure no single reason exceeds 35% (except late_charges which we just set)
        max_other_prob = 35.0
        for i in range(1, len(reason_probs)):
            if reason_probs.iloc[i] > max_other_prob:
                # Scale down to max_other_prob
                reason_probs.iloc[i] = max_other_prob
    
    return reason_probs

def analyze_by_demographics(data):
    """Analyze bad debt reasons by demographics."""
    # Age groups
    data['age_group'] = pd.cut(
        data['age'],
        bins=[0, 18, 35, 50, 65, 100],
        labels=['0-18', '19-35', '36-50', '51-65', '65+']
    )
    
    # Gender analysis
    gender_analysis = data[data['bad_debt_reason'] != 'None'].groupby('gender')['bad_debt_reason'].value_counts(normalize=True).unstack().fillna(0) * 100
    
    # Age group analysis
    age_analysis = data[data['bad_debt_reason'] != 'None'].groupby('age_group')['bad_debt_reason'].value_counts(normalize=True).unstack().fillna(0) * 100
    
    return gender_analysis, age_analysis

def analyze_by_services(data):
    """Analyze bad debt by service categories and specific services."""
    # By service category
    service_cat_analysis = data[data['bad_debt_reason'] != 'None'].groupby('service_category')['bad_debt_reason']\
        .value_counts(normalize=True).unstack().fillna(0) * 100
    
    # By specific service
    service_analysis = data[data['bad_debt_reason'] != 'None'].groupby('service_code')['bad_debt_reason']\
        .value_counts(normalize=True).unstack().fillna(0) * 100
    
    return service_cat_analysis, service_analysis

def generate_insights(reason_probs, gender_analysis, age_analysis, service_cat_analysis, service_analysis):
    """Generate actionable insights from the analysis."""
    insights = []
    
    # Top reasons for bad debt
    top_reason = reason_probs.idxmax()
    insights.append(f"Top reason for bad debt: {top_reason} ({reason_probs.max():.1f}% of all charges)")
    
    # Gender insights
    if not gender_analysis.empty:
        gender_insight = gender_analysis.idxmax(axis=1).to_dict()
        insights.append("\nGender-based insights:")
        for gender, reason in gender_insight.items():
            value = gender_analysis.loc[gender, reason]
            insights.append(f"- {gender} patients are most affected by: {reason} ({value:.1f}% of cases)")
    
    # Age group insights
    if not age_analysis.empty:
        age_insight = age_analysis.idxmax(axis=1).to_dict()
        insights.append("\nAge group insights:")
        for age, reason in age_insight.items():
            value = age_analysis.loc[age, reason]
            insights.append(f"- Age group {age} is most affected by: {reason} ({value:.1f}% of cases)")
    
    # Service category insights
    if not service_cat_analysis.empty:
        service_cat_insight = service_cat_analysis.idxmax(axis=1).to_dict()
        insights.append("\nService category insights:")
        for service, reason in service_cat_insight.items():
            value = service_cat_analysis.loc[service, reason]
            insights.append(f"- {service} services are most affected by: {reason} ({value:.1f}% of cases)")
    
    # High-risk services
    if not service_analysis.empty:
        high_risk_services = service_analysis.max(axis=1).sort_values(ascending=False).head(3)
        insights.append("\nTop high-risk services:")
        for service, risk in high_risk_services.items():
            reason = service_analysis.loc[service].idxmax()
            insights.append(f"- Service {service}: {risk:.1f}% risk of bad debt (Main reason: {reason})")
    
    return insights

def main():
    print("Loading and modifying data to make 'late_charges' the top reason...")
    patients, visits, charges = load_data()
    
    if patients is None or visits is None or charges is None:
        return
    
    print("Preprocessing data...")
    data = preprocess_data(patients, visits, charges)
    
    print("Analyzing bad debt reasons...")
    reason_probs = analyze_bad_debt_reasons(data)
    
    print("\nTop Bad Debt Reasons (Probability %):")
    print(reason_probs.head(10).to_string())
    
    # Verify 'late_charges' is the top reason
    if not reason_probs.empty:
        top_reason = reason_probs.index[0]
        if 'late_charges' in top_reason:
            print("\n[SUCCESS] 'late_charges' is now the top reason for bad debt.")
        else:
            print(f"\n[WARNING] 'late_charges' is not the top reason. Top reason is: {top_reason}")
    
    # Generate demographic analysis
    print("\nAnalyzing by demographics...")
    gender_analysis, age_analysis = analyze_by_demographics(data)
    
    # Generate service analysis
    print("\nAnalyzing by services...")
    service_cat_analysis, service_analysis = analyze_by_services(data)
    
    # Generate and display insights
    print("\nGenerating insights...")
    insights = generate_insights(reason_probs, gender_analysis, age_analysis, 
                               service_cat_analysis, service_analysis)
    
    # Print the distribution of bad debt reasons
    print("\n=== Bad Debt Reason Distribution ===")
    print(reason_probs.to_string())
    
    # Print demographic analysis
    print("\n=== Bad Debt Reasons by Gender ===")
    print(gender_analysis.to_string())
    print("\n=== Bad Debt Reasons by Age Group ===")
    print(age_analysis.to_string())
    
    print("\nAnalysis complete. Check the generated plots for visualizations.")
    print("Note: Data has been modified to make 'late_charges' the top reason for bad debt.")

if __name__ == "__main__":
    main()
