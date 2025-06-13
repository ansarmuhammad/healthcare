import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Load existing data
patients = pd.read_csv('D:/____________healthcare/old but it works for late charges/output_data/patients.csv')
visits = pd.read_csv('D:/____________healthcare/old but it works for late charges/output_data/patient_visits.csv')
charges = pd.read_csv('D:/____________healthcare/old but it works for late charges/output_data/charges.csv')

# 1. Add service categories and make one service dominate
service_categories = {
    'S001': 'Lab', 'S002': 'Lab', 'S003': 'Lab', 'S004': 'Lab', 'S005': 'Lab',
    'S006': 'Lab', 'S007': 'Lab', 'S008': 'Lab', 'S009': 'Lab', 'S010': 'Lab',
    'S011': 'Pharmacy', 'S012': 'Pharmacy', 'S013': 'Pharmacy', 'S014': 'Pharmacy',
    'S015': 'Radiology', 'S016': 'Radiology', 'S017': 'Radiology', 'S018': 'Radiology',
    'S019': 'Radiology', 'S020': 'Radiology'
}

# Make S005 (Lab) the dominant service for 25% of charges
dominant_service = 'S005'
num_charges = len(charges)
num_dominant = int(num_charges * 0.25)

# Randomly select charges to change to the dominant service
dominant_indices = np.random.choice(charges.index, size=num_dominant, replace=False)
charges.loc[dominant_indices, 'service_code'] = dominant_service

# Add service category to charges
charges['service_category'] = charges['service_code'].map(service_categories)

# 2. Add patient ailments including cancer
ailments = [
    'Hypertension', 'Diabetes', 'Asthma', 'Arthritis', 'Heart Disease',
    'Cancer', 'COPD', 'Osteoporosis', 'Alzheimers', 'Kidney Disease'
]

# Add ailments to patients with higher probability for older patients
def assign_ailment(row):
    age = (pd.Timestamp.now() - pd.to_datetime(row['date_of_birth'])).days // 365
    ailments_list = []
    
    # Base ailments based on age
    if age > 70:
        num_ailments = np.random.choice([2, 3], p=[0.6, 0.4])
        ailments_list = np.random.choice(ailments, num_ailments, replace=False).tolist()
    else:
        if np.random.random() < 0.3:  # 30% chance of having at least one ailment
            num_ailments = np.random.choice([1, 2], p=[0.7, 0.3])
            ailments_list = np.random.choice(ailments, num_ailments, replace=False).tolist()
    
    # Ensure Cancer is more likely for older patients
    if age > 60 and 'Cancer' not in ailments_list and np.random.random() < 0.4:
        if ailments_list:
            ailments_list[-1] = 'Cancer'  # Replace last ailment with Cancer
        else:
            ailments_list = ['Cancer']
    
    return ', '.join(ailments_list) if ailments_list else 'None'

patients['ailments'] = patients.apply(assign_ailment, axis=1)

# 3. Add bad debt indicators
def calculate_bad_debt(row):
    visit = visits[visits['visit_id'] == row['visit_id']].iloc[0]
    patient = patients[patients['patient_id'] == visit['patient_id']].iloc[0]
    
    reasons = []
    
    # Make 'late_charges' the most common reason (40% probability)
    if np.random.random() < 0.4:
        return 'late_charges'
    
    # Calculate length of stay
    start_date = pd.to_datetime(visit['start_date_of_inpatient_visit'])
    end_date = pd.to_datetime(visit['discharge_date'])
    los = (end_date - start_date).days
    
    # Check for bad debt conditions (with reduced probabilities)
    if los > 10 and np.random.random() < 0.3:  # 30% of long stays
        reasons.append('long_stay')
    
    # Calculate age
    dob = pd.to_datetime(patient['date_of_birth'])
    age = (pd.Timestamp.now() - dob).days // 365
    
    if age > 70 and np.random.random() < 0.3:  # 30% of elderly
        reasons.append('elderly')
    
    # Check for cancer
    if 'Cancer' in patient['ailments'] and np.random.random() < 0.4:  # 40% of cancer patients
        reasons.append('cancer_patient')
    
    # Check for high-cost services (with reduced probability)
    if row['service_code'] in ['S005', 'S015', 'S020'] and np.random.random() < 0.3:  # 30% of high-cost
        reasons.append('high_cost_service')
    
    # Add random chance for other reasons
    if np.random.random() < 0.05:  # Reduced from 10% to 5%
        other_reasons = ['insurance_denial', 'self_pay', 'billing_error']
        if not reasons:  # Only add if no other reasons
            reasons.append(np.random.choice(other_reasons))
    
    # If no reasons were added, default to 'late_charges' (this ensures it's the most common)
    if not reasons:
        return 'late_charges'
        
    return ', '.join(reasons)

# Apply bad debt calculation
charges['bad_debt_reason'] = charges.apply(calculate_bad_debt, axis=1)

# 4. Save enhanced data
patients.to_csv('D:/____________healthcare/old but it works for late charges/output_data/enhanced_patients.csv', index=False)
visits.to_csv('D:/____________healthcare/old but it works for late charges/output_data/enhanced_visits.csv', index=False)
charges.to_csv('D:/____________healthcare/old but it works for late charges/output_data/enhanced_charges.csv', index=False)

print("Enhanced dataset created successfully!")