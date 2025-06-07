import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_and_prepare_data():
    """
    Loads patient, visit, and charge data, merges them, engineers features,
    and prepares a visit-level DataFrame for risk analysis.

    Returns:
        pandas.DataFrame: A DataFrame where each row represents a unique visit_id
                          with engineered features and a target variable, or None if
                          data files are not found.
    """
    # Look for CSV files in the current directory
    patients_path = "patients.csv"
    visits_path = "patient_visits.csv"
    charges_path = "charges.csv"

    try:
        patients_df = pd.read_csv(patients_path)
        patient_visits_df = pd.read_csv(visits_path)
        charges_df = pd.read_csv(charges_path)
    except FileNotFoundError:
        print("Error: One or more data files not found. Please run generate_data.py first.")
        return None

    # Convert date columns to datetime objects
    patient_visits_df['start_date_of_inpatient_visit'] = pd.to_datetime(patient_visits_df['start_date_of_inpatient_visit'])
    patient_visits_df['discharge_date'] = pd.to_datetime(patient_visits_df['discharge_date'])
    charges_df['charge_date'] = pd.to_datetime(charges_df['charge_date'])
    charges_df['charge_transfer_date_from_lab'] = pd.to_datetime(charges_df['charge_transfer_date_from_lab'])

    # Merge DataFrames
    # Merge charges with patient visits
    merged_df = pd.merge(charges_df, patient_visits_df, on='visit_id', how='left')
    # Merge the result with patients
    merged_df = pd.merge(merged_df, patients_df, on='patient_id', how='left')

    # Engineer features at charge level
    merged_df['is_late_charge'] = merged_df['charge_date'] > merged_df['discharge_date']

    # Aggregate data per visit_id
    visit_level_features = merged_df.groupby('visit_id').agg(
        total_charges_per_visit=('charge_amount', 'sum'),
        num_total_charges_per_visit=('charge_id', 'count'), # Assuming charge_id is unique per charge
        num_late_charges_per_visit=('is_late_charge', 'sum'),
        amount_of_late_charges_per_visit=('charge_amount', lambda x: x[merged_df.loc[x.index, 'is_late_charge']].sum()),
        # For visit-specific dates and room_type, take the first observed value (should be consistent per visit_id)
        start_date_of_inpatient_visit=('start_date_of_inpatient_visit', 'first'),
        discharge_date=('discharge_date', 'first'),
        room_type=('room_type', 'first'),
        patient_id=('patient_id', 'first') # Keep patient_id for potential future use or context
    ).reset_index()

    # Calculate length_of_stay_days
    visit_level_features['length_of_stay_days'] = \
        (visit_level_features['discharge_date'] - visit_level_features['start_date_of_inpatient_visit']).dt.days
    
    # Define the target variable
    visit_level_features['is_high_risk_visit'] = visit_level_features['num_late_charges_per_visit'] > 0
    
    # Select and reorder columns for clarity if desired
    final_columns = [
        'visit_id', 'patient_id', 'room_type', 'length_of_stay_days',
        'total_charges_per_visit', 'num_total_charges_per_visit',
        'num_late_charges_per_visit', 'amount_of_late_charges_per_visit',
        'is_high_risk_visit', 
    ]
    existing_final_columns = [col for col in final_columns if col in visit_level_features.columns]
    visit_level_df = visit_level_features[existing_final_columns]

    return visit_level_df

def train_and_evaluate_model(visit_level_df):
    """
    Trains a Decision Tree Classifier to predict 'is_high_risk_visit' and evaluates it.

    Args:
        visit_level_df (pandas.DataFrame): DataFrame from load_and_prepare_data().

    Returns:
        tuple: (trained_model, accuracy, classification_report_str, feature_importances_dict)
               or None if an error occurs.
    """
    if visit_level_df is None or visit_level_df.empty:
        print("Error: Input DataFrame is empty or None.")
        return None

    try:
        # One-hot encode 'room_type' before splitting
        visit_level_df_encoded = pd.get_dummies(visit_level_df, columns=['room_type'], prefix='room', dummy_na=False)

        # Define features (X) and target (y)
        excluded_cols = ['visit_id', 'patient_id', 'is_high_risk_visit', 
                         'start_date_of_inpatient_visit', 'discharge_date'] 
        
        feature_columns = [col for col in visit_level_df_encoded.columns if col not in excluded_cols]
        
        X = visit_level_df_encoded[feature_columns]
        y = visit_level_df_encoded['is_high_risk_visit']

        if X.empty or len(X) != len(y):
            print("Error: Feature set X is empty or has mismatched length with target y after encoding.")
            return None
        
        if y.nunique() < 2:
            print(f"Warning: The target variable 'is_high_risk_visit' has only {y.nunique()} unique value(s). Model training might not be meaningful.")
            if y.nunique() == 0: return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)

        # Initialize and train Decision Tree Classifier
        model = DecisionTreeClassifier(random_state=42, max_depth=7)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cl_report_str = classification_report(y_test, y_pred, zero_division=0)

        # Get feature importances
        importances = model.feature_importances_
        feature_names = X.columns 
        feature_importances_dict = dict(zip(feature_names, importances))
        feature_importances_dict = dict(sorted(feature_importances_dict.items(), key=lambda item: item[1], reverse=True))

        return model, accuracy, cl_report_str, feature_importances_dict

    except Exception as e:
        print(f"An error occurred during model training or evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print("Attempting to load and prepare data...")
    analysis_df = load_and_prepare_data()

    if analysis_df is not None:
        print("\n--- Resulting DataFrame Head (from load_and_prepare_data) ---")
        print(analysis_df.head())
        
        print("\nAttempting to train and evaluate model...")
        model_results = train_and_evaluate_model(analysis_df)

        if model_results:
            trained_model, accuracy, cl_report, feat_importances = model_results
            
            print("\n--- Model Evaluation Results ---")
            print("Approach: Identifying visits prone to bad debt by predicting 'is_high_risk_visit' (any late charges) using a Decision Tree Classifier based on visit characteristics like charges, length of stay, and room type.")
            print("ML Algorithm: Decision Tree Classifier")
            print(f"\nAccuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(cl_report)
            print("\nFeature Importances:")
            if feat_importances:
                for feature, importance in feat_importances.items():
                    print(f"- {feature}: {importance:.4f}")
                
                print("\nPotential Recommendations:")
                top_n = 5 
                recommendation_count = 0
                for feature, importance_score in list(feat_importances.items())[:top_n]:
                    if importance_score < 0.01 and recommendation_count > 0 : 
                        break 

                    if recommendation_count >= 3 and importance_score < 0.05: 
                        break

                    recommendation_made = False
                    if 'amount_of_late_charges_per_visit' == feature : 
                        print("- Investigate and streamline processes related to charge capture for services that are frequently posted late. Reducing the number and amount of late charges is crucial.")
                        recommendation_made = True
                    elif 'length_of_stay_days' == feature:
                        print("- Analyze long-stay visits to identify potential inefficiencies in billing during or post-discharge. Ensure all services for long stays are captured promptly.")
                        recommendation_made = True
                    elif feature.startswith('room_'): 
                        room_name = feature.replace('room_', '').replace('_', ' ') 
                        print(f"- Review billing accuracy and timeliness specifically for visits involving {room_name.title()}. These may have more complex billing that could lead to errors or delays.")
                        recommendation_made = True
                    elif 'total_charges_per_visit' == feature:
                        print("- For visits with very high total charges, ensure extra scrutiny on charge accuracy and completeness before final billing to prevent payment delays or disputes that could arise from late additions.")
                        recommendation_made = True
                    
                    if recommendation_made:
                        recommendation_count +=1
                        if recommendation_count >= 3 and top_n == 3: 
                            break
                
                if recommendation_count == 0:
                    print("- No specific recommendations generated based on the top features' thresholds. Review feature importances for further insights.")
            else:
                print("Could not retrieve feature importances, so no recommendations can be generated.")
        else:
            print("Model training and evaluation failed.")
    else:
        print("Data loading failed. Please check file paths and ensure generate_data.py has been run.")
