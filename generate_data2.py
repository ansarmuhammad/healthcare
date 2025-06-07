import pandas as pd
from faker import Faker
import random
import os

def generate_patient_data(num_patients):
    """
    Generates a Pandas DataFrame with synthetic patient data.

    Args:
        num_patients (int): The number of patients to generate.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'patient_id', 'name', 
                          'date_of_birth', and 'gender'.
    """
    fake = Faker()
    data = []
    genders = ["Male", "Female"]  # Define possible genders

    for i in range(1, num_patients + 1):
        name = fake.name()
        # Generate date of birth as a string in YYYY-MM-DD format
        date_of_birth = fake.date_of_birth(minimum_age=0, maximum_age=90).strftime('%Y-%m-%d')
        gender = random.choice(genders)  # Randomly select gender
        data.append([i, name, date_of_birth, gender])

    df = pd.DataFrame(data, columns=['patient_id', 'name', 'date_of_birth', 'gender'])
    return df

def generate_patient_visit_data(patient_df):
    """
    Generates a Pandas DataFrame with synthetic patient visit data.

    Args:
        patient_df (pandas.DataFrame): DataFrame containing patient data,
                                     must include 'patient_id'.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'visit_id', 'patient_id',
                          'start_date_of_inpatient_visit', 'discharge_date',
                          and 'room_type'.
    """
    fake = Faker()
    visit_data = []
    visit_id_counter = 1
    room_types = ['General Ward', 'Semi-Private', 'Private', 'ICU']
    today = pd.to_datetime('today')

    for index, patient in patient_df.iterrows():
        num_visits_for_patient = random.randint(1, 5) # Each patient has 1 to 5 visits

        for _ in range(num_visits_for_patient):
            # Ensure start_date is before today
            start_date = fake.date_time_between(start_date='-5y', end_date='now', tzinfo=None)
            
            # Ensure discharge_date is after start_date and before or equal to today
            # Max discharge days after start_date to avoid unrealistically long stays
            max_discharge_days = (today - start_date).days if (today - start_date).days > 0 else 1
            if max_discharge_days == 1 and (today - start_date).seconds < 1 : # handle case where start_date is very close to today
                 discharge_date = start_date + pd.Timedelta(days=random.randint(1,10))
                 if discharge_date > today:
                    discharge_date = today
            else:
                discharge_date = fake.date_time_between(start_date=start_date, end_date=start_date + pd.Timedelta(days=random.randint(1,min(30,max_discharge_days))), tzinfo=None)
                # if by chance discharge_date became same as start_date, add a small delta
                if discharge_date == start_date:
                    discharge_date = start_date + pd.Timedelta(days=random.randint(1,10))

                # cap discharge date to today
                if discharge_date > today:
                    discharge_date = today
            
            # Final check to ensure start_date is strictly before discharge_date
            if start_date >= discharge_date:
                discharge_date = start_date + pd.Timedelta(days=random.randint(1,10))
                if discharge_date > today: # Recalculate if it exceeds today
                    start_date = fake.date_time_between(start_date='-5y', end_date='-1d', tzinfo=None) # ensure start is well in past
                    discharge_date = fake.date_time_between(start_date=start_date, end_date=start_date + pd.Timedelta(days=random.randint(1,30)), tzinfo=None)
                    if discharge_date > today:
                        discharge_date = today
                    if start_date >= discharge_date: # if still an issue, set discharge to be start_date + 1 day (or today if that's sooner)
                         discharge_date = min(start_date + pd.Timedelta(days=1), today)


            room_type = random.choice(room_types)
            
            visit_data.append([
                visit_id_counter,
                patient['patient_id'],
                start_date.strftime('%Y-%m-%d %H:%M:%S'),
                discharge_date.strftime('%Y-%m-%d %H:%M:%S'),
                room_type
            ])
            visit_id_counter += 1
            
    df_visits = pd.DataFrame(visit_data, columns=[
        'visit_id', 'patient_id', 'start_date_of_inpatient_visit',
        'discharge_date', 'room_type'
    ])
    return df_visits

def generate_charge_data(patient_visit_df):
    """
    Generates a Pandas DataFrame with synthetic charge data for patient visits.

    Args:
        patient_visit_df (pandas.DataFrame): DataFrame containing patient visit data,
                                           must include 'visit_id', 'patient_id',
                                           'start_date_of_inpatient_visit', and 'discharge_date'.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'charge_id', 'visit_id', 
                          'service_code', 'charge_amount', 'charge_date', 
                          and 'charge_transfer_date_from_lab'.
    """
    fake = Faker()
    charge_data = []
    charge_id_counter = 1
    service_codes = [f'S{str(i).zfill(3)}' for i in range(1, 21)] # S001, S002, ..., S020
    today = pd.to_datetime('today')

    # Identify patients for late charges (10% of unique patient_ids in patient_visit_df)
    unique_patient_ids = patient_visit_df['patient_id'].unique()
    num_late_charge_patients = int(len(unique_patient_ids) * 0.1)
    late_charge_patient_ids = random.sample(list(unique_patient_ids), k=max(1, num_late_charge_patients)) # ensure at least one if possible

    # Debug: print which patients are selected for late charges
    # print(f"Patients selected for late charges: {late_charge_patient_ids}")


    for index, visit in patient_visit_df.iterrows():
        visit_id = visit['visit_id']
        patient_id = visit['patient_id'] # Get patient_id for this visit
        # Convert string dates from visit_df to datetime objects
        visit_start_date = pd.to_datetime(visit['start_date_of_inpatient_visit'])
        visit_discharge_date = pd.to_datetime(visit['discharge_date'])
        
        num_charges_for_visit = random.randint(100, 150) # Generate 100 to 150 charges per visit
        
        # Determine if this visit belongs to a patient who should have late charges
        patient_has_late_charges = patient_id in late_charge_patient_ids
        # print(f"Visit ID: {visit_id}, Patient ID: {patient_id}, Belongs to late charge patient: {patient_has_late_charges}")

        # Flag to ensure at least one late charge is generated for the selected patients' visits
        # This needs to be per-patient, not per-visit. We'll handle this by checking if a late charge has already been made for this patient.
        # For simplicity in this structure, we'll try to make one charge late if patient_has_late_charges is true for this visit.
        # A more robust approach might involve tracking late charges per patient across all their visits.
        made_a_late_charge_for_this_visit = False


        for i in range(num_charges_for_visit):
            service_code = random.choice(service_codes)
            charge_amount = round(random.uniform(10.00, 1000.00), 2)
            
            is_late_charge = False
            if patient_has_late_charges and not made_a_late_charge_for_this_visit and i == 0: # Try to make the first charge late for simplicity
                is_late_charge = True
                made_a_late_charge_for_this_visit = True # Mark that a late charge was made for this specific visit

            if is_late_charge:
                # Generate charge_date *after* discharge_date
                # Ensure it's not too far in the future from discharge, e.g., within 30 days after discharge
                charge_date_dt = fake.date_time_between(start_date=visit_discharge_date + pd.Timedelta(days=1), 
                                                      end_date=visit_discharge_date + pd.Timedelta(days=30), tzinfo=None)
                if charge_date_dt > today : charge_date_dt = today # cap at today

                # charge_transfer_date_from_lab also after discharge_date for late charges
                charge_transfer_date_dt = fake.date_time_between(start_date=charge_date_dt, 
                                                                end_date=charge_date_dt + pd.Timedelta(days=5), tzinfo=None)
                if charge_transfer_date_dt > today: charge_transfer_date_dt = today # cap at today
                # print(f"Late Charge! Visit: {visit_id}, Discharge: {visit_discharge_date}, Charge Date: {charge_date_dt}, Transfer: {charge_transfer_date_dt}")

            else:
                # Generate charge_date on or before discharge_date (but after or on visit_start_date)
                if visit_start_date == visit_discharge_date: # Handle cases where start and discharge are same day
                     charge_date_dt = visit_start_date
                else:
                    try:
                        charge_date_dt = fake.date_time_between(start_date=visit_start_date, end_date=visit_discharge_date, tzinfo=None)
                    except ValueError: # Can happen if start_date is exactly discharge_date due to previous logic
                        charge_date_dt = visit_start_date


                # charge_transfer_date_from_lab on or before discharge_date (but after or on charge_date)
                if charge_date_dt == visit_discharge_date:
                    charge_transfer_date_dt = charge_date_dt
                else:
                    try:
                        charge_transfer_date_dt = fake.date_time_between(start_date=charge_date_dt, end_date=visit_discharge_date, tzinfo=None)
                    except ValueError: # Can happen if charge_date_dt is exactly visit_discharge_date
                         charge_transfer_date_dt = charge_date_dt


            charge_data.append([
                charge_id_counter,
                visit_id,
                service_code,
                charge_amount,
                charge_date_dt.strftime('%Y-%m-%d %H:%M:%S'),
                charge_transfer_date_dt.strftime('%Y-%m-%d %H:%M:%S')
            ])
            charge_id_counter += 1

    df_charges = pd.DataFrame(charge_data, columns=[
        'charge_id', 'visit_id', 'service_code', 'charge_amount', 
        'charge_date', 'charge_transfer_date_from_lab'
    ])
    return df_charges

if __name__ == '__main__':
    # Generate data
    # Generate data
    patient_df = generate_patient_data(num_patients=100)
    visit_df = generate_patient_visit_data(patient_df)
    charge_df = generate_charge_data(visit_df)

    # Define output directory
    output_dir_name = "output_data"
    
    # Determine the directory where the script itself is located
    # This assumes __file__ is defined and gives the path to the current script.
    try:
        script_parent_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback if __file__ is not defined (e.g., in some interactive environments or specific execution contexts)
        # In this case, assume current working directory is the intended parent for 'output_data' 
        # or that 'healthcare_data_generation' is the intended base if running from repo root.
        # For consistency with the problem description's desire for output_data to be where the script is,
        # we will assume that if __file__ is not defined, the CWD is .../healthcare_data_generation/
        # or we are running from repo root and need to specify 'healthcare_data_generation' explicitly.
        # Given the previous turn's success with a hardcoded base, we'll try a flexible approach.
        # If script is at /app/healthcare_data_generation/generate_data.py, then script_parent_dir is /app/healthcare_data_generation
        # This should work if script is run from /app or /app/healthcare_data_generation
        script_parent_dir = os.getcwd() 
        if not script_parent_dir.endswith("healthcare_data_generation"):
            # If CWD is not the script's directory (e.g. /app), append it.
            script_parent_dir = os.path.join(script_parent_dir, "healthcare_data_generation")


    # Define the full path for the output directory, inside the script's directory
    output_path = os.path.join(script_parent_dir, output_dir_name)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Define file paths within the new output directory
    patients_csv_path = os.path.join(output_path, 'patients.csv')
    visits_csv_path = os.path.join(output_path, 'patient_visits.csv')
    charges_csv_path = os.path.join(output_path, 'charges.csv')

    # Save DataFrames to CSV files without the index
    patient_df.to_csv(patients_csv_path, index=False)
    visit_df.to_csv(visits_csv_path, index=False)
    charge_df.to_csv(charges_csv_path, index=False)

    print(f"Patient data saved to {patients_csv_path}")
    print(f"Patient visit data saved to {visits_csv_path}")
    print(f"Charge data saved to {charges_csv_path}")
