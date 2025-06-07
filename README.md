# healthcare

Initial Prompt to Jules.google.com


Use my github repo to upload the data and code: https://github.com/ansarmuhammad/healthcare
Please upload all data and code to this repo without asking, just do it as soon as a you finish a task


I need you to please create for me the following datasets in csv format for an inpatient hospital scenarios and focus on the inpatient billing system

1. Patient record (assumption is that the patient has been discharged)
2. Patient visit record (specially mention the discharge_date, start_date_of_inpatient_visit, room_type, etc.)
3. Charge which mentions the service_code, charge_amount, patient_visit, charge_date, charge_transfer_date_from_lab
4. Create charge data above such that 100% of the charges are on or before discharge_date for 90% of the patients
5. For late_charges which are those charges which are posted after the patient discharge_date, mention the charge_transfer_date_from_lab as a date after discharge_date of the patient
6. Generate record for atleast 100 patients
7. Generate atleast 100 charges for each patient

******************* JULES MADE SOME ERRORS INITIALLY BUT AFTER A FEW MINOR PROMPTS I HAD MY DATA ************************************************

Next prompt is below

awesome, good job. now can you now use all the files above to find out what is wrong with the process

what can lead to bad debt and losses

write a new program that achieves this in a repeatable way using all the data you generated

the program should clearly output

which approach it took
mention the machine learning algorithm it used
mention what it found
mention how to fix it
