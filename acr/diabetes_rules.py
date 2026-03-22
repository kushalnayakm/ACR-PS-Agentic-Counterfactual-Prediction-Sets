DIABETES_RULES = {
    # Immutable features: Age and Pedigree (Genetics)
    'Age': {'mutable': False},
    'DiabetesPedigreeFunction': {'mutable': False},
    
    # Mutable: Glucose, BMI, BloodPressure, Insulin, SkinThickness
    'Glucose': {'mutable': True, 'constraint': None},
    'BMI': {'mutable': True, 'constraint': None},
    'BloodPressure': {'mutable': True, 'constraint': None},
    'Insulin': {'mutable': True, 'constraint': None},
    'Pregnancies': {'mutable': True, 'constraint': 'increase_only'}, 
    # Logic: Pregnancies can't decrease (unless we consider it immutable)
}
