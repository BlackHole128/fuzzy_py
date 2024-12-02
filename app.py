from flask import Flask, request, jsonify
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins
# Define fuzzy variables
age = ctrl.Antecedent(np.arange(0, 101, 1), 'age')
sex = ctrl.Antecedent(np.arange(0, 2, 1), 'sex')  # 0 = Female, 1 = Male
bone_density = ctrl.Antecedent(np.arange(0, 101, 1), 'bone_density')  # for Osteoporosis
blood_pressure = ctrl.Antecedent(np.arange(90, 201, 1), 'blood_pressure')  # for Hypertension

# Define the drug output for both diseases
drug_osteoporosis = ctrl.Consequent(np.arange(0, 3, 1), 'drug_osteoporosis')
drug_hypertension = ctrl.Consequent(np.arange(0, 3, 1), 'drug_hypertension')

# Membership functions for Age
age['young'] = fuzz.trimf(age.universe, [0, 0, 40])
age['middle_aged'] = fuzz.trimf(age.universe, [40, 50, 60])
age['elderly'] = fuzz.trimf(age.universe, [60, 80, 100])

# Membership functions for Sex
sex['female'] = fuzz.trimf(sex.universe, [0, 0, 1])
sex['male'] = fuzz.trimf(sex.universe, [1, 1, 1])

# Membership functions for Bone Density
bone_density['mild'] = fuzz.trimf(bone_density.universe, [0, 0, 50])
bone_density['moderate'] = fuzz.trimf(bone_density.universe, [30, 50, 70])
bone_density['severe'] = fuzz.trimf(bone_density.universe, [60, 100, 100])

# Membership functions for Blood Pressure
blood_pressure['normal'] = fuzz.trimf(blood_pressure.universe, [90, 90, 120])
blood_pressure['high'] = fuzz.trimf(blood_pressure.universe, [120, 150, 180])
blood_pressure['very_high'] = fuzz.trimf(blood_pressure.universe, [170, 200, 200])

# Membership functions for Osteoporosis Drug
drug_osteoporosis['raloxifene'] = fuzz.trimf(drug_osteoporosis.universe, [0, 0, 1])
drug_osteoporosis['alendronate'] = fuzz.trimf(drug_osteoporosis.universe, [1, 1, 2])
drug_osteoporosis['denosumab'] = fuzz.trimf(drug_osteoporosis.universe, [2, 2, 2])

# Membership functions for Hypertension Drug
drug_hypertension['lisinopril'] = fuzz.trimf(drug_hypertension.universe, [0, 0, 1])
drug_hypertension['amlodipine'] = fuzz.trimf(drug_hypertension.universe, [1, 1, 2])
drug_hypertension['hydrochlorothiazide'] = fuzz.trimf(drug_hypertension.universe, [2, 2, 2])

# Fuzzy rules for Osteoporosis
rule1 = ctrl.Rule(age['elderly'] & sex['female'] & bone_density['mild'], drug_osteoporosis['raloxifene'])
rule2 = ctrl.Rule(age['elderly'] & sex['female'] & bone_density['moderate'], drug_osteoporosis['raloxifene'])
rule3 = ctrl.Rule(age['middle_aged'] & bone_density['moderate'], drug_osteoporosis['alendronate'])
rule4 = ctrl.Rule(age['middle_aged'] & bone_density['severe'], drug_osteoporosis['alendronate'])
rule5 = ctrl.Rule(age['elderly'] & bone_density['moderate'], drug_osteoporosis['alendronate'])
rule6 = ctrl.Rule(age['elderly'] & bone_density['severe'], drug_osteoporosis['denosumab'])

# Fuzzy rules for Hypertension
rule7 = ctrl.Rule(age['young'] & sex['male'] & blood_pressure['very_high'], drug_hypertension['lisinopril'])
rule8 = ctrl.Rule(age['middle_aged'] & blood_pressure['high'], drug_hypertension['amlodipine'])
rule9 = ctrl.Rule(age['elderly'] & blood_pressure['very_high'], drug_hypertension['hydrochlorothiazide'])
rule10 = ctrl.Rule(sex['female'] & blood_pressure['high'], drug_hypertension['amlodipine'])

# Control systems
osteoporosis_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
hypertension_ctrl = ctrl.ControlSystem([rule7, rule8, rule9, rule10])
osteoporosis_prescription = ctrl.ControlSystemSimulation(osteoporosis_ctrl)
hypertension_prescription = ctrl.ControlSystemSimulation(hypertension_ctrl)


def get_prescribed_drug(age_input, sex_input, bone_density_input, blood_pressure_input):
    osteoporosis_prescription.input['age'] = age_input
    osteoporosis_prescription.input['sex'] = sex_input
    osteoporosis_prescription.input['bone_density'] = bone_density_input
    osteoporosis_prescription.compute()

    hypertension_prescription.input['age'] = age_input
    hypertension_prescription.input['sex'] = sex_input
    hypertension_prescription.input['blood_pressure'] = blood_pressure_input
    hypertension_prescription.compute()

    # Osteoporosis drugs
    osteoporosis_output = osteoporosis_prescription.output['drug_osteoporosis']
    if osteoporosis_output < 1:
        osteoporosis_drug = "Raloxifene"
    elif osteoporosis_output < 2:
        osteoporosis_drug = "Alendronate"
    else:
        osteoporosis_drug = "Denosumab"

    # Hypertension drugs
    hypertension_output = hypertension_prescription.output['drug_hypertension']
    if hypertension_output < 1:
        hypertension_drug = "Lisinopril"
    elif hypertension_output < 2:
        hypertension_drug = "Amlodipine"
    else:
        hypertension_drug = "Hydrochlorothiazide"

    return osteoporosis_drug, hypertension_drug

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = data['age']
    sex = data['sex']
    bone_density = data['bone_density']
    blood_pressure = data['blood_pressure']

    osteoporosis_drug, hypertension_drug = get_prescribed_drug(age, sex, bone_density, blood_pressure)

    return jsonify({
        "osteoporosis_drug": osteoporosis_drug,
        "hypertension_drug": hypertension_drug
    })

if __name__ == '__main__':
    app.run(debug=True)
