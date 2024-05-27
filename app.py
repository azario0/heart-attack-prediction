from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('encoders_scalers.pkl', 'rb') as file:
    encoders_scalers = pickle.load(file)

le_sex = encoders_scalers['le_sex']
le_chest_pain = encoders_scalers['le_chest_pain']
le_resting_ecg = encoders_scalers['le_resting_ecg']
le_exercise_angina = encoders_scalers['le_exercise_angina']
le_st_slope = encoders_scalers['le_st_slope']
mms_oldpeak = encoders_scalers['mms_oldpeak']
ss_age = encoders_scalers['ss_age']
ss_resting_bp = encoders_scalers['ss_resting_bp']
ss_cholesterol = encoders_scalers['ss_cholesterol']
ss_max_hr = encoders_scalers['ss_max_hr']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = request.form['sex']
        chest_pain = request.form['chest_pain']
        resting_bp = float(request.form['resting_bp'])
        cholesterol = float(request.form['cholesterol'])
        fasting_bs = int(request.form['fasting_bs'])
        resting_ecg = request.form['resting_ecg']
        max_hr = float(request.form['max_hr'])
        exercise_angina = request.form['exercise_angina']
        oldpeak = float(request.form['oldpeak'])
        st_slope = request.form['st_slope']

        sex = le_sex.transform([sex])[0]
        chest_pain = le_chest_pain.transform([chest_pain])[0]
        resting_ecg = le_resting_ecg.transform([resting_ecg])[0]
        exercise_angina = le_exercise_angina.transform([exercise_angina])[0]
        st_slope = le_st_slope.transform([st_slope])[0]

        age = ss_age.transform([[age]])[0][0]
        resting_bp = ss_resting_bp.transform([[resting_bp]])[0][0]
        cholesterol = ss_cholesterol.transform([[cholesterol]])[0][0]
        max_hr = ss_max_hr.transform([[max_hr]])[0][0]
        oldpeak = mms_oldpeak.transform([[oldpeak]])[0][0]
        
        input_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

        
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            result = "The patient is likely to have a heart attack."
        else:
            result = "The patient is unlikely to have a heart attack."

        return render_template('index.html', result=result)

    return render_template('index.html', result='')

if __name__ == '__main__':
    app.run(debug=True)
