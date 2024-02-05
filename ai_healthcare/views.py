from django.shortcuts import render
import pandas as pd
import joblib
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
def index(request):
    return render(request,'index.html')


def diabetes_prediction(request):

    if request.method == 'POST':

        scaler = joblib.load(r'media\Diabetes Prediction\MinMax_Diabetes_Prediction.joblib')
        model = joblib.load(r'media\Diabetes Prediction\rnf_Diabetes_Prediction.joblib')

        gender = int(request.POST.get('gender'))
        age = float(request.POST.get('age'))
        hypertension = int(request.POST.get('hypertension'))
        heart_disease = int(request.POST.get('heart_disease'))
        smoking_history = int(request.POST.get('smoking_history'))
        bmi = float(request.POST.get('bmi'))
        HbA1c_level = float(request.POST.get('HbA1c_level'))
        blood_glucose_level = float(request.POST.get('blood_glucose_level'))

        input_features = [[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]]
        input_features_scaled = scaler.transform(input_features)

        predicted_diabetes_status = model.predict(input_features_scaled)

        if predicted_diabetes_status >= 0.5:
            percentage = predicted_diabetes_status[0] * 100
            prediction = f"The patient may have diabetes with a confidence of {percentage:.2f}%."
        else:
            prediction = "The patient may not have diabetes."

        return render(request, 'diabetes_prediction.html', {'prediction': prediction})

    return render(request,'diabetes_prediction.html')

def disease_prediction(request):
    data = pd.read_json('media\Disease Prediction\prediction_data.json')
    columns = [element.capitalize().replace('_', ' ') for element in data.columns]
    return render(request, 'disease_prediction.html', {'columns': columns})

def process_columns(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        selected_columns = data.get('columns', [])
        print('Selected Columns:', selected_columns)
        selected_column = [element.lower().replace(' ', '_') for element in selected_columns]
        predict_df = pd.read_json('media\Disease Prediction\prediction_data.json')
        predict_df = predict_df.apply(lambda col: 1 if col.name in selected_column else 0)
        rf_model = joblib.load(r'media\Disease Prediction\rf_Disease_Prediction.joblib') 
        predictions = rf_model.predict([predict_df.values.tolist()])
        print(predictions)
        result_message = f"The patient may be at {predictions[0]} Disease." 
        return render(request, 'disease_prediction_result.html', {'result_message': result_message})


    return render(request, 'disease_prediction.html', {'result_message': result_message})




def heart_attack_prediction(request):

    if request.method == 'POST':
        scaler = joblib.load(r'media\Heart Attack Prediction\scaler_Heart_Attack_Prediction.joblib')  
        rf_model = joblib.load(r'media\Diabetes Prediction\rnf_Diabetes_Prediction.joblib')  

        age = float(request.POST.get('age'))
        gender = int(request.POST.get('gender'))
        cp = int(request.POST.get('cp'))
        trestbps = float(request.POST.get('trestbps'))
        chol = float(request.POST.get('chol'))
        fbs = int(request.POST.get('fbs'))
        restecg = int(request.POST.get('restecg'))
        thalach = float(request.POST.get('thalach'))
        exang = int(request.POST.get('exang'))
        oldpeak = float(request.POST.get('oldpeak'))
        slope = int(request.POST.get('slope'))
        ca = int(request.POST.get('ca'))
        thal = int(request.POST.get('thal'))

        input_data = pd.DataFrame({'age': [age],'sex': [gender],'cp': [cp],'trtbps': [trestbps],'chol': [chol],'fbs': [fbs],'restecg': [restecg],'thalach': [thalach],'exng': [exang],'oldpeak': [oldpeak],'slp': [slope],'caa': [ca],'thall': [thal]})

        input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

        prediction = rf_model.predict(input_data_scaled)[0]

        if prediction >= 0.5:
            prediction *= 100
            result_message = f"The patient may be at {prediction} percentage risk of a heart attack."
        else:
            result_message = "The patient may not be at risk of a heart attack."

        return render(request, 'heart_attack_prediction.html', {'result_message': result_message})

    return render(request,'heart_attack_prediction.html')

def liver_disease_prediction(request):

    if request.method == 'POST':
        model = joblib.load(r'media\Liver Disease Prediction\Liver_Disease_Prediction.joblib')  

        age = float(request.POST.get('Age'))
        gender = int(request.POST.get('gender'))
        total_bilirubin = float(request.POST.get('Total_Bilirubin'))
        direct_bilirubin = float(request.POST.get('Direct_Bilirubin'))
        alkaline_phosphotase = float(request.POST.get('Alkaline_Phosphotase'))
        alamine_aminotransferase = float(request.POST.get('Alamine_Aminotransferase'))
        aspartate_aminotransferase = float(request.POST.get('Aspartate_Aminotransferase'))
        total_proteins = float(request.POST.get('Total_Protiens'))
        albumin = float(request.POST.get('Albumin'))
        albumin_globulin_ratio = float(request.POST.get('Albumin_and_Globulin_Ratio'))

        input_features = [[age, gender, total_bilirubin, direct_bilirubin,
                           alkaline_phosphotase, alamine_aminotransferase,
                           aspartate_aminotransferase, total_proteins, albumin,
                           albumin_globulin_ratio]]


        predicted_liver_disease = model.predict(input_features)
        
        if predicted_liver_disease >= 0.5:
            percentage = predicted_liver_disease[0] * 100
            prediction = f"The patient may have diabetes with a confidence of {percentage:.2f}%."
        else:
            prediction = "The patient may not have liver disease."

        return render(request, 'liver_disease_prediction.html', {'prediction': prediction})

    return render(request,'liver_disease_prediction.html')
    
