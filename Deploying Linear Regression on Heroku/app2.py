#app
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('ml_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('html_ui.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    #Fuel_Type_Diesel=0
    if request.method == 'POST':
        
        fixed_acidity = float(request.form['fixed acidity'])
        volatile_acidity = float(request.form['volatile acidity'])
        citric_acid = float(request.form['citric acid'])
        residual_sugar = float(request.form['residual sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free sulfur dioxide'])
        
        total_sulfur_dioxide = float(request.form['total sulfur dioxide'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol']) 
        
        prediction=model.predict([[fixed_acidity,volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,pH, sulphates, alcohol]])
        
        output=prediction
        if output==1:
            return render_template('html_ui.html',result="Low quality: Level 1")
        elif output==2:
            return render_template('html_ui.html',result="OK quality: Level 2")
        elif output==3:
            return render_template('html_ui.html',result="average quality: Level 3")
        elif output==4:
            return render_template('html_ui.html',result="good quality: Level 4")
        else:
            return render_template('html_ui.html',result="best quality: Level 5")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)