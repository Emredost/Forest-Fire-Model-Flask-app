import pickle
from flask import Flask,request, jsonify, render_template

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app= application

## import ridge regressor and standart scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('/index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Retrieve form data
            temperature = float(request.form.get("Temperature"))
            rh = float(request.form.get("RH"))
            ws = float(request.form.get("Ws"))
            rain = float(request.form.get("Rain"))
            ffmc = float(request.form.get("FFMC"))
            dmc = float(request.form.get("DMC"))
            isi = float(request.form.get("ISI"))
            classes = float(request.form.get("Classes"))
            region = float(request.form.get("Region"))
            
            # Scale the input data
            input_data = standard_scaler.transform([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
            
            # Perform prediction
            prediction = ridge_model.predict(input_data)
            
            # Return prediction on the same page
            return render_template("home.html", result=prediction[0])
        
        except Exception as e:
            # Handle and display errors for debugging
            return render_template("home.html", result=f"Error: {e}")
    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")  