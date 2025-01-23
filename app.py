import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('machine downtime.pickle','rb'))
@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    output=model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]  # Get input data from form
        final_features = [np.array(data)]  # Prepare data for prediction

        # Perform the prediction using the model
        prediction = model.predict(final_features)[0]

        # Convert output to "Yes" or "No"
        result = "Yes" if prediction == 0 else "No"

        return render_template('home.html', prediction_text=f"Machine Downtime: {result}")

    except Exception as e:
        return render_template('home.html', prediction_text=f"Error: {str(e)}")

if __name__=="__main__":
    app.run(debug=True)