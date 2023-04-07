import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#create flask app

app = Flask(__name__)

#Load pickle model

model = pickle.load(open("model.pkl", "rb"))

#Define method

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    
    return render_template("index.html", prediction_text="This iris specie is called {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)