from distutils.log import debug
from fileinput import filename
import pandas as pd
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import pickle
import classification
import json

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSION = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.secret_key = '12345678'

model = pickle.load(open("model.pkl", "rb"))

@app.route('/', methods=['GET'])
def hello_world():
    return "Hello World"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        lead1 = request.form["ekg1"]
        lead2 = request.form["ekg2"]

        sample_time = ["00:00:0" + str("%.3f" % round(i * 0.004, 3)) for i in range(len(json.loads(lead2)))]

        dict = {'time': sample_time, 'lead1': json.loads(lead1), 'lead2': json.loads(lead2)}
        df = pd.DataFrame(dict)

        feature = classification.cek(df)
        result = model.predict(feature)
        return json.dumps({"hasil": str(result)})
    
if __name__ == '__main__':
    app.run(port=3000, debug=True)