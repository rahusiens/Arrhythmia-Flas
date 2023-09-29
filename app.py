from distutils.log import debug
from fileinput import filename
import pandas as pd
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import pickle
import classification
import json
import requests

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSION = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.secret_key = '12345678'

model = pickle.load(open("model.pkl", "rb"))

@app.route('/', methods=['GET'])
def hello_world():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        id = request.form["patient_id"]
        lead1 = request.form["lead1"]
        lead2 = request.form["lead2"]
        
        lead1 = json.loads(lead1)
        lead2 = json.loads(lead2)

        sample_time = ["00:00:0" + str("%.3f" % round(i * 0.004, 3)) for i in range(len(lead2))]

        dict = {'time': sample_time, 'lead1': lead1, 'lead2': lead2}
        df = pd.DataFrame(dict)

        feature = classification.cek(df)
        result = model.predict(feature)

        r = requests.post("http://127.0.0.1:8000/api/send_data",
            data={'patient_id': id,
                    'time': str(sample_time),
                    'lead1': str(lead1),
                    'lead2': str(lead2)
            })

        return json.dumps({"hasil": str(result)})
    
if __name__ == '__main__':
    app.run(port=3000, debug=True)