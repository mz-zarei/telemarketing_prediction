"""
@author: Mohammad Zarei
"""

import pickle
from flask import Flask, jsonify, request


# Declare a flask app
app = Flask(__name__)

# Some inputs
MODEL_PATH = "../development/rf_model.sav"

# Load the trained model
model = pickle.load(open(MODEL_PATH, 'rb'))


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json(force=True)
        data = [list(data.values())]
        pred = model.predict(data)[0]

        return jsonify({'result': str(pred)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)