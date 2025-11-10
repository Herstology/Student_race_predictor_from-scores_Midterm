import pickle
import numpy as np
from flask import Flask, request, render_template

# Load model and scaler
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        math = float(request.form['math_score'])
        reading = float(request.form['reading_score'])
        writing = float(request.form['writing_score'])

        features = np.array([[math, reading, writing]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        return render_template('index.html', prediction_text=f'Predicted Group: {prediction}')
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
