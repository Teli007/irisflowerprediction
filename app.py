# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'iris.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        SL = float(request.form['Sepal.Length'])
        SW = float(request.form['Sepal.Width'])
        PL = float(request.form['Petal.Length'])
        PW = float(request.form['Petal.Width'])
        
        data = np.array([[SL, SW, PL, PW,]])
        my_prediction = classifier.predict(data)
        
        return render_template("index.html",pred="Predicted_Flower:{} ".format(my_prediction))

if __name__ == '__main__':
	app.run(debug=True)