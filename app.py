# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:13:30 2020

@author: shhiv
"""

import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

"# Initialize falsk app"
app = Flask(__name__)

"# Loading the model"
model = pickle.load(open('model.pkl', 'rb'))

#model.predict([[4323.00,3214,12345,0,1,0]])

#print(model.predict([[123107.34, 41231.77	,661238.43]]))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])

def predict():
    features = [int(i) for i in request.form.values()]
    """
    x  =  features[-1]
    ints = list(map(int, features[:-1]))
    if(x  == 'California'):
       ints.extend([1,0,0])
    elif(x == 'Florida'):
        ints.extend([0,1,0])
    elif(x == 'New York'):
        ints.extend([0,0,1])
    else:
        ints.extend([0,0,0])

    #final_featuers = np.array(features)
    """
    preds = model.predict([features])
    
    op = np.round(preds, 2)
    
    return render_template('index.html', prediction_text = 'Profit ${}'.format(op))

if __name__ == "__main__":
    app.run(debug=True)


 


"""
features = [i for i in [4567.00,3456,12345,'California']]
x  =  features[-1]
ints = list(map(int, features[:-1]))
if(x  == 'California'):
    ints.extend([1,0,0])
elif(x == 'Florida'):
    ints.extend([0,1,0])
elif(x == 'New York'):
    ints.extend([0,0,1])
else:
    ints.extend([0,0,0])

    #final_featuers = np.array(features)
    
preds = model.predict([ints])
    
op = np.round(preds, 2)"""