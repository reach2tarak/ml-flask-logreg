import os
import requests
import re
from flask import Flask, Response, render_template, url_for, request, send_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn

app = Flask(__name__)

@app.route('/')
def home():
     return render_template('home.html')

@app.route('/download')
def download():
    with open("/home/tarak/Desktop/MyPythonDS/ML-Flask-App/data/pred.csv") as fp:
         csv = fp.read()
    return Response(csv, mimetype="text/csv", headers={"Content-disposition": "attachment; filename=myplot.csv"})

@app.route('/predict', methods=['POST'])
def predict():

     df = pd.read_csv ("/home/tarak/Desktop/MyPythonDS/ML-Flask-App/data/input.csv")
     X = df[["gmat", "gpa","work_experience"]]
     y = df['admitted']
     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)  #in this case, you may choose to set the test_size=0. You should get the same prediction here
     logistic_regression= LogisticRegression()
     logistic_regression.fit(X_train,y_train)

     if 'pred_file' in request.files:
          pred_file = request.files['pred_file']
          print(pred_file)
     if pred_file.filename != '': 
          df2 = pd.read_csv (pred_file)
          
          y_pred=logistic_regression.predict(df2)
     else:
          y_pred='Unable to predict'	

     return render_template('result.html', pred = y_pred)

if __name__ == '__main__':
     app.run(debug=False)
