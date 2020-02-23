from flask import Flask, Response, render_template, url_for, request
import pandas as pd
import numpy as np
import re
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve, auc, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import  RobustScaler 


app = Flask(__name__)

@app.route('/')
def home():
     return render_template('home.html')

@app.route('/download')
def download():
    with open("data/pred.csv") as fp:
         csv = fp.read()
    return Response(csv, mimetype="text/csv", headers={"Content-disposition": "attachment; filename=predict.csv"})


## Character Encoding
risk_map = {'No Bureau History Available':-1, 
          'Not Scored: No Activity seen on the customer (Inactive)':-1,
          'Not Scored: Sufficient History Not Available':-1,
          'Not Scored: No Updates available in last 36 months':-1,
          'Not Scored: Only a Guarantor':-1,
          'Not Scored: More than 50 active Accounts found':-1,
          'Not Scored: Not Enough Info available on the customer':-1,
          'Very Low Risk':4,
          'Low Risk':3,
          'Medium Risk':2, 
          'High Risk':1,
          'Very High Risk':0
        }
sub_risk =  {
            'unknown':-1,
            'I':5,
            'L':2,
            'A':13,
            'D':10,
            'M':1,
            'B':12,
            'C':11,
            'E':9,
            'H':6,
            'F':8,
            'K':3,
            'G':7,
            'J':4
            }
employment_map = {
    'Self employed':0,
    'Salaried':1,
    np.nan:-1
}


@app.route('/predict', methods=['POST'])
def predict():

    df = pd.read_csv ("data/train.csv")
    
    ## Outliner Treatment
    cols_with_outliers=['disbursed_amount', 'asset_cost', 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS',
                    'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT','PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT',
                    'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS','SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT', 
                    'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT', 'NEW.ACCTS.IN.LAST.SIX.MONTHS', 
                    'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']
    for col in cols_with_outliers:
        if (df[col].dtypes !='object'): 
            l,u = outlier_treatment(df[col])
            df[ (df[col] > u) | (df[col] < l) ]
            df.drop(df[ (df[col] > u) | (df[col] < l) ].index , inplace=True)


    ## Drop Features            
    delete_cols = ["UniqueID", "MobileNo_Avl_Flag", "manufacturer_id", "branch_id", "State_ID", "supplier_id", "Employee_code_ID", "Current_pincode_ID", "ltv"]
    df.drop(delete_cols, axis=1, inplace=True)

    ## Prepare Data
    train_data = prepare_data(df)

    if 'pred_file' in request.files:
        pred_file = request.files['pred_file']
    if pred_file.filename != '': 
        test_data = pd.read_csv (pred_file)
        test_data = prepare_data(test_data)
        

    ## Drop un wanted fields from Train Data
    to_drop = ['Date.of.Birth', 'Employment.Type', 'DisbursalDate', 
           'PRIMARY.INSTAL.AMT', 'PERFORM_CNS.SCORE.DESCRIPTION', 'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 
           'Age',  'credit_risk', 'credit_risk_grade',
       ]

    train_data.drop(to_drop, axis=1, inplace=True)
    test_data.drop(to_drop, axis=1, inplace=True)
    ## Train & Test 
    X = train_data.drop('loan_default',axis=1)
    y = train_data['loan_default']

    ## Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)

    Predictions = logmodel.predict(test_data)
    print("Predictions:")
    print(Predictions)
    
    av = accuracy_score(test_data, Predictions)
    print("Accuracy Score")
    print(av)

    cr = classification_report(test_data, Predictions)
    print("Classification Report")
    print(cr)

    return render_template('result.html', pred = Predictions, av = av, cr = cr)

## Outlier Engineering
def outlier_treatment(datacolumn):
        sorted(datacolumn)
        Q1,Q3 = np.percentile(datacolumn , [25,75])
        IQR = Q3 - Q1
        lower_range = Q1 - (1.5 * IQR)
        upper_range = Q3 + (1.5 * IQR)
        return lower_range,upper_range

## Feature Engineering
def features_engineering(df):
    df['DisbursalDate'] = pd.to_datetime(df['DisbursalDate'], format = "%d-%m-%y",infer_datetime_format=True)
    df['Date.of.Birth'] = pd.to_datetime(df['Date.of.Birth'], format = "%d-%m-%y",infer_datetime_format=True)
    now = pd.Timestamp('now')
    df['Age'] = (now - df['Date.of.Birth']).astype('<m8[Y]').astype(int)
    age_mean = int(df[df['Age']>0]['Age'].mean())
    df.loc[:,'age'] = df['Age'].apply(lambda x: x if x>0 else age_mean)
    df['disbursal_months_passed'] = ((now - df['DisbursalDate'])/np.timedelta64(1,'M')).astype(int)
    df['average_act_age_in_months'] = df['AVERAGE.ACCT.AGE'].apply(lambda x : int(re.findall(r'\d+',x)[0])*12 + int(re.findall(r'\d+',x)[1]))
    df['credit_history_length_in_months'] = df['CREDIT.HISTORY.LENGTH'].apply(lambda x : int(re.findall(r'\d+',x)[0])*12 + int(re.findall(r'\d+',x)[1]))
    df['number_of_0'] = (df == 0).astype(int).sum(axis=1)
    
    df.loc[:,'credit_risk'],df.loc[:,'credit_risk_grade']  = credit_risk(df["PERFORM_CNS.SCORE.DESCRIPTION"])
    
    df.loc[:,'loan_to_asset_ratio'] = df['disbursed_amount'] /df['asset_cost']
    df.loc[:,'no_of_accts'] = df['PRI.NO.OF.ACCTS'] + df['SEC.NO.OF.ACCTS']
    df.loc[:,'no_of_active_accts'] = df['PRI.ACTIVE.ACCTS'] + df['SEC.ACTIVE.ACCTS']
    df.loc[:,'pri_inactive_accts'] = df['PRI.NO.OF.ACCTS'] - df['PRI.ACTIVE.ACCTS']
    df.loc[:,'sec_inactive_accts'] = df['SEC.NO.OF.ACCTS'] - df['SEC.ACTIVE.ACCTS']
    df.loc[:,'tot_inactive_accts'] = df['pri_inactive_accts'] + df['sec_inactive_accts']
    df.loc[:,'tot_overdue_accts'] = df['PRI.OVERDUE.ACCTS'] + df['SEC.OVERDUE.ACCTS']
    df.loc[:,'tot_current_balance'] = df['PRI.CURRENT.BALANCE'] + df['SEC.CURRENT.BALANCE']
    df.loc[:,'tot_sanctioned_amount'] = df['PRI.SANCTIONED.AMOUNT'] + df['SEC.SANCTIONED.AMOUNT']
    df.loc[:,'tot_disbursed_amount'] = df['PRI.DISBURSED.AMOUNT'] + df['SEC.DISBURSED.AMOUNT']
    df.loc[:,'tot_installment'] = df['PRIMARY.INSTAL.AMT'] + df['SEC.INSTAL.AMT']
    df.loc[:,'bal_disburse_ratio'] = np.round((1+df['tot_disbursed_amount'])/(1+df['tot_current_balance']),2)
    df.loc[:,'pri_tenure'] = (df['PRI.DISBURSED.AMOUNT']/( df['PRIMARY.INSTAL.AMT']+1)).astype(int)
    df.loc[:,'sec_tenure'] = (df['SEC.DISBURSED.AMOUNT']/(df['SEC.INSTAL.AMT']+1)).astype(int)
    df.loc[:,'disburse_to_sactioned_ratio'] =  np.round((df['tot_disbursed_amount']+1)/(1+df['tot_sanctioned_amount']),2)
    df.loc[:,'active_to_inactive_act_ratio'] =  np.round((df['no_of_accts']+1)/(1+df['tot_inactive_accts']),2)
    return df



def credit_risk(df):
    d1=[]
    d2=[]
    for i in df:
        p = i.split("-")
        if len(p) == 1:
            d1.append(p[0])
            d2.append('unknown')
        else:
            d1.append(p[1])
            d2.append(p[0])

    return d1,d2

def check_pri_installment(row):
    if row['PRIMARY.INSTAL.AMT']<=1:
        return 0
    else:
        return row['PRIMARY.INSTAL.AMT']

def label_data(df):
    print('labeling started')
    df.loc[:,'credit_risk_label'] = df['credit_risk'].apply(lambda x: risk_map[x])
    df.loc[:,'sub_risk_label'] = df['credit_risk_grade'].apply(lambda x: sub_risk[x])
    df.loc[:,'employment_label'] = df['Employment.Type'].apply(lambda x: employment_map[x])
    print('labeling done')
    return df

def data_correction(df):
    print('invalid data handling started')
    #Many customers have invalid date of birth, so immute invalid data with mean age
    df.loc[:,'PRI.CURRENT.BALANCE'] = df['PRI.CURRENT.BALANCE'].apply(lambda x: 0 if x<0 else x)
    df.loc[:,'SEC.CURRENT.BALANCE'] = df['SEC.CURRENT.BALANCE'].apply(lambda x: 0 if x<0 else x)
    
    #loan that do not have current pricipal outstanding should have 0 primary installment
    df.loc[:,'new_pri_installment']= df.apply(lambda x : check_pri_installment(x),axis=1)
    print('done')
    return df

def prepare_data(df):
    df = data_correction(df)
    df = features_engineering(df)
    df = label_data(df)
    return df        



if __name__ == '__main__':
     app.run(debug=False)
