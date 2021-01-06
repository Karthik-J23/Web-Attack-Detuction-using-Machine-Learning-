from flask import Flask, jsonify, request,render_template
import numpy as np
import joblib
import pandas as pd
import re
import logging
logging.basicConfig(filename='error.log',level=logging.ERROR,format='%(asctime)s:%(levelname)s:%(message)s')
import sys

import flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'DDOS attack Detuction'

@app.route('/index')
def index():
    return flask.render_template('index.html')



@app.route('/predict', methods=['POST'])

def predict():

    traffic=pd.read_csv(request.form['csvfile'])

    bool_val=traffic.isnull().any().any()
    if bool_val == True:
        logging.error('some column values are left empty')
        return flask.render_template('empty_columns.html')
        sys.exit()



    feature_count=traffic.columns
    size=len(feature_count)

    if size < 79:
        logging.error('this file has less features in it')
        return flask.render_template('less_feat.html')
        sys.exit()

    value=traffic['Timestamp'].isin([0]).any().any()
    if value == True:
        logging.error('The Timestamp is zero')
        return flask.render_template('timestamp_zero.html')
        sys.exit()




    #Data PreProcessing
    traffic['Timestamp'] = pd.to_datetime(traffic['Timestamp']).astype(np.int64)
    columns=traffic.columns

    for i in columns:
        traffic[i]=traffic[i].astype(float)

    #Drooping the least important features (analysed via EDA)
    traffic.drop(['Bwd PSH Flags'],axis=1,inplace=True)
    traffic.drop(['Bwd URG Flags'],axis=1,inplace=True)
    traffic.drop(['Fwd Byts/b Avg'],axis=1,inplace=True)
    traffic.drop(['Fwd Pkts/b Avg'],axis=1,inplace=True)
    traffic.drop(['Fwd Blk Rate Avg'],axis=1,inplace=True)
    traffic.drop(['Bwd Byts/b Avg'],axis=1,inplace=True)
    traffic.drop(['Bwd Pkts/b Avg'],axis=1,inplace=True)
    traffic.drop(['Bwd Blk Rate Avg'],axis=1,inplace=True)

    traffic =  traffic.drop_duplicates(keep="first")
    traffic['Flow Byts/s']=traffic['Flow Byts/s'].replace([np.inf, -np.inf], np.nan)
    traffic['Flow Pkts/s']=traffic['Flow Pkts/s'].replace([np.inf, -np.inf], np.nan)
    traffic=traffic.replace([np.inf, -np.inf], np.nan)
    traffic=traffic.replace(np.nan, 0)
    columns=traffic.columns

    #Feature Reduction
    perm_imp = joblib.load('perm_imp') #loading the perm imp model
    coefficients = perm_imp.feature_importances_
    absCoefficients = abs(coefficients)
    Perm_imp_features = pd.concat((pd.DataFrame(columns,columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
    least_features=Perm_imp_features.iloc[50 :,0]

    #Dropping the least important features identified via PermutationImportance

    data=least_features.tolist()
    for i in data:
        traffic.drop(labels=[i],axis=1,inplace=True)

    DT_model = joblib.load('DT_model')
    pred = DT_model.predict(traffic)
    print('the request is ', pred)
    if pred > 0:
        pred='This is a Bening request'
    else:
        pred='This is a Mallicious  request'

    return jsonify({'pred': pred})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
