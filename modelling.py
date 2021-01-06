import pandas as pd
import numpy as np
import seaborn as sns
import itertools
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from eli5.sklearn import PermutationImportance
import joblib
from sklearn.metrics import accuracy_score

#sample datatsets read
df1=pd.read_csv('02-14-2018.csv')
df2=pd.read_csv('02-15-2018.csv')

#sampling the large datatsets
Strat_df1=df1.groupby('Label', group_keys=False).apply(lambda x: x.sample(10000)) # Applying Stratified Sampling on the datasets
Strat_df2=df2.groupby('Label', group_keys=False).apply(lambda x: x.sample(10000))

#concating the stratified samples
final_dataset=pd.concat([Strat_df1,Strat_df2])
del Strat_df1,Strat_df2

#replacing the text features to numerical values
final_dataset.replace(to_replace=['Infilteration','Bot','DoS attacks-GoldenEye','DoS attacks-Hulk','DoS attacks-Slowloris','SSH-Bruteforce','FTP-BruteForce','DDOS attack-HOIC','DoS attacks-SlowHTTPTest','DDOS attack-LOIC-UDP','Brute Force -Web','Brute Force -XSS','SQL Injection'],value=0,inplace=True) #encoding the Anamalous and Normal values as 0 and 1 to visualize
final_dataset.replace(to_replace=['Benign'],value=1,inplace=True)

#Converting the feature datatypes
final_dataset['Timestamp'] = pd.to_datetime(final_dataset['Timestamp']).astype(np.int64)

columns=final_dataset.columns

for i in columns:
    final_dataset[i]=final_dataset[i].astype(float)

#dropping the common zero values
final_dataset.drop(['Bwd PSH Flags'],axis=1,inplace=True)
final_dataset.drop(['Bwd URG Flags'],axis=1,inplace=True)
final_dataset.drop(['Fwd Byts/b Avg'],axis=1,inplace=True)
final_dataset.drop(['Fwd Pkts/b Avg'],axis=1,inplace=True)
final_dataset.drop(['Fwd Blk Rate Avg'],axis=1,inplace=True)
final_dataset.drop(['Bwd Byts/b Avg'],axis=1,inplace=True)
final_dataset.drop(['Bwd Pkts/b Avg'],axis=1,inplace=True)
final_dataset.drop(['Bwd Blk Rate Avg'],axis=1,inplace=True)

#Data Preprocessing
final_dataset =  final_dataset.drop_duplicates(keep="first")

final_dataset['Flow Byts/s']=final_dataset['Flow Byts/s'].replace([np.inf, -np.inf], np.nan)
final_dataset['Flow Pkts/s']=final_dataset['Flow Pkts/s'].replace([np.inf, -np.inf], np.nan)
final_dataset=final_dataset.replace([np.inf, -np.inf], np.nan)
final_dataset=final_dataset.replace(np.nan, 0)

#Feture Engineering
P_Y = final_dataset['Label'] #Splitting the Xi and Yi to apply the Permutation Importance model and identify the important features to further apply model
P_X= final_dataset.drop(['Label'],axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(P_X, P_Y, test_size=0.2)

#applying the permutation Importance Feature selection technique
PX_train=np.asarray(X_train)
PX_test=np.asarray(X_test)
PY_train=np.asarray(Y_train)
PY_test=np.asarray(Y_test)

sel = SelectFromModel(PermutationImportance(RandomForestClassifier(), cv=5)).fit(PX_train, PY_train)
X_train2 = sel.transform(PX_train)
X_test2 = sel.transform(PX_test)

model = RandomForestClassifier()
model.fit(X_train2,Y_train)
columns = X_train.columns
joblib.dump(model, 'perm_imp') #dumping the trained model
coefficients = model.feature_importances_
absCoefficients = abs(coefficients)
Perm_imp = pd.concat((pd.DataFrame(columns,columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
least_features=Perm_imp.iloc[50 :,0] #identify the least importance features in the dataset

#dropping the least important features
data=least_features.tolist()
for i in data:
    final_dataset.drop(labels=[i],axis=1,inplace=True)

print(final_dataset.shape)

#Applying the DT  Model
y = final_dataset['Label']
X = final_dataset.drop(['Label'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 101)

from sklearn.tree import DecisionTreeClassifier
DT_clf = DecisionTreeClassifier(random_state=0)

DT_clf.fit(X_train, y_train)
joblib.dump(DT_clf, 'DT_model') #dump the DT model
