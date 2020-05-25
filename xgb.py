import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
import pickle
import joblib
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

"""
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')
df["embark_town"].fillna("Southampton", inplace = True)
df["age"].fillna(np.mean(df["age"]),inplace=True)

df.fillna('mean', axis=1, inplace=True) 
df.drop("class",inplace=True,axis=1)

for f in df.columns: 
    if df[f].dtype=='object': #categorical value encoding
        label = preprocessing.LabelEncoder() 
        label.fit(list(df[f].values)) 
        df[f] = label.transform(list(df[f].values)) #To get a column's encoding

output = open('label_encoder.pkl', 'wb')
pickle.dump(label, output)
output.close()

y = df["survived"]
df.drop(["alive","survived","deck","who"], axis=1, inplace = True)
X= df
X.rename({"pclass": "passenger class", "sibsp":"# of siblings","parch":"# of parents","fare":"passenger fare"}, axis=1, inplace=True)
"""
p = preprocess()

def xgb_params(): #find best parameters by grid search cross val
    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_informative=3,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               random_state=0,
                               shuffle=False)
    clf = XGBClassifier()
    param_grid = { 
        'loss': ['deviance', 'exponential'],
        'learning_rate': [0.01,0.025,0.05],
        'subsample':[0.75, 0.8,1],
        'n_estimators': [100,150,175,200,500],
        'max_depth':[5,6,7,8],
        
    }
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 10)
    CV_rfc.fit(X, y)
    return (CV_rfc.best_params_)

model =XGBClassifier(max_depth=7, loss = 'deviance',learning_rate= 0.01,subsample=1,n_estimators=200,n_jobs=-1)


kfold = KFold(n_splits=10, shuffle= True, random_state=42)

i=1
scores=[]
for train_index, test_index in kfold.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    print("\nFold ",i)
    i+=1
    print(conf_mat)
    print("Accuracy:",accuracy_score(y_test,y_pred))
    scores.append(accuracy_score(y_test,y_pred))
print("\nAverage accuracy:", np.mean(scores))


plot_importance(model, ax=None, height=0.1, xlim=None, ylim=None, title='Feature importance by SGB', xlabel='F score', ylabel='Features', importance_type='weight', max_num_features=5, grid=True)
pyplot.show()

joblib.dump(model, "xgboost.sav")























