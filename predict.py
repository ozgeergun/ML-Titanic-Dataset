import pickle
import joblib
import pandas as pd

inp = int(input("for using decision tree press 1, xgboost press 2\n"))

pkl_file = open('label_encoder.pkl', 'rb')
encoder = pickle.load(pkl_file) 
pkl_file.close()


if(inp == 1):
    model = joblib.load("decision_tree.sav")
    
else:
    model = joblib.load("xgboost.sav")

df = pd.read_csv("passengers.csv")

for f in (df.columns): 
    if df[f].dtype == "object": 
        df[f] = encoder.fit_transform(df[f]) 
        
prediction = model.predict(df)
print(prediction)