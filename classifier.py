
class preprocess():
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    import seaborn as sns
    from sklearn import preprocessing
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
    from IPython.display import display, Image
    from sklearn import tree
    import pydotplus 
    import pickle
    import joblib
    from xgboost import XGBClassifier
    from xgboost import plot_importance
    from matplotlib import pyplot
    from sklearn.metrics import confusion_matrix
    
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')
    df["embark_town"].fillna("Southampton", inplace = True)
    df["age"].fillna(np.mean(df["age"]),inplace=True)
    
    df.fillna('mean', axis=1, inplace=True) 
    df.drop("class",inplace=True,axis=1)
    print(df.head())
    for f in df.columns: 
        if df[f].dtype=='object': #categorical value encoding
            label = preprocessing.LabelEncoder() 
            label.fit(list(df[f].values)) 
            df[f] = label.transform(list(df[f].values)) #To get a column's encoding
    print(df.head())
    
    output = open('label_encoder.pkl', 'wb')
    pickle.dump(label, output)
    output.close()
    
    y = df["survived"]
    df.drop(["alive","survived","deck","who"], axis=1, inplace = True)
    X= df
    
            
    
    
    """
    pkl_file = open('label_encoder.pkl', 'rb')
    encoder = pickle.load(pkl_file) 
    pkl_file.close()
    
    """
    
