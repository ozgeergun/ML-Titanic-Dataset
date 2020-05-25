import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import pydotplus 
import pickle
import joblib
from sklearn.metrics import confusion_matrix


def decision_tree_parameters(): #find best parameters by grid search cross val
    
    X, y = make_classification()   
    rfc = DecisionTreeClassifier()     
    param_grid = { 
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth':[10,15,20,30,50],
        'splitter':['best', 'random'],
        'criterion': ['gini', 'entropy']
    }
    
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10)
    CV_rfc.fit(X, y)
    print(CV_rfc.best_params_)
    return CV_rfc.best_params_


model=DecisionTreeClassifier(criterion = 'gini', max_depth = 10, max_features = 'auto', splitter = 'best')
             
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

    
dot_data = tree.export_graphviz(model, out_file=None, feature_names = X.columns, class_names=["died","alive"])
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("decision_tree.png")


joblib.dump(model, 'decision_tree.sav')






















