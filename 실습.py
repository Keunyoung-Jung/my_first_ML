import pandas as pd
from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

concrete = pd.read_csv('concrete_na.csv')
concrete['labels'] = pd.qcut(concrete.strength,2,[0,1])
print(concrete.head())


num_attribs = ['cement', 'slag', 'ash', 'water', 'superplastic' , 
               'coarseagg','fineagg','age','strength']

cat_attribs = ['labels']

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self, X , y=None):
        return self
    
    def transform(self,X):
        return X[self.attribute_names].values
    
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy='mean')),
    ('std_scaler',StandardScaler())])

print(num_pipeline)

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs))
    ])
print(cat_pipeline)

full_pipeline = FeatureUnion([
    ('num_pipeline',num_pipeline),
    ('cat_pipeline',cat_pipeline)
    ])
print(full_pipeline.fit_transform(concrete))

dataSet = full_pipeline.fit_transform(concrete)

X = dataSet[:,:-2]
y = dataSet[:,-2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_reg = LinearRegression()
log_reg.fit(X_train,y_train)

print(log_reg.score(X_test,y_test))

X=dataSet[:,:-2]
y=dataSet[:,-1]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in split.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

print(log_reg.score(X_test, y_test))