import pandas as pd
concrete = pd.read_csv('concrete_na.csv')
concrete['labels'] = pd.qcut(concrete.strength,2,[0,1])
print(concrete.head())

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

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
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

print(log_reg.score(X_train,y_train))
print(log_reg.predict())
