import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

concrete = pd.read_csv('concrete_na.csv')
concrete = concrete.fillna(concrete.mean())
concrete['labels'] = pd.qcut(concrete.strength,2,[0,1])
print(concrete.head())


num_attribs = ['cement', 'slag', 'ash', 'water', 'superplastic' , 
               'coarseagg','fineagg','age']

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

#print(num_pipeline)

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs))
    ])
#print(cat_pipeline)

full_pipeline = FeatureUnion([
    ('num_pipeline',num_pipeline),
    ('cat_pipeline',cat_pipeline)
    ])
#print(full_pipeline.fit_transform(concrete))
dataSet = full_pipeline.fit_transform(concrete)

X = dataSet[:,:-2]
y = dataSet[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rnd_clf = RandomForestClassifier()
log_clf = LogisticRegression()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[('rf',rnd_clf),('lr',log_clf),('svc',svm_clf)],
    voting='hard')
#voting_clf.fit(X_train,y_train)

for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__,accuracy_score(y_test, y_pred))