from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import numpy as np

iris = datasets.load_iris()

X = iris['data'][:,3:]
y = (iris['target'] == 2).astype(np.int)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
for train_index , test_index in split.split(X,y):
    strat_train_set = X[train_index]
    strat_train_label = y[train_index]
    strat_test_set = X[train_index]
    strat_test_label = y[train_index]
    
log_reg = LogisticRegression()
log_reg.fit(strat_train_set,strat_train_label)

print(log_reg.score(strat_test_set,strat_test_label))
print(log_reg.predict(strat_test_set[16:20]))