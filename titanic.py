import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import roc_auc_score, make_scorer
import matplotlib.pyplot as plt

# Import raw data and separate the labels
train    = pd.read_csv('train.csv')
test     = pd.read_csv('test.csv')
labels   = train['Survived']
train    = train.drop('Survived',1)
passengers = test['PassengerId'] 

# Subset the features to those that we want to use for now
train = train[['Pclass','Sex','Age','SibSp','Parch','Embarked']]
test  = test[['Pclass','Sex','Age','SibSp','Parch','Embarked']]

# Impute missing Ages where necessary
imp = Imputer()
train['Age'] = imp.fit_transform(train[['Age']]).ravel()
test['Age'] = imp.transform(test[['Age']]).ravel()

# Assume port of embarkation was S (most frequent) if we don't have additional info
train['Embarked'].fillna(value='S',inplace=True)
test['Embarked'].fillna(value='S',inplace=True)

# Check for null values
# print([train[col].isnull().values.any() for col in train.columns])
# print([test[col].isnull().values.any() for col in test.columns])

# One-Hot encode all categorical features
train = pd.get_dummies(train, columns=['Pclass','Sex','SibSp','Parch','Embarked'])
test = pd.get_dummies(test, columns=['Pclass','Sex','SibSp','Parch','Embarked'])
test = test.drop('Parch_9',1) # remove a stray feature

# Pick a classifier to train
# from sklearn.neighbors import KNeighborsClassifier # 0.50, 0.68
# clf = KNeighborsClassifier(n_neighbors=8)

# from sklearn.linear_model import LogisticRegression # 0.49, 0.72
# clf = LogisticRegression(C=10.0,random_state=42)

# from sklearn.tree import DecisionTreeClassifier # 0.51, 0.69
# clf = DecisionTreeClassifier(max_depth=1, random_state=42)

# from sklearn.ensemble import AdaBoostClassifier # 0.51, 0.75, 0.75
# clf = AdaBoostClassifier(n_estimators=60, random_state=42)

from sklearn.ensemble import RandomForestClassifier # 0.52, 0.70, 0.75
# min_samples_split=0.0675 gives 0.82 w/ 1 pt gap
clf = RandomForestClassifier(n_estimators=1000, max_depth=6, random_state=42, n_jobs=2)
# clf = RandomForestClassifier(n_estimators=200, min_samples_split=0.03, random_state=42, n_jobs=2)

from sklearn.model_selection import learning_curve
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, test_scores = learning_curve(clf, train, labels, cv=5, train_sizes=train_sizes)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std  = np.std(train_scores, axis=1)
test_scores_mean  = np.mean(test_scores, axis=1)
test_scores_std   = np.std(test_scores, axis=1)

plt.title("Learning Curves")
plt.plot(train_sizes, train_scores_mean, color="b", label="Training Score")
plt.plot(train_sizes, test_scores_mean, color="r", label="Cross-Val Score")
plt.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std, color="b", alpha=0.1)
plt.fill_between(train_sizes, test_scores_mean-test_scores_std, test_scores_mean+test_scores_std, color="r", alpha=0.1)
plt.legend(loc="best")
plt.xlabel("# Samples")
plt.ylabel("Score")
plt.show()

# Identify important features:
# clf.fit(train,labels)
# importances = zip(train.columns,clf.feature_importances_)
# importances = sorted(importances, key=lambda x: x[1], reverse=True)
# print(importances)

# from sklearn.ensemble import BaggingClassifier # 0.50, 0.69
# clf = BaggingClassifier(n_estimators=10, random_state=42)

# scores = cross_val_score(clf, train, labels, cv=200)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# 
# roc_scorer = make_scorer(roc_auc_score)
# rocs = cross_val_score(clf, train, labels, cv=5, scoring=roc_scorer)
# print("ROC-minus: %0.2f, var=%0.2f" % ((np.median(rocs)-rocs.std()*2), rocs.std()*2))

# df = pd.DataFrame(passengers)
# preds = clf.predict(test)
# df['Survived'] = preds
# df.to_csv('submission.csv', index=False)

