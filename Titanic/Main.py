import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
test_ids = test["PassengerId"]

# not very optimized for accuracy
def clean(data):
    data = data.drop(["Ticket", "PassengerId", "Name", "Cabin"], axis=1)
    
    cols = ['SibSp', 'Parch', 'Fare', 'Age']
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
        
    data.Embarked.fillna("U")
    return data

data = clean(data)
test = clean(test)

print(data.head())
sns.heatmap(data.corr(), cmap="afmhot")
plt.show();

# Label encoder
le = preprocessing.LabelEncoder()
cols = ["Sex", "Embarked"]
for col in cols:
    data[col] = le.fit_transform(data[col])
    test[col] = le.transform(test[col])
    print(le.classes_)
print(data.head())

y = data["Survived"]
X = data.drop("Survived", axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# classifier
clf = RidgeClassifier(random_state=0, max_iter=1000, alpha=100).fit(X_train, y_train)
# clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

preds = clf.predict(X_val)

# accuracy test
print(accuracy_score(y_val, preds))

submission_preds = clf.predict(test)

df = pd.DataFrame({"passengerid": test_ids.values,
                   "survived": submission_preds,
                  })

df.to_csv("submission.csv", index=False)
sub = pd.read_csv("submission.csv")
