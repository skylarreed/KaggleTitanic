import numpy as np
import pandas as pd
#  Viewing the chart
import matplotlib.pyplot as plt 

from sklearn.ensemble import RandomForestClassifier

import os

for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("input/train.csv")
train_data.head()

test_data = pd.read_csv("input/test.csv")
test_data.head()

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

# Create the FamilySize feature for the test data
train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"]
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"]

# Create the Age categories feature for the test data
age_bins = [0, 18, 30, 50, float('inf')]  # Define the age bins
age_labels = ['Child', 'Young Adult', 'Adult', 'Senior']  # Define labels for each bin
train_data['AgeCategory'] = pd.cut(train_data['Age'], bins=age_bins, labels=age_labels)
y = train_data["Survived"]

test_data['AgeCategory'] = pd.cut(test_data['Age'], bins=age_bins, labels=age_labels)
features = ["Pclass", "Sex", "SibSp", "Parch", "FamilySize","AgeCategory"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")