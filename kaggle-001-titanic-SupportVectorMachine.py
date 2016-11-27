# pylint: disable=C0103
# From the data about Titanic survivors/deaths in train.csv, predict the survivors/deaths in test.csv


import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from collections import defaultdict


# Translate string label (male female) to numerical value for ML
d = defaultdict(preprocessing.LabelEncoder)

# load dataset
df_dataset = pd.read_csv('./train.csv', delimiter=",", header=0)

# Transform label to numerical values and Remove missing values
# df_strinput = df_dataset[["Sex", "Cabin", "Embarked"]].fillna('N/A',  inplace=False)
df_strinput = df_dataset[["Sex", "Cabin"]].fillna('N/A',  inplace=False)
df_tr = df_strinput.apply(lambda x: d[x.name].fit_transform(x))

# Input for training
df_mlinput = pd.concat([
                        # df_dataset[["Survived","Pclass", "Age", "SibSp", "Parch", "Fare"]],
                        df_dataset[["Survived","Pclass", "Age", "SibSp", "Parch"]],
                        df_tr
                        ]
                        , axis=1)
df_mlinput.info()

df_mlinput.dropna(subset=["Age"], inplace=True)

df_mlinput.to_csv("check.csv",na_rep="N/A")

svm_model = LinearSVC()
svm_model.fit(df_mlinput.drop(['Survived'], axis=1).values, df_mlinput[["Survived"]].values.ravel())

######## Predict

# Load test
df_test = pd.read_csv('./test.csv', delimiter=",", header=0)
# df_strtest = df_test[["Sex","Cabin", "Embarked"]].fillna('N/A',inplace=False)
df_strtest = df_test[["Sex","Cabin"]].fillna('N/A',inplace=False)
df_tr_test = df_strtest.apply(lambda x: d[x.name].fit_transform(x))

# Input for testing
df_testinput = pd.concat([
                        # df_test[["Pclass", "Age", "SibSp", "Parch", "Fare"]],
                        df_test[["Pclass", "Age", "SibSp", "Parch"]],
                        df_tr_test
                        ]
                        , axis=1)
df_testinput.info()

# # There is a blank for a P3 class passenger for the ticket fare
# df_testinput["Fare"] = df_testinput["Fare"].fillna(
#     df_testinput.groupby("Pclass")["Fare"].transform("mean")
#     )

# Set missing age to mean according to Pclass
df_testinput["Age"] = df_testinput["Age"].fillna(
    # df_testinput.groupby("Pclass")["Age"].transform("mean")
    df_testinput.groupby("Pclass")["Age"].transform("median")
    )

out = svm_model.predict(df_testinput.values)
df_test["Survived"] = out

df_test[['PassengerId', 'Survived']].to_csv("out_LinearSVC.csv", index=False)
