# pylint: disable=C0103
# From the data about Titanic survivors/deaths in train.csv, predict the survivors/deaths in test.csv


# Solution using Random Forests. Accuracy on 50% of Kaggle published data : 0.69
# Issue : Scikit doesn't manage labeled data
# Issue : Scikit does not manage empty values
# Labeled data converted to Integer
# Empty value on Age converted to arbitrary data : Age=100, Fare = mean of Fares payed by passenger of the same class
# Improvement ==> removed empty from the training data. But what to do for the Test data ?


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from collections import defaultdict


# Translate string label (male female) to numerical value for ML
d = defaultdict(preprocessing.LabelEncoder)

# load dataset
df_dataset = pd.read_csv('./train.csv', delimiter=",", header=0)

# Transform label to numerical values and Remove missing values
df_strinput = df_dataset[["Sex", "Cabin", "Embarked"]].fillna('N/A',  inplace=False)
df_tr = df_strinput.apply(lambda x: d[x.name].fit_transform(x))

# Input for testing
df_mlinput = pd.concat([
                        df_dataset[["Pclass", "Age", "SibSp", "Parch", "Fare"]],
                        df_tr
                        ]
                        , axis=1)
df_mlinput.info()

# Arbitrary set age to 100 for missing age need to do the same for data to classify
df_mlinput["Age"] = df_mlinput["Age"].fillna(100)

rf_model = RandomForestClassifier(n_jobs=-1, oob_score=True, n_estimators=100)
rf_model.fit(df_mlinput.values, df_dataset[["Survived"]].values.ravel())


out = rf_model.predict(df_mlinput.values)
np.savetxt("out.csv", out, delimiter=",")



# Load test
df_test = pd.read_csv('./test.csv', delimiter=",", header=0)
df_strtest = df_test[["Sex","Cabin", "Embarked"]].fillna('N/A',inplace=False)
df_tr_test = df_strtest.apply(lambda x: d[x.name].fit_transform(x))

# Input for training
df_testinput = pd.concat([
                        df_test[["Pclass", "Age", "SibSp", "Parch", "Fare"]],
                        df_tr_test
                        ]
                        , axis=1)
df_testinput.info()

# Arbitrary set age to 100 for missing age need to do the same for data to classify
df_testinput["Age"] = df_testinput["Age"].fillna(100)

# There is a blank for a P3 class passenger for the ticket fare
df_testinput["Fare"] = df_testinput["Fare"].fillna(
    df_testinput.groupby("Pclass")["Fare"].transform("mean")
    )


out = rf_model.predict(df_testinput.values)
np.savetxt("out.csv", out, delimiter=",")