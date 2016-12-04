#!/usr/bin/env python3
# -*- coding utf-8 -*-

# pylint disable=C0103
# pylint disable=E1101
# pylint disable=C0326

import re as re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, RobustScaler, Binarizer, StandardScaler
from sklearn_pandas import DataFrameMapper, cross_val_score

train = pd.read_csv('./train.csv', delimiter=',', header=0) #Can't use column 0 Passenger ID as index, it's disappearing in the transform proc'
test = pd.read_csv('./test.csv', delimiter=',', header=0)

# print(train.head(10))


######################################
####### Feature Engineering ##########
#### Transformers and Imputers #######
######################################

####### Name Transformers ############

######## Regexp setup for name transformers
re_title = re.compile('(?<=, ).*?\.') #Get the title, for example Mrs.
re_name = re.compile('^.*?,') #Get the name, for example Futrelle,

class PP_TitleTransformer(TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        df = pd.DataFrame(X) #if using X directly it thinks it's a list ... dynalic typing ...
        return df.assign(CptTitle = df['Name'].map(lambda s:
                re_title.search(s).group()))

class PP_TitleCatTransformer(TransformerMixin):
    def __init__(self):
        self.dicoRef = {
            "Mr.": 2,
            "Mrs.": 2,
            "Miss.": 0,
            "Master.": 0,
            "Don.": 3,
            "Rev.": 3,
            "Dr.": 3,
            "Mme.": 2,
            "Ms.": 2,
            "Major.": 3,
            "Lady.": 3,
            "Sir.": 3,
            "Mlle.": 1,
            "Col.": 3,
            "Capt.": 3,
            "the Countess.": 3,
            "Jonkheer.": 3,
            "Dona.": 3
        }
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        df = pd.DataFrame(X)
        return df.assign( \
            CptTitleCat = df["CptTitle"] 
                          .map(lambda s:
                               self.dicoRef[s]))

####### Fare Transformers ############
class PP_FareImputer(TransformerMixin):
    def __init__(self):
        self.df_Fare = pd.DataFrame()
    
    def fit(self, X, y=None, **fit_params):
        self.df_Fare = X.groupby(['Pclass','Sex','CptTitle'])['Fare'].median()
        return self

    def combineFare(DF1, DF2):
        df2 = pd.DataFrame(DF2).reset_index()
        df = pd.merge(DF1,df2,on=['Pclass','Sex','CptTitle'],how='left')
        df['Fare'] = df["Fare_x"].fillna(df["Fare_y"])
        df.drop(['Fare_x','Fare_y'], axis=1, inplace=True)
        return df
    
    def transform(self, X, **transform_params):
        df = pd.DataFrame(X) #if using X directly it thinks it's a list ... dynalic typing ...
        return PP_FareImputer.combineFare(df,self.df_Fare)

class PP_FareGroupTransformer(TransformerMixin):
    def __init__(self):
        self.fare_range = np.concatenate(([0],[1],np.arange(5,100,5),[np.inf]))
    
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        df = pd.DataFrame(X)
        return df.assign( \
            CptFareGroup = pd.cut(
                df['Fare'], bins=self.fare_range,labels=False, include_lowest=True
        ))

####### Debug Transformer ###########
class DebugTransformer(TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        X.to_csv('./debug.csv')
        return X

######################################
######### Features Selection #########
######################################

mapper = DataFrameMapper([
    (['Pclass'], Binarizer()),
#   (['CptAge'], nothing),
   ('CptTitle', LabelBinarizer()),
    ('Sex', LabelBinarizer()),
   (['SibSp'], None),
   (['Parch'], None),
    (['Fare'], None),
#    ('CptDeck', LabelBinarizer()),
    (['CptTitleCat'], None),
#    ('CptName', LabelBinarizer()),
#    (['CptNameFreq'], nothing),
    # (['CptFamSize'], nothing),
    (['CptFareGroup'], None)
    # (['CptAgeGroup'], nothing),
    # (['CptFarePerson'], nothing),
#    ('CptEmbarked', LabelBinarizer()),
    # (['CptTicketFreq'], nothing),
#    (['CptAgeClass'], nothing)
    ])

######################################
######## Transformer Pipeline ########
######################################

pipe = Pipeline([
    # ("extract_deck",PP_DeckTransformer()),
    ("extract_title", PP_TitleTransformer()),
    ("extract_titlecat", PP_TitleCatTransformer()),
    # ("extract_familyName",PP_FamNameTransformer()),
    # ("extract_namefreq",PP_FamNameFreqTransformer()),
    # ("extract_famsize",PP_FamSizeTransformer()),
    ("fillNA_Fare", PP_FareImputer()),
    ("extract_faregroup",PP_FareGroupTransformer()),
    # ("extract_fareperson",PP_FarePersonTransformer()),
    # ("extract_AgeGroup",PP_AgeGroupTransformer()),
    # ("fillNA_AgeGroup",PP_AgeGroupImputer()),
    # ("fillNA_Age",PP_AgeTransformer()),
    # ("fillNA_Embarked",PP_EmbarkedImputer()),
    # ("extract_ticketfreq",PP_TicketFreqTransformer()),
    # ("extract_ageclass",PP_AgeClassTransformer()),
    ("DEBUG",DebugTransformer()),
     ("featurize", mapper),
    ("forest", RandomForestClassifier())
    ])


crossval = cross_val_score(pipe, train, train['Survived'], cv=10)
print("Cross Validation Scores are: ", crossval)
print("Mean CrossVal score is: ", crossval.mean())

