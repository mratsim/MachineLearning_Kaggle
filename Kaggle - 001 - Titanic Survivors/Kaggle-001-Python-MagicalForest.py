#!/usr/bin/env python3
# -*- coding utf-8 -*-

# pylint disable=C0103
# pylint disable=E1101, C0326

import re as re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, RobustScaler, Binarizer, StandardScaler
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.model_selection import GridSearchCV
from time import time

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

class PP_TitleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        df = pd.DataFrame(X) #if using X directly it thinks it's a list ... dynalic typing ...
        return df.assign(CptTitle = df['Name'].map(lambda s:
                re_title.search(s).group()))

class PP_TitleCatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # 0, young, 1 unmarried, 2 normal, 3 high status
        self.dicoRef = {
            "Mr.": 2,
            "Mrs.": 2,
            "Miss.": 1,
            "Master.": 0,
            "Don.": 3,
            "Rev.": 3,
            "Dr.": 3,
            "Mme.": 2,
            "Ms.": 1,
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
        return df.assign(
            CptTitleCat = df["CptTitle"] 
                          .map(lambda s:
                               self.dicoRef[s]))

class PP_FamNameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        df = pd.DataFrame(X) #if using X directly it thinks it's a list ... dynalic typing ...
        return df.assign(CptName = df['Name'].map(lambda s:
                re_name.search(s).group()))

class PP_FamNameFreqTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.df_FamName = pd.DataFrame()

    def fit(self, X, y=None, **fit_params):
        self.df_FamName = pd.DataFrame(X)
        return self

    def transform(self, X):
        dfX = pd.DataFrame(X)
        df = dfX
        if not df.equals(self.df_FamName):
            df = pd.concat([self.df_FamName, df])
        return dfX.assign(
            CptNameFreq = df.groupby('CptName')['CptName']
                            .transform('count')
        ) #TODO confirm if indexing match between df and dfX

####### Cabin Transformers ##########
class PP_DeckTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        df = pd.DataFrame(X) #if using X directly it thinks it's a list ... dynalic typing ...
        return df.assign(
            CptDeck = np.where(
                pd.isnull(df['Cabin']),
                'U',
                df['Cabin'].map(lambda s: str(s)[0])
                )
        )


####### Family Transformers ##########
class PP_FamSizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        df = pd.DataFrame(X) #if using X directly it thinks it's a list ... dynalic typing ...
        return df.assign(
            CptFamSize = df['SibSp'] + df['Parch'] + 1
            )

####### Fare Transformers ############
class PP_FareImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.df_Fare = pd.DataFrame()
    
    def fit(self, X, y=None, **fit_params):
        self.df_Fare = X.groupby(['Pclass','Sex','CptTitle'])['Fare'] \
                        .median() \
                        .reset_index()
        return self

    def combineFare(DF1, DF2):
        df2 = pd.DataFrame(DF2)
        df = pd.merge(DF1,df2,on=['Pclass','Sex','CptTitle'],how='left')
        df['Fare'] = df["Fare_x"].fillna(df["Fare_y"])
        df.drop(['Fare_x','Fare_y'], axis=1, inplace=True)
        return df
    
    def transform(self, X, **transform_params):
        df = pd.DataFrame(X) #if using X directly it thinks it's a list ... dynalic typing ...
        return PP_FareImputer.combineFare(df,self.df_Fare)

class PP_FareGroupTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fare_range = np.concatenate(
            ([0],[1],np.arange(5,100,5),[np.inf])
            )
    
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        df = pd.DataFrame(X)
        return df.assign(
            CptFareGroup = pd.cut(
                df['Fare'], bins=self.fare_range,labels=False, include_lowest=True
        ))

class PP_FarePersonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        df = pd.DataFrame(X)
        return df.assign(
            CptFarePerson = df['Fare'] / df['CptFamSize']
        )

####### Ticket Transformers ###########
class PP_TicketFreqTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.df_Ticket = pd.DataFrame()

    def fit(self, X, y=None, **fit_params):
        self.df_Ticket = pd.DataFrame(X)
        return self

    def transform(self, X):
        dfX = pd.DataFrame(X)
        df = dfX
        if not df.equals(self.df_Ticket):
            df = pd.concat([self.df_Ticket, df]).reset_index()
        return dfX.assign(
            CptTicketFreq = df.groupby('Ticket')['Ticket']
                            .transform('count')
        ) #TODO confirm if indexing match between df and dfX

####### Embarked Imputer ############
class PP_EmbarkedImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.df_Embarked = pd.DataFrame()
    
    def fit(self, X, y=None, **fit_params):
        self.df_Embarked = X.groupby(['Pclass','Sex','CptTitle'])['Embarked'] \
                        .agg(lambda x:x.value_counts().index[0]) \
                        .reset_index() #value_counts avoid having multiple return values compared to mode
        return self

    def combineEmbarked(DF1, DF2):
        df2 = pd.DataFrame(DF2)
        df = pd.merge(DF1,df2,on=['Pclass','Sex','CptTitle'],how='left') #might also create level_3 column for empty join on CptTitle
        df['Embarked'] = df["Embarked_x"].fillna(df["Embarked_y"])
        #df.drop(['Embarked_x','Embarked_y'], axis=1, inplace=True)
        return df
    
    def transform(self, X, **transform_params):
        df = pd.DataFrame(X) #if using X directly it thinks it's a list ... dynalic typing ...
        return PP_EmbarkedImputer.combineEmbarked(df,self.df_Embarked)

####### Age Transformers ############
class PP_AgeGroupTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_range = np.concatenate(
            ([0],np.arange(4,64,6),[np.inf])
            )
    
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        df = pd.DataFrame(X)
        return df.assign(
            CptAgeGroup = pd.cut( #Normally cut automatically bins NA in NA
                df['Age'], bins=self.age_range,labels=False, include_lowest=True
        ))

class PP_AgeGroupImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.df_AgeGroup = pd.DataFrame()
    
    def fit(self, X, y=None, **fit_params):
        self.df_AgeGroup = X.groupby(['Pclass','Sex','CptTitle'])['CptAgeGroup'] \
                        .agg(lambda x:x.value_counts().index[0]) \
                        .reset_index()
        return self

    def combineAgeGroup(DF1, DF2):
        df2 = pd.DataFrame(DF2)
        df = pd.merge(DF1,df2,on=['Pclass','Sex','CptTitle'],how='left') #might also create level_3 column for empty join on CptTitle
        df['CptAgeGroup'] = df["CptAgeGroup_x"].fillna(df["CptAgeGroup_y"])
        #df.drop(['CptAgeGroup_x','CptAgeGroup_y'], axis=1, inplace=True)
        df.to_csv('./df_after_merge.csv')
        return df
    
    def transform(self, X, **transform_params):
        df = pd.DataFrame(X) #if using X directly it thinks it's a list ... dynamic typing ...
        return PP_AgeGroupImputer.combineAgeGroup(df,self.df_AgeGroup)

class PP_AgeImputer(BaseEstimator, TransformerMixin):
    #depends from AgeGroup and CptTitleCat, so might introduce bias
    def __init__(self):
        self.df_Age = pd.DataFrame()
    
    def fit(self, X, y=None, **fit_params):
        self.df_Age = X.groupby(['Pclass','Sex','CptTitleCat'])['Age'] \
                        .median() \
                        .reset_index()
        return self

    def combineAge(DF1, DF2):
        df2 = pd.DataFrame(DF2)
        df = pd.merge(DF1,df2,on=['Pclass','Sex','CptTitleCat'],how='left') #might also create level_3 column for empty join on CptTitleCat
        df['Age'] = df["Age_x"].fillna(df["Age_y"])
        #df.drop(['Age_x','Age_y'], axis=1, inplace=True)
        return df
    
    def transform(self, X, **transform_params):
        df = pd.DataFrame(X) #if using X directly it thinks it's a list ... dynamic typing ...
        return PP_AgeImputer.combineAge(df,self.df_Age)

class PP_AgeClassTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        df = pd.DataFrame(X)
        return df.assign(
            CptAgeClass = df['Age'] * df['Pclass']
        )

####### Debug Transformer ###########
class DebugTransformer(BaseEstimator, TransformerMixin):
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

#Needed otherwise the Sex column gives only 1 feature instead of 2 and I can't check the top features'
class LabelBinForBinaryVal(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


mapper = DataFrameMapper([
    (['Pclass'], Binarizer()),
  (['Age'], None),
   ('CptTitle', LabelBinarizer()),
    ('Sex', LabelBinForBinaryVal()),
    # ('Ticket', LabelBinarizer()),
   (['SibSp'], None),
   (['Parch'], None),
    (['Fare'], None),
   ('CptDeck', LabelBinarizer()),
    (['CptTitleCat'], None),
#    ('CptName', LabelBinarizer()),
#    (['CptNameFreq'], None),
    (['CptFamSize'], None),
    (['CptFareGroup'], None),
    (['CptAgeGroup'], None),
    (['CptFarePerson'], None),
   ('Embarked', LabelBinarizer()),
    (['CptTicketFreq'], None),
   (['CptAgeClass'], None)
    ])

######################################
######## Transformer Pipeline ########
######################################

pipe = Pipeline([
    ("extract_familyName",PP_FamNameTransformer()),
    # ("extract_namefreq",PP_FamNameFreqTransformer()), #Buggy reindex issue on test data (but not on CrossVal)
    ("extract_deck",PP_DeckTransformer()),
    ("extract_ticketfreq",PP_TicketFreqTransformer()), #Note: for some reason ther eis a reindex error if put far down the pipeline
    ("extract_title", PP_TitleTransformer()),
    ("extract_titlecat", PP_TitleCatTransformer()),
    ("extract_famsize", PP_FamSizeTransformer()),
    ("fillNA_Fare", PP_FareImputer()),
    ("extract_faregroup", PP_FareGroupTransformer()),
    ("extract_fareperson", PP_FarePersonTransformer()),
    ("fillNA_Age",PP_AgeImputer()),
    ("extract_AgeGroup",PP_AgeGroupTransformer()),
    #("fillNA_AgeGroup",PP_AgeGroupImputer()), #Useless if we fill age before
    ("fillNA_Embarked",PP_EmbarkedImputer()),
    ("extract_ageclass",PP_AgeClassTransformer()),
    ("DEBUG",DebugTransformer()),
     ("featurize", mapper),
    ("forest", RandomForestClassifier(
        n_estimators=15, 
        max_features="sqrt", 
        min_samples_split=18,
        criterion= 'gini',
        n_jobs=-1))
    ])

################ Training ################################

X_train = train
y_train = train['Survived']

# ##### Cross Validation
crossval = cross_val_score(pipe, X_train, y_train, cv=10)
print("Cross Validation Scores are: ", crossval.round(3))
print("Mean CrossVal score is: ", round(crossval.mean(),3))

# ##### GridSearch Tune hyperparameters #######
# param_grid = { "forest__n_estimators"      : [1000], #[10,30,100,200, 300,1000],
#            "forest__criterion"         : ["gini", "entropy"],
#            "forest__max_features"      : [3, 5,"auto","sqrt","log2"],
#            "forest__max_depth"         : [None, 10, 20],
#            "forest__min_samples_split" : [2, 4],
#         #    "forest__min_samples_leaf": [1, 3, 10, 50],
#         #    "forest__bootstrap": [True, False],
#         #    "forest__oob_score": [True,False]
#            }

# # run grid search
# grid_search = GridSearchCV(pipe, param_grid=param_grid)
# start = time()
# grid_search.fit(X_train, y_train)

# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#       % (time() - start, len(grid_search.cv_results_['params'])))
# report(grid_search.cv_results_)


##### Fit ######
pipe.fit(X_train, y_train)

####### Get top features and noise #######

dummy, model = pipe.steps[-1]

feature_list = []
for feature in pipe.named_steps['featurize'].features:
    try:
        for feature_value in feature[1].classes_:
            feature_list.append(feature[0]+'_'+feature_value)
    except:
        feature_list.append(feature[0])


# # Can't do the following with Binarizers and Label Binarizers due to change in columns
top_features = pd.DataFrame({'feature':feature_list,'importance':np.round(model.feature_importances_,3)})
top_features = top_features.sort_values('importance',ascending=False).set_index('feature')
# print(top_features)
top_features.to_csv('top_features.csv')
top_features.plot.bar()

predictions = pipe.predict(test)
result = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
    })

result.to_csv('python-magicalforest.csv', index=False)