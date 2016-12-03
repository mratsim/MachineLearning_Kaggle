
using DecisionTree
using DataFrames
using DataFramesMeta
using DataArrays
using Gadfly
using ScikitLearn: DataFrameMapper, @sk_import, Pipelines, fit!, predict
using ScikitLearn.CrossValidation
using ScikitLearnBase: @declare_hyperparameters, BaseEstimator
import ScikitLearnBase.simple_get_params

# @sk_import linear_model: LogisticRegression
@sk_import preprocessing: (LabelBinarizer, RobustScaler, Binarizer, StandardScaler)

# Note for debugging, changing samething inside a function require kernel reloading :/

train = readtable("train.csv")
test = readtable("test.csv")
head(train)

####### Exploration phase #######

# describe(train)

# Somehow adding the color visualization generate an error about int not defined in Gadfly
# Need to edit source code and replace int by Int
# plot(train, x="Sex", y="Survived", color="Survived", Geom.histogram(position=:stack), Scale.color_discrete_manual("red","green"))

# plot(train, x=:Age, y=:Survived, color=:Survived, Geom.histogram(bincount=15,position=:dodge), Scale.color_discrete_manual("orange","green"))

## Optimus Title, the transformer that gets title from the name field
type PP_TitleTransformer <: ScikitLearnBase.BaseEstimator
end

@declare_hyperparameters(PP_TitleTransformer, Symbol[]) ##Symbol is a temp mesure while waiting for new release of ScikitLearn

ScikitLearnBase.fit!(self::PP_TitleTransformer, X::DataFrame, y=nothing) = return self

function ScikitLearnBase.transform(self::PP_TitleTransformer, X::DataFrame)
    @linq X |>
    transform(CptTitle = map(s->match(r"(?<=, ).*?\.", s).match, :Name))
end

function pp_title(df::DataFrame) ##For debugging the transformation
    @linq df |>
    transform(CptTitle = map(s->match(r"(?<=, ).*?\.", s).match, :Name))
end

## Optimus Fare Imputer
## Only useful for the test data
## directly update the Fare column

type PP_FareImputer <: ScikitLearnBase.BaseEstimator
    df_Fare::DataFrame
    PP_FareImputer() = new()
end

@declare_hyperparameters(PP_FareImputer, Symbol[])

function ScikitLearnBase.fit!(self::PP_FareImputer, X::DataFrame, y=nothing)
    self.df_Fare = by(X, [:Pclass,:Sex,:CptTitleCat], df -> median(dropna(df[:Fare])))
    return self
end

function fillNA_Fare(AGI::PP_FareImputer,fl_Fare, in_Pclass, tx_Sex, in_Titlecat)
    df = AGI.df_Fare
    ifelse(
        isna(fl_Fare),
        reshape(df[
            (df[:Pclass].==in_Pclass)&
            (df[:Sex].==tx_Sex)&
            (df[:CptTitleCat].==in_Titlecat)
        ,:x1])[1],
    fl_Fare)
end

function ScikitLearnBase.transform(self::PP_FareImputer, X::DataFrame)
    result = @byrow! X begin
        :Fare = fillNA_Fare(self,:Fare,:Pclass,:Sex,:CptTitleCat)
    end
    return result
end

## Optimus Fare bucket transformer, the transformer that bins Fares
## This transformer would not have seen the light without the magical searchsortedfirst.
## (DataFrames.jl needs a cut function)
type PP_FareGroupTransformer <: ScikitLearnBase.BaseEstimator
end

@declare_hyperparameters(PP_FareGroupTransformer, Symbol[])

ScikitLearnBase.fit!(self::PP_FareGroupTransformer, X::DataFrame, y=nothing) = return self

function ScikitLearnBase.transform(self::PP_FareGroupTransformer, X::DataFrame)
    @linq X |>
    transform(CptFareGroup = map(s->
    if isna(s) return s
    else ifelse(s==0,0,
        searchsortedfirst(5.0:5.0:100.0,s)
        ) end,:Fare)
    )
end


# Magical searchsortedfirst for binning bucketing. (DataFrames.jl needs a cut function)
# However if s==0 is poisoned by NAtype
function pp_FareGroup(df::AbstractDataFrame) ## For debugging the transformation
    @linq df |> transform(CptFareGroup = map(s->
    if isna(s) return s
    else ifelse(s==0,0,
        searchsortedfirst(5.0:5.0:100.0,s)
        ) end,:Fare)
    )
end

## Optimus Deck, the transformer that gets deck from the Cabin field
type PP_DeckTransformer <: ScikitLearnBase.BaseEstimator
end

@declare_hyperparameters(PP_DeckTransformer, Symbol[])

ScikitLearnBase.fit!(self::PP_DeckTransformer, X::DataFrame, y=nothing) = return self

function ScikitLearnBase.transform(self::PP_DeckTransformer, X::DataFrame)
    @linq X |>
    transform(CptDeck = map(s->ifelse(isna(s),"Unknown",s), :Cabin)) |>
        ## Need two step otherwise Julia complains about no index method for NAtypes, pesky Julia
        transform(CptDeck = map(s->ifelse(s=="Unknown",s,s[1:1]), :CptDeck))
end

function pp_deck(df::DataFrame) ##For debugging the transformation
    @linq df |>
        transform(CptDeck = map(s->ifelse(isna(s),"Unknown",s), :Cabin)) |>
        ## Need two step otherwise Julia complains about no index method for NAtypes, pesky Julia
        transform(CptDeck = map(s->ifelse(s=="Unknown",s,s[1:1]), :CptDeck))
end

#Dictionary for socio-prof categories
#Dictionary for consistent referencing (transform input to lower case for insensitive use)
#Master. --> children 0
#Miss. Mlle --> unmarried 1
#Mr. Mrs. Ms. --> normal 2
#Honorifics --> rich people
dicoRef = Dict(
            "Mr." => 2,
            "Mrs."=> 2,
            "Miss." => 1,
            "Master." => 0,
            "Don."=> 3,
            "Rev."=>3,
            "Dr."=>3,
            "Mme."=>2,
            "Ms."=>2,
            "Major."=>3,
            "Lady."=>3,
            "Sir."=>3,
            "Mlle."=>1,
            "Col."=>3,
            "Capt."=>3,
            "the Countess."=>3,
            "Jonkheer."=>3,
            "Dona."=>3
    )

## Optimus Title Social Category, the transformer that gets the social standing
## from the CptTitle field
type PP_TitleCatTransformer <: ScikitLearnBase.BaseEstimator
end

@declare_hyperparameters(PP_TitleCatTransformer, Symbol[])

ScikitLearnBase.fit!(self::PP_TitleCatTransformer, X::DataFrame, y=nothing) = return self

function ScikitLearnBase.transform(self::PP_TitleCatTransformer, X::DataFrame)
    @linq X |>
    transform(CptTitleCat = map(s->dicoRef[s], :CptTitle))
end

function pp_titlecat(df::AbstractDataFrame) ## For debugging the transformation
    @linq df |> transform(CptTitleCat = map(s->dicoRef[s], :CptTitle))
end

## Optimus Family Name frequency, the transformer that gets the family name frequency
## potential issue, if family are split between training and test data?
## 
## Use fit! to save between training and test data
## so need to avoid double counting on fit_transform
type PP_FamNameFreqTransformer <: ScikitLearnBase.BaseEstimator
    df_FamName::DataFrame
    PP_FamNameFreqTransformer() = new()
end

@declare_hyperparameters(PP_FamNameFreqTransformer, Symbol[])

function ScikitLearnBase.fit!(self::PP_FamNameFreqTransformer, X::DataFrame, y=nothing)
    self.df_FamName = X
    return self
end

function ScikitLearnBase.transform(self::PP_FamNameFreqTransformer, X::DataFrame)
    df = ifelse(isequal(X,self.df_FamName), X, vcat(X,self.df_FamName))
    @linq  df |>
        transform(CptNameFreq = map(s->match(r"^.*?,", s).match, :Name))|>
        by(:PassengerId,CptNameFreq = length(:CptNameFreq)) |>
        join(X, on=:PassengerId)
end

function pp_namefreq(df::AbstractDataFrame) ## For debugging the transformation
    @linq df |>
        transform(CptNameFreq = map(s->match(r"^.*?,", s).match, :Name))|>
        groupby(:CptNameFreq)|>
        transform(CptNameFreq = length(:CptNameFreq)) ## TODO : is length really a count equivalent ?
end

## Optimus ticket frequency, the transformer that gets the ticket frequency
## potential issue, if family are split between training and test data?
## 
## Use fit! to save between training and test data
## so need to avoid double counting on fit_transform
type PP_TicketFreqTransformer <: ScikitLearnBase.BaseEstimator
    df_TicketFreq::DataFrame
    PP_TicketFreqTransformer() = new()
end

@declare_hyperparameters(PP_TicketFreqTransformer, Symbol[])

function ScikitLearnBase.fit!(self::PP_TicketFreqTransformer, X::DataFrame, y=nothing)
    self.df_TicketFreq = X
    return self
end

function ScikitLearnBase.transform(self::PP_TicketFreqTransformer, X::DataFrame)
    df = ifelse(isequal(X,self.df_TicketFreq), X, vcat(X,self.df_TicketFreq))
    @linq  df |>  by(:PassengerId,CptTicketFreq = length(:Ticket)) |>
        join(X, on=:PassengerId)
end

function pp_ticketfreq(df::AbstractDataFrame) ## For debugging the transformation
    @linq df |>
        groupby(:Ticket)|>
        transform(CptTicketFreq = length(:Ticket)) ## TODO : is length really a count equivalent ?
end

## Optimus Family Size
type PP_FamSizeTransformer <: ScikitLearnBase.BaseEstimator
end

@declare_hyperparameters(PP_FamSizeTransformer, Symbol[])

ScikitLearnBase.fit!(self::PP_FamSizeTransformer, X::DataFrame, y=nothing) = return self

function ScikitLearnBase.transform(self::PP_FamSizeTransformer, X::DataFrame)
    @linq X |>
    transform(CptFamSize = :SibSp + :Parch + 1 )
end

function pp_familysize(df::AbstractDataFrame) ## For debugging the transformation
    @linq df |>
    transform(CptFamSize = :SibSp + :Parch + 1 )
end

## Optimus Embarked transformer (to be replaced by a Regressor?)

type PP_EmbarkedImputer <: ScikitLearnBase.BaseEstimator
    df_Embarked::DataFrame
    PP_EmbarkedImputer() = new()
end

@declare_hyperparameters(PP_EmbarkedImputer, Symbol[])

function ScikitLearnBase.fit!(self::PP_EmbarkedImputer, X::DataFrame, y=nothing)
    self.df_Embarked = by(X, [:Pclass,:Sex,:CptTitleCat], df -> mode(dropna(df[:Embarked])))
    return self
end

function fillNA_Embarked(EI::PP_EmbarkedImputer,tx_Embarked, in_Pclass, tx_Sex, in_Titlecat)
    df = EI.df_Embarked
    ifelse(
        isna(tx_Embarked),
        reshape(df[
            (df[:Pclass].==in_Pclass)&
            (df[:Sex].==tx_Sex)&
            (df[:CptTitleCat].==in_Titlecat)
        ,:x1])[1],
    tx_Embarked)
end

function ScikitLearnBase.transform(self::PP_EmbarkedImputer, X::DataFrame)
    result = @byrow! X begin
        @newcol CptEmbarked::DataArray{String}
        :CptEmbarked = fillNA_Embarked(self,:Embarked,:Pclass,:Sex,:CptTitleCat)
    end
    return result
end

## Optimus Age bucket transformer, the transformer that bins Age
## This transformer would not have seen the light without the magical searchsortedfirst.
## (DataFrames.jl needs a cut function)
type PP_AgeGroupTransformer <: ScikitLearnBase.BaseEstimator
end

@declare_hyperparameters(PP_AgeGroupTransformer, Symbol[])

ScikitLearnBase.fit!(self::PP_AgeGroupTransformer, X::DataFrame, y=nothing) = return self

function ScikitLearnBase.transform(self::PP_AgeGroupTransformer, X::DataFrame)
    @linq X |> transform(CptAgeGroup = map(s->
    ifelse(~isna(s),searchsortedfirst(4.0:6.0:64.0,s),s),:Age))
end


# Magical searchsortedfirst for binning bucketing. (DataFrames.jl needs a cut function)
function pp_AgeGroup(df::AbstractDataFrame) ## For debugging the transformation
    @linq df |> transform(CptAgeGroup = map(s->
    ifelse(~isna(s),searchsortedfirst(4.0:6.0:64.0,s),s),:Age))
end



## Optimus Fare per person
type PP_FarePersonTransformer <: ScikitLearnBase.BaseEstimator
end

@declare_hyperparameters(PP_FarePersonTransformer, Symbol[])

ScikitLearnBase.fit!(self::PP_FarePersonTransformer, X::DataFrame, y=nothing) = return self

function ScikitLearnBase.transform(self::PP_FarePersonTransformer, X::DataFrame)
    @linq X |>
    transform(CptFarePerson = :Fare ./ :CptFamSize )
end

function pp_fareperson(df::AbstractDataFrame) ## For debugging the transformation
    @linq df |>
    transform(CptFarePerson = :Fare ./ :CptFamSize )
end

## Optimus Age group imputer transformer (to be replaced by a Regressor?)
## Note ! Can fail during Cross validation because the filter by group by returns empty :/


type PP_AgeGroupImputer <: ScikitLearnBase.BaseEstimator
    df_AgeGroup::DataFrame
    PP_AgeGroupImputer() = new()
end

@declare_hyperparameters(PP_AgeGroupImputer, Symbol[])

function ScikitLearnBase.fit!(self::PP_AgeGroupImputer, X::DataFrame, y=nothing)
    self.df_AgeGroup = by(X, [:Pclass,:Sex,:CptTitleCat], df -> mode(dropna(df[:CptAgeGroup])))
    return self
end

function fillNA_AgeCat(AGI::PP_AgeGroupImputer,in_AgeGroup, in_Pclass, tx_Sex, in_Titlecat)
    df = AGI.df_AgeGroup
    ifelse(
        isna(in_AgeGroup),
        reshape(df[
            (df[:Pclass].==in_Pclass)&
            (df[:Sex].==tx_Sex)&
            (df[:CptTitleCat].==in_Titlecat)
        ,:x1])[1],
    in_AgeGroup)
end

function ScikitLearnBase.transform(self::PP_AgeGroupImputer, X::DataFrame)
    result = @byrow! X begin
        :CptAgeGroup = fillNA_AgeCat(self,:CptAgeGroup,:Pclass,:Sex,:CptTitleCat)
    end
    return result
end

## Optimus Age imputer transformer (to be replaced by a Regressor?)
## Note ! Can fail during Cross validation because the filter by group by returns empy :/
## For missing values, only infer from non-missing field
## to avoid accumulate approx from actual data
## ==> Don't use CptAgeGroup

type PP_AgeTransformer <: ScikitLearnBase.BaseEstimator
    df_Age::DataFrame
    PP_AgeTransformer() = new()
end

@declare_hyperparameters(PP_AgeTransformer, Symbol[])

function ScikitLearnBase.fit!(self::PP_AgeTransformer, X::DataFrame, y=nothing)
    self.df_Age = by(X, [:Pclass,:Sex,:CptTitleCat], df -> median(dropna(df[:Age])))
    return self
end

function fillNA_Age(AGI::PP_AgeTransformer,in_Age, in_Pclass, tx_Sex, in_Titlecat)
    df = AGI.df_Age
    ifelse(
        isna(in_Age),
        reshape(df[
            (df[:Pclass].==in_Pclass)&
            (df[:Sex].==tx_Sex)&
            (df[:CptTitleCat].==in_Titlecat)
        ,:x1])[1],
    in_Age)
end

function ScikitLearnBase.transform(self::PP_AgeTransformer, X::DataFrame)
    result = @byrow! X begin
        @newcol CptAge::DataArray{Float64}
        :CptAge = fillNA_Age(self,:Age,:Pclass,:Sex,:CptTitleCat)
    end
    return result
end




## For testing only. cannot automatically test data in a pipeline
## because groupby will be different between train and test
function pp_MissingAge(df::AbstractDataFrame)
    @linq df |>
    groupby([:Pclass,:Sex,:CptTitle]) |>
          transform(CptAge = ifelse(isna(:Age),median(dropna(:Age)),:Age))
end

# Check Pipeline before NA prediction steps
Z = @linq train |> pp_title |> pp_titlecat |> pp_deck |>
pp_namefreq |> pp_familysize |> pp_FareGroup |> pp_fareperson |> pp_AgeGroup |>
pp_ticketfreq |> pp_MissingAge

## Optimus Debug
type InspectTransformer <: ScikitLearnBase.BaseEstimator
end

@declare_hyperparameters(InspectTransformer, Symbol[])

ScikitLearnBase.fit!(self::InspectTransformer, X::DataFrame, y=nothing) = return self

function ScikitLearnBase.transform(self::InspectTransformer, X::DataFrame)
    writetable("debug.csv",X)
    return X
end

# Create model
mapper = DataFrameMapper([
    ([:Pclass], Binarizer()),
#    ([:CptAge], nothing),
    (:CptTitle, LabelBinarizer()),
    (:Sex, LabelBinarizer()),
    ([:SibSp], nothing),
    ([:Parch], nothing),
#    ([:Fare], nothing),
    (:CptDeck, LabelBinarizer()),
    ([:CptTitleCat], nothing),
#    ([:CptNameFreq], nothing),
#    ([:CptFamSize], nothing),
    ([:CptFareGroup], nothing),
    ([:CptAgeGroup], nothing),
    ([:CptFarePerson], nothing),
    (:CptEmbarked, LabelBinarizer()),
    ([:CptTicketFreq], nothing)
    ]);

pipe = Pipelines.Pipeline([
    ("extract_deck",PP_DeckTransformer()),
    ("extract_title", PP_TitleTransformer()),
    ("extract_titlecat",PP_TitleCatTransformer()),
    ("extract_namefreq",PP_FamNameFreqTransformer()),
    ("extract_famsize",PP_FamSizeTransformer()),
    ("fillNA_Fare",PP_FareImputer()),
    ("extract_faregroup",PP_FareGroupTransformer()),
    ("extract_fareperson",PP_FarePersonTransformer()),
    ("extract_AgeGroup",PP_AgeGroupTransformer()),
    ("fillNA_AgeGroup",PP_AgeGroupImputer()),
    ("fillNA_Age",PP_AgeTransformer()),
    ("fillNA_Embarked",PP_EmbarkedImputer()),
    ("extract_ticketfreq",PP_TicketFreqTransformer()),
    ("DEBUG",InspectTransformer()),
     ("featurize", mapper),
    ("forest", RandomForestClassifier(ntrees=100,nsubfeatures=7)) #nsubfeatures, partialsampling, maxdepth
    ])

X_train = train
Y_train = convert(Array, train[:Survived])


#Cross Validation - check model accuracy
crossval = round(cross_val_score(pipe, X_train, Y_train, cv =10), 2)
print("\n",crossval,"\n")
print(mean(crossval))

model = fit!(pipe, X_train, Y_train)
# print(model)

result=DataFrame()
result[:PassengerId] = test[:PassengerId]
result[:Survived] = @data predict(model,test)

result

writetable("julia-magicalforests.csv",result)
