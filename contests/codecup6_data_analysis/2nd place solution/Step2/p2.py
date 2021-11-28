import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


df = pd.read_csv('travel_insurance/train.csv')

train, validation = train_test_split(df, test_size=.2, stratify=df.TravelInsurance, random_state=0)


def preprocess(df):
    return np.stack([
        df.Age.values,
        (df['Employment Type'] == 'Private Sector/Self Employed').values,
        (df.GraduateOrNot == 'Yes').values,
        df.AnnualIncome.values,
        df.FamilyMembers.values,
        df.ChronicDiseases.values,
        (df.FrequentFlyer == 'Yes').values,
        (df.EverTravelledAbroad=='Yes').values,
    ], axis=1).astype('float32')


def fit(model, data):
    return model.fit(data.drop(columns='TravelInsurance'), data['TravelInsurance']=='Yes')


def auc(model, data):
    return roc_auc_score(
        y_score=model.predict_proba(data.drop(columns='TravelInsurance'))[:,1],
        y_true=data.TravelInsurance=='Yes'
    )


    
model = Pipeline([
    ('preprocess', FunctionTransformer(preprocess)),
    ('classifier', RandomForestClassifier(random_state=0)),
])

fit(model, train)

print(auc(model, validation))


test = pd.read_csv('travel_insurance/test.csv')

result = test[['Customer Id']].assign(prediction=model.predict_proba(test)[:,1])
result.to_csv('output.csv', header=True, index=False)