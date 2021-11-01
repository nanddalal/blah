import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

def preprocess_raw_data(df):
    lookup = {
        'unk': 0,
        'male': 1,
        'female': 2,
    }
    df['Sex'] = df['Sex'].apply(lambda x: lookup.get(x, 0))
    #print('before dropping nan age', len(df))
    #df = df[~df['Age'].isna()]
    #print('after dropping nan age', len(df))
    df['Age'] = df['Age'].apply(lambda x: 30 if pd.isna(x) else x)
    return df

def corr1(df):
    plt.figure(figsize=(20,12))
    sns.heatmap(df.corr().abs(), annot=True)

def corr2(df):
    cols = [
        'Survived',
        'Pclass',
        'Sex',
        'Age',
        'Fare',
    ]
    sns.pairplot(df[cols],hue="Survived")

def partition_dev_data(df_dev):
    df_dev = df_dev.sample(frac=1.0)
    nsamples = len(df_dev)
    ntrain = int(nsamples*0.8)
    df_train = df_dev[:ntrain]
    df_val = df_dev[ntrain:]
    return df_train, df_val

class Model():

    def __init__(self):
        self.feature_cols = [
            'Pclass',
            'Sex',
            'Age',
            'Fare',
            'SibSp',
            'Parch',
        ]
        self.gt_col = ['Survived']

        self.fitted = False

    def train(self, df):

        if self.fitted:
            raise

        self.preprocessor = StandardScaler()
        # self.pca = PCA()
        # self.model = LogisticRegression()
        # self.model = RandomForestClassifier()
        self.model = RandomForestClassifier(class_weight='balanced')

        x = df[self.feature_cols]
        y = df[self.gt_col]

        self.preprocessor.fit(x)
        x = self.preprocessor.transform(x)
        # self.pca.fit(x)
        # x = self.pca.transform(x)
        self.model.fit(x, y)

        for name, importance in zip(self.feature_cols, self.model.feature_importances_):
            print(name, importance)

        self.fitted = True

    def predict(self, df):

        if not self.fitted:
            raise

        x = df[self.feature_cols]
        y = df[self.gt_col]

        x = self.preprocessor.transform(x)
        # x = self.pca.transform(x)
        y_prob = self.model.predict_proba(x)

        assert y_prob.ndim == 2
        assert y_prob.shape[1] == 2
        y_prob = y_prob[:,1]

        df['true'] = y
        df['prob'] = y_prob

        return df

def evaluate_model(df, plot=False):
    n = len(df)
    prev = df['true'].mean()
    acc = accuracy_score(df['true'], df['prob']>0.5)
    if prev == 0 or prev == 1:
        roc_auc = 0
        ap = 0
    else:
        roc_auc = roc_auc_score(df['true'], df['prob'])
        ap = average_precision_score(df['true'], df['prob'])

    if plot:

        plt.figure(figsize=(20,12))

        fpr, tpr, _ = roc_curve(df['true'], df['prob'])
        sens = tpr
        spec = 1-fpr
        plt.subplot(121)
        plt.plot(spec, sens)
        plt.grid()

        prec, rec , _ = precision_recall_curve(df['true'], df['prob'])
        plt.subplot(122)
        plt.plot(prec, rec)
        plt.grid()

    return {
        'n': n,
        'prev': prev,
        'acc': acc,
        'roc_auc': roc_auc,
        'ap': ap,
    }

def run_pipeline(df_train, df_val):
    model = Model()
    model.train(df_train)

    df_train = model.predict(df_train)
    df_val = model.predict(df_val)

    print(evaluate_model(df_train))
    print(evaluate_model(df_val, plot=True))

    #cols = [
    #    'Pclass',
    #]
    #for k, v in df_val.groupby(cols):
    #    print(k)
    #    print(evaluate_model(v))
