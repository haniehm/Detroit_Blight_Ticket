

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Lasso

def blight_model():
    
    original_train = pd.read_csv('train.csv', encoding='iso-8859-1', dtype={11:str,12:str,31:str})

    original_test = pd.read_csv('test.csv', dtype={11:str,12:str,31:str})

    from sklearn.model_selection import train_test_split
    train, cv = train_test_split(original_train, test_size=0.4, random_state=42)

    keep_cols = [
         u'agency_name',
         u'inspector_name',
         u'violation_street_name',
         u'state',
         u'violation_code',
         u'disposition',
         u'fine_amount',
         u'late_fee',
         u'discount_amount',
         u'clean_up_cost',
         u'judgment_amount',
    ]


    class DataFrameImputer(TransformerMixin):

        def __init__(self):
            """Impute missing values.

            Columns of dtype object are imputed with the most frequent value 
            in column.

            Columns of other types are imputed with mean of column.

            """
        def fit(self, X, y=None):

            self.fill = pd.Series([X[c].value_counts().index[0]
                if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
                index=X.columns)

            return self

        def transform(self, X, y=None):
            return X.fillna(self.fill)

    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return X[self.attribute_names]

    num_attribs = [
     'fine_amount',
     'late_fee',
     'discount_amount',
     'clean_up_cost',
     'judgment_amount'
    ]

    agency_columns = [
     'agency_name',
    ]

    inspector_columns = [
     'inspector_name',
    ]

    street_columns = [
     'violation_street_name',
    ]

    state_columns = [
     'state',
    ]

    violation_code_columns = [
     'violation_code',
    ]

    dispo_columns = [
     'disposition',
    ]

    num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attribs)),
            ('std_scaler', StandardScaler()),
        ])

    agency_pipeline = Pipeline([
            ('selector', DataFrameSelector(agency_columns)),
            ('label_binarizer', LabelBinarizer()),
        ])

    inspector_pipeline = Pipeline([
            ('selector', DataFrameSelector(inspector_columns)),
            ('label_binarizer', LabelBinarizer()),
        ])

    street_pipeline = Pipeline([
            ('selector', DataFrameSelector(street_columns)),
            ('label_binarizer', LabelBinarizer()),
        ])

    state_pipeline = Pipeline([
            ('selector', DataFrameSelector(state_columns)),
            ('imputer', DataFrameImputer()),
            ('label_binarizer', LabelBinarizer()),
        ])

    violation_code_pipeline = Pipeline([
            ('selector', DataFrameSelector(violation_code_columns)),
            ('label_binarizer', LabelBinarizer()),
        ])

    dispo_pipeline = Pipeline([
            ('selector', DataFrameSelector(dispo_columns)),
            ('label_binarizer', LabelBinarizer()),
        ])

    full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("agency_pipeline", agency_pipeline),
            ("inspector_pipeline", inspector_pipeline),            ("state_pipeline", state_pipeline),
            ("violation_code_pipeline", violation_code_pipeline),
            ("dispo_pipeline", dispo_pipeline),
        ])

    t = train[keep_cols + ['compliance']].dropna(axis=0, how='any')
    c = cv[keep_cols + ['compliance']].dropna(axis=0, how='any')

    y = t["compliance"].copy()
    X = t.drop("compliance", axis=1)
    y_cv = c["compliance"].copy()
    X_cv = c.drop("compliance", axis=1)

    X_transformed = full_pipeline.fit_transform(X)
    X_cv_transformed = full_pipeline.transform(X_cv)

    las = Lasso(alpha=0.005)
    las.fit(X_transformed,y)

    X_test = original_test[keep_cols]
    X_test_transformed = full_pipeline.transform(X_test)
    
    result = pd.Series(las.predict(X_test_transformed), index=original_test['ticket_id'].values.tolist())    
    return result
blight_model()