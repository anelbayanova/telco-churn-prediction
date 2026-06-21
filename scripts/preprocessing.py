#preprocessing.py — Кастомные трансформеры для Pipeline


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FixTotalCharges(BaseEstimator, TransformerMixin):
    
    #Превращает столбец TotalCharges из текста в число.
    

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out.loc[:, 'TotalCharges'] = pd.to_numeric(X_out['TotalCharges'], errors='coerce')
        return X_out

    def get_feature_names_out(self, input_features=None):
        return input_features


class AddFeatures(BaseEstimator, TransformerMixin):
    """
    Создаёт 3 новых признака внутри Pipeline:
    - tenure_bucket
    - charges_per_tenure
    - n_services
    """

    SERVICE_COLS = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    NOT_ACTIVE = {'No', 'No internet service', 'No phone service'}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # 1. Готовим данные во временных переменных (не трогая исходный DataFrame)
        tenure_ser = pd.to_numeric(X_out['tenure'], errors='coerce').fillna(0)
        
        bucket_ser = pd.cut(
            tenure_ser,
            bins=[-1, 12, 24, 48, np.inf],
            labels=['0-12', '13-24', '25-48', '49+']
        ).astype(str)

        total_ser = pd.to_numeric(X_out['TotalCharges'], errors='coerce').fillna(0)
        charges_per_tenure_ser = total_ser / tenure_ser.clip(lower=1)

        # Вычисляем n_services через векторные операции
        n_services_ser = pd.Series(0, index=X_out.index)
        for col in self.SERVICE_COLS:
            if col in X_out.columns:
                n_services_ser += (~X_out[col].isin(self.NOT_ACTIVE)).astype(int)

        # 2. Безопасно добавляем новые колонки за один шаг через .assign()
        # Этот метод для Copy-on-Write, создает новый DataFrame в памяти
        X_out = X_out.assign(
            tenure_bucket=bucket_ser,
            charges_per_tenure=charges_per_tenure_ser,
            n_services=n_services_ser
        )

        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        
        features = list(input_features)
        new_cols = ['tenure_bucket', 'charges_per_tenure', 'n_services']
        for col in new_cols:
            if col not in features:
                features.append(col)
                
        return np.array(features, dtype=object)