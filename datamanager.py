from data.config import data_config
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataManager:
    
    def __init__(self, data_config, ignore_lags=False, target_col=None):
        self.data_config = data_config
        self._x_cols, self._y_cols = data_config["input_param_names"], data_config["target_param_names"]
        if target_col:
            self._y_cols = [target_col]
        self.df = pd.read_csv(data_config["file_path"])[self._x_cols + self._y_cols + ["profile_id"]]
        self.ignore_lags = ignore_lags
        self.add_feats_names = None
        
    
    def create_add_features(self):
        add_feats = {
            "i_s": lambda x: np.sqrt(x["i_d"] ** 2 + x["i_q"] ** 2),     # Current vector norm
            "u_s": lambda x: np.sqrt(x["u_d"] ** 2 + x["u_q"] ** 2),     # Voltage vector norm
            "S_el": lambda x: x["i_s"] * x["u_s"],                       # Apparent power
            "P_el": lambda x: x["i_d"] * x["u_d"] + x["i_q"] * x["u_q"], # Effective power
            }
        self.add_feats_names = list(add_feats.keys())
        return self.df.assign(**add_feats)
    
    def featurize(self):
        self.df = self.create_add_features()
        self.df = self.create_rolling_mean_features(fill_method=0)
        # self.df = self.create_ewma_features(fill_method=0)
        self.df = self.create_rolling_std_features(fill_method=0)
        if not self.ignore_lags:
            self.df = self.create_lag_features(fill_method=0)

    def create_lag_features(self, fill_method):
        df = self.df.copy()
        lags = data_config["lags"]
        for feature_name in self._x_cols + self.add_feats_names + self._y_cols:
            for lag in lags:
                df[f"{feature_name}_lag_{lag}"] = df[f"{feature_name}"].shift(lag)
        return df.fillna(fill_method)
    
    def create_rolling_mean_features(self, fill_method):
        df = self.df.copy()
        lags = data_config["lags"]
        for feature_name in self._x_cols:
            # df[f"{feature_name}_p10"] = df[feature_name].quantile(0.1)
            # df[f"{feature_name}_p90"] = df[feature_name].quantile(0.9)
            df[f"{feature_name}_rolling_mean"] = df[feature_name].rolling(window=int(lags[0])).mean()
        return df.fillna(fill_method)
    
    def create_rolling_std_features(self, fill_method):
        df = self.df.copy()
        lags = data_config["lags"]
        for feature_name in self._x_cols:
            df[f"{feature_name}_rolling_std"] = df[feature_name].rolling(window=int(lags[0])).std()
        return df.fillna(fill_method)
    
    def create_ewma_features(self, fill_method):
        df = self.df.copy()
        lags = data_config["lags"]
        for feature_name in self._x_cols:
            df[f"{feature_name}_rolling_ewma"] = df[feature_name].ewm(com=int(lags[0])).mean()
        return df.fillna(fill_method)
                
    
    def split_data(self, scale=False):
        df = self.df.copy()
        test_data = df[df["profile_id"].isin(data_config["testset"])].drop(columns=['profile_id'])
        val_data = df[df["profile_id"].isin(data_config["valset"])].drop(columns=['profile_id'])
        train_data = df[~df["profile_id"].isin(data_config["testset"] + data_config["valset"])].drop(columns=['profile_id'])
        
        scaler = StandardScaler()

        X_train, y_train = train_data.drop(columns=self._y_cols), train_data[self._y_cols]
        X_val, y_val = val_data.drop(columns=self._y_cols), val_data[self._y_cols]
        X_test, y_test = test_data.drop(columns=self._y_cols), test_data[self._y_cols]
        
        if scale:
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        return X_train, np.squeeze(y_train), X_val, np.squeeze(y_val), X_test, np.squeeze(y_test), scaler
    
    
    
    