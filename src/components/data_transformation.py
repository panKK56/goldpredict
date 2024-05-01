import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def initiate_data_transformer(train_path,test_path):
    logging.info('Data transformation started')
    try:
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)
        train_df['date'] = pd.to_datetime(train_df['date'])
        test_df['date'] = pd.to_datetime(test_df['date'])
        train_df.set_index('date',drop=True,inplace=True)
        test_df.set_index('date',drop=True,inplace=True)

        return train_df,test_df
    except Exception as e:
        raise CustomException(e,sys)

def initiate_df_transformer(data_path):
    logging.info('Data transformation started')
    try:
        df=pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        df.set_index('date',drop=True,inplace=True)
        

        return df
    except Exception as e:
        raise CustomException(e,sys)
    

def initiate_stationary_converter(train_path,test_path):
    logging.info('Converting to stationary')
    try:
        # train_df=pd.read_csv(train_path)
        # test_df=pd.read_csv(test_path)
        train_df_diff=train_path.diff().dropna()
        test_df_diff=test_path.diff().dropna()

        return train_df_diff,test_df_diff
    except Exception as e:
        raise CustomException(e,sys)
def initiate_rnn_scaler(data_path):
    try:
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np

        # Load your dataset
        # Assuming your dataset has a column named 'price' containing the daily gold prices
        # Replace 'your_dataset.csv' with the path to your dataset
        dataset = data_path.copy()
        gold_prices = dataset['price'].values.reshape(-1, 1)

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_gold_prices = scaler.fit_transform(gold_prices)

        # Define the window size for input sequences
        window_size = 30

        # Create sequences of input data and corresponding labels
        X = []
        y = []
        for i in range(len(scaled_gold_prices) - window_size):
            X.append(scaled_gold_prices[i:i+window_size, 0])
            y.append(scaled_gold_prices[i+window_size, 0])
        X = np.array(X)
        y = np.array(y)

        # Reshape the input data for LSTM
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Split the data into training and testing sets
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        return(X_train,X_test,y_train,y_test,scaled_gold_prices,scaler,dataset)
    
    except Exception as e:
        raise CustomException(e,sys)

def adf_test(series):
    result = adfuller(series.dropna())
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    if result[1] <= 0.05:
        print("Reject the null hypothesis")
        print("Data is stationary")
    else:
        print("Fail to reject the null hypothesis")
        print("Data is non-stationary")