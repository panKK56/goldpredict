import sys
import os
from src.exception import CustomException
from src.logger import logging
from math import sqrt
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from pmdarima import auto_arima

from src.utils import save_object
from src.components.data_transformation import initiate_data_transformer,initiate_df_transformer,initiate_rnn_scaler
from src.components.data_transformation import initiate_stationary_converter
from src.components.data_transformation import adf_test
from src.components.data_ingestion import DataIngestion

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

obj=DataIngestion()
train_data,test_data,df=obj.initiate_data_ingestion()
train_arr,test_arr=initiate_data_transformer(train_path=train_data,test_path=test_data)
df_arr=initiate_df_transformer(df)
train_diff,test_diff=initiate_stationary_converter(train_arr,test_arr)
X_train,X_test,y_train,y_test,scaled_gold_prices,scaler,dataset=initiate_rnn_scaler(df_arr)

class Model_Trainer():
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def tune_arima(train, test):
        logging.info('arima hyper parameter tunning.')
        # Fit auto_arima function to the dataset
        model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(train)

        # Make predictions on the test set
        predictions = model.predict(n_periods=len(test))

        # Calculate RMSE
        rmse = sqrt(mean_squared_error(test, predictions))
        print('RMSE: ', rmse)
        return model.order



    def arima_model(train_arr,test_arr,df_arr):

        # Call the function
        pdq=tune_arima(train_arr, test_arr)
        logging.info('Arima model building started.')
        try:
            arima=ARIMA(train_arr,order=(5,0,5))
            model_fit=arima.fit()
            y_hat_ar=df_arr.copy()
            y_hat_ar['arima_forecast']=model_fit.predict(df_arr.index.min(),df_arr.index.max())
            rmse_arima = np.sqrt(mean_squared_error(test_arr, y_hat_ar['arima_forecast'][test_arr.index.min():])).round(2)
            print(rmse_arima)
        except Exception as e:
            raise CustomException(e,sys)

    def sarima_model(self,train_arr,test_arr,df_arr):
        logging.info('Sarimax model building started.')
        try:
            model = SARIMAX(train_arr, order=(1,1,1), seasonal_order=(1,1,1,12))
            model_fit= model.fit()
            y_hat_sarima = df_arr.copy()
            y_hat_sarima['sarima_forecast'] = model_fit.predict(df_arr.index.min(), df_arr.index.max())
            y_hat_sarima['sarima_forecast'] =(y_hat_sarima['sarima_forecast'])
            rmse_sarima = np.sqrt(mean_squared_error(df_arr, y_hat_sarima['sarima_forecast'])).round(2)
            print('Sarima rmse : ',rmse_sarima)
            
            
        except Exception as e:
            raise CustomException(e,sys)
        

    def rnn_model(self,X_train,X_test,y_train,y_test):
        try:

            # Build the RNN model
            model = Sequential([
                LSTM(50, input_shape=(X_train.shape[1], 1)),
                Dense(1)
            ])

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X_train, y_train, epochs=100, batch_size=32)

            # Evaluate the model
            loss = model.evaluate(X_test, y_test)
            print("Test Loss:", loss)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
        except Exception as e:
            raise CustomException(e,sys)



if __name__=='__main__':
    mod=Model_Trainer()
    mod.rnn_model(X_train,X_test,y_train,y_test)

