import pandas as pd
from src.pipeline.predict_pipeline import Predict_Pipeline
from src.components.data_transformation import initiate_data_transformer,initiate_df_transformer,initiate_rnn_scaler
from src.components.data_ingestion import DataIngestion
from src.utils import load_object

model_path=r'artifacts\\model.pkl'
model=load_object(file_path=model_path)


obj=DataIngestion()
train_data,test_data,df=obj.initiate_data_ingestion()
df_arr=initiate_df_transformer(df)
X_train,X_test,y_train,y_test,scaled_gold_prices,scaler,dataset=initiate_rnn_scaler(df_arr)

pipe=Predict_Pipeline()
# Usage example:
window_size=30
num_future_days = 40  # Change this to the desired number of future days to predict
last_window = scaled_gold_prices[-window_size:].reshape(1, window_size, 1)
future_predictions = pipe.predict_future_days(model, scaler, last_window, num_future_days,dataset)
print(future_predictions)
