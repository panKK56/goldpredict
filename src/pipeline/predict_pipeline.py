import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class Predict_Pipeline:
    def __init__(self) -> None:
        pass

    def predict(self,steps):
        try:
            model_path='artifacts\model.pkl'
            model=load_object(file_path=model_path)
            preds=model.forecast(steps)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict_future_days(self,model, scaler, last_window, num_days,dataset):
        try:
            """
            Predict future gold prices for the specified number of days.
            
            Args:
            - model: Trained LSTM model
            - scaler: MinMaxScaler used for normalization
            - last_window: Last window of data from the dataset
            - num_days: Number of future days to predict
            
            Returns:
            - DataFrame with predicted prices for future days
            """
            future_predictions = []
            for _ in range(num_days):
                prediction = model.predict(last_window)
                future_predictions.append(prediction[0, 0])
                
                # Update the input sequence for the next prediction
                last_window = np.roll(last_window, -1, axis=1)
                last_window[0, -1, 0] = prediction[0, 0]

            # Inverse transform the future predictions
            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

            # Generate future dates for the predictions
            last_date = dataset.index[-1]
            future_dates = pd.date_range(start=last_date, periods=num_days + 1)[1:]  # Start from the day after the last date

            # Create a DataFrame for the future predictions
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_predictions.flatten()})

            # Set the 'Date' column as the index
            future_df.set_index('Date', inplace=True)

            return future_df
        
        except Exception as e:
            raise CustomException(e,sys)

