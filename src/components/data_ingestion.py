import os 
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered data ingestion method.')
        try:
            df=pd.read_csv(r'C:\projects\goldpredict\notebook\data\gold2024.csv')
            logging.info('Reading data as dataframe,')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train test split initiated.')
            size=int(len(df)*0.8)
            train_set=df[:size]
            test_set=df[size:]

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of data is completed.')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path)
        except Exception as e:
            raise CustomException(e,sys)
        

