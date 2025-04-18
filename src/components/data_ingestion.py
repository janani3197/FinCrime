import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    raw_data_parquet_path: str = os.path.join('artifacts', "raw_data.parquet")
    train_data_parquet_path: str = os.path.join('artifacts', "train_data.parquet")
    test_data_parquet_path: str = os.path.join('artifacts', "test_data.parquet")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method/component.")
        try:
            # Step 1: Read the raw CSV file
            df = pd.read_csv('notebook/data/financial_dataset.csv')
            logging.info("Read the dataset as a DataFrame.")

            # Step 2: Save raw data as Parquet
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_parquet_path), exist_ok=True)
            df.to_parquet(self.ingestion_config.raw_data_parquet_path, index=False)
            logging.info(f"Saved raw data to Parquet at {self.ingestion_config.raw_data_parquet_path}")

            # Step 3: Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Performed train-test split.")

            # Step 4: Save train and test data as Parquet
            train_set.to_parquet(self.ingestion_config.train_data_parquet_path, index=False)
            test_set.to_parquet(self.ingestion_config.test_data_parquet_path, index=False)
            logging.info("Saved train and test data to Parquet format.")

            return (
                self.ingestion_config.train_data_parquet_path,
                self.ingestion_config.test_data_parquet_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path, test_path)

