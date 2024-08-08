from src.exception import CustomException
from src.logger import logging
import sys
from src.components.data_ingestion import DataIngestion
## from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformation

if __name__=="__main__":

    logging.info("the execution has started")


    try:
        data_ingestion=DataIngestion()
        ## data_ingestion_config=DataIngestionConfig()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        data_transformation=DataTransformation()
        train_arr,test_arr,obj_path=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        

    except Exception as e:
        raise CustomException(e,sys)