from sklearn.impute import SimpleImputer ## Handling missing value
from sklearn.preprocessing import StandardScaler,OneHotEncoder ## Handling feature scaling
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys,os
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initated')

            ## Categorical and numerical columns
            
            categorical_cols= ['Company', 'TypeName', 'CPU BRAND', 'Gpu Brand', 'OpSys']
            numerical_cols=   ['Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'HDD', 'SSD']

            logging.info('Data Transformation pipeline initiated')

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="mean")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_cols),
                ("cat_pipelines",cat_pipeline,categorical_cols)

            ])

            logging.info('Data_transformation_completed')

            return preprocessor
        

        except Exception as e:
            logging.info('Exception occured in Data Transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            ###
            train_df=pd.read_csv(train_data_path)
            train_df=train_df.drop(columns=['Gpu'],axis=1)
            test_df=pd.read_csv(test_data_path)
            test_df=test_df.drop(columns=['Gpu'],axis=1)

            logging.info("Read train and test data completed")
            logging.info(f'Train Dataframe Head : \n {train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n {test_df.head().to_string()}')

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformation_object()

            target_column='Price'

            ### Dividing the dataset into independent and dependent feature
            ### Training data
            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            ### test data
            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            ### Data Transformation
            logging.info("Applying preprocessing object on training and testing datasets")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

             # Print shapes of the input feature arrays

            # Print shapes to confirm
            print(input_feature_train_arr.shape, type(input_feature_train_arr))
            print(input_feature_test_arr.shape, type(input_feature_test_arr))
            print(input_feature_train_df.shape, type(input_feature_train_df))
            print(input_feature_test_df.shape, type(input_feature_test_df))
            print(np.array(target_feature_train_df).shape)
            print(np.array(target_feature_test_df).shape)
            train_arr = np.c_[
                input_feature_train_arr.toarray(), np.array(target_feature_train_df).reshape(-1, 1)
            ]
            # train_arr = np.column_stack((input_feature_train_arr, target_feature_train_df))
            test_arr = np.c_[input_feature_test_arr.toarray(), np.array(target_feature_test_df).reshape(-1, 1)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )




        

        except Exception as e:
            raise CustomException(e,sys)