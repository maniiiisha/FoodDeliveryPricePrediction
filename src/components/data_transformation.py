import sys 
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated.')

            categorical_cols = ['Delivery_person_ID', 'Order_Date', 'Time_Orderd',
                                'Time_Order_picked', 'Weather_conditions', 'Road_traffic_density',
                                'Type_of_order', 'Type_of_vehicle', 'Festival', 'City']
            numerical_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
                                'Restaurant_longitude', 'Delivery_location_latitude',
                                'Delivery_location_longitude', 'Vehicle_condition',
                                'multiple_deliveries']

            '''Weather_conditions_categories = ['Fog', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Sunny']
            Road_traffic_density_categories = ['Jam', 'High', 'Medium', 'Low']
            Type_of_order_categories = ['Snack', 'Meal', 'Drinks', 'Buffet']
            Type_of_vehicle_categories = ['motorcycle', 'scooter', 'electric_scooter', 'bicycle']
            Festival_categories = ['No', 'Yes']
            City_categories = ['Metropolitian', 'Urban', 'Semi-Urban']'''

            logging.info('Pipeline initiated.')

            # Numerical Pipeline

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])
            
            return preprocessor
            logging.info('Pipeline completed.')

        except Exception as e:
            logging.info('Error in Data Transformation.')
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe: \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()
            
            drop_columns = ['Time_taken (min)', 'ID']
            target_column_name = ['Time_taken (min)']

            target_feature_train_df = train_df[target_column_name]
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)

            target_feature_test_df = test_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)

            # Transforming using preprocessor object

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing datasets.')

            target_feature_train_df = np.array(target_feature_train_df)
            target_feature_test_df = np.array(target_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved.')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info('Exception occured in the initiate_data_transformation')
            raise CustomException(e, sys)