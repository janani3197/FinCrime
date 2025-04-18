import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import chi2_contingency

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        try:
            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load the training and test data
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)
            logging.info("Parquet files loaded successfully.")

            # ---- UNDERSAMPLING ----
            logging.info("Applying undersampling on training data.")
            fraud_df = train_df[train_df['isFraud'] == 1]
            non_fraud_df = train_df[train_df['isFraud'] == 0].sample(n=len(fraud_df), random_state=42)
            train_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)

            # ---- Feature Selection: Chi-squared Test ----
            # Choose categorical columns for the Chi-squared test
            categorical_cols = ['type', 'nameOrig', 'nameDest']

            # Apply Chi-squared test between each categorical feature and the target variable
            for col in categorical_cols:
                contingency = pd.crosstab(train_df[col], train_df['isFraud'])
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
                logging.info(f"Chi-squared test for {col}:")
                logging.info(f"Chi-squared statistic: {chi2_stat}")
                logging.info(f"p-value: {p_value}")
                
                # You can choose a threshold for p-value (e.g., 0.05) to decide if the feature is significant
                if p_value < 0.05:
                    logging.info(f"{col} is significantly related to isFraud (p-value < 0.05).")
                else:
                    logging.info(f"{col} is NOT significantly related to isFraud (p-value >= 0.05).")

            # ---- Select Significant Categorical Columns Based on Chi-squared Test ----
            selected_cat_cols = [col for col in categorical_cols if chi2_contingency(pd.crosstab(train_df[col], train_df['isFraud']))[1] < 0.05]
            logging.info(f"Selected categorical columns: {selected_cat_cols}")

            # ---- Label Encoding for Significant Categorical Columns ----
            label_encoders = {}
            for col in selected_cat_cols:
                le = LabelEncoder()
                train_df[col] = le.fit_transform(train_df[col].astype(str))
                test_df[col] = le.transform(test_df[col].astype(str))
                label_encoders[col] = le

            # ---- Correlation for Numerical Features ----
            target_col = "isFraud"
            X = train_df.drop(columns=[target_col])
            y = train_df[target_col]

            numerical_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

            # Correlation for numerical features with target
            corr = train_df[numerical_cols + [target_col]].corr()[target_col].abs()
            selected_num_cols = corr[corr > 0.1].drop(target_col).index.tolist()
            logging.info(f"Selected numerical columns: {selected_num_cols}")

            # ---- Preprocessing Pipeline ----
            preprocessor = self.get_data_transformer_object(selected_num_cols, selected_cat_cols)

            # Prepare data for model
            input_train = train_df[selected_num_cols + selected_cat_cols]
            input_test = test_df[selected_num_cols + selected_cat_cols]
            y_train = train_df[target_col]
            y_test = test_df[target_col]

            X_train_transformed = preprocessor.fit_transform(input_train)
            X_test_transformed = preprocessor.transform(input_test)

            # Combine the transformed features with the target column
            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            # Save the preprocessor object for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Data transformation completed successfully.")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
