import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from evidently.report import Report
from evidently import ColumnMapping

from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset

# Function to preprocess input data
def preprocess_input(input_sales):
    sales_data = np.array([float(sale) for sale in input_sales.split(',')])

    # Pad the input if it has less than 6 values
    if sales_data.shape[0] < 6:
        padding = np.zeros(6 - sales_data.shape[0])
        sales_data = np.concatenate((padding, sales_data))

    # Trim the input if it has more than 6 values
    if sales_data.shape[0] > 6:
        sales_data = sales_data[-6:]  # Keep the last 6 values

    return sales_data.reshape(-1, 1)

# Sales prediction function
def sales_prediction(input_sales, model_path):
    sales_data = preprocess_input(input_sales)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(sales_data)

    # Ensure the input is correctly reshaped for the model
    # The model expects an input shape of (1, 5, 6)
    model_input = np.zeros((1, 5, 6))

    # Fill in the scaled_data into model_input
    for i in range(5):
        if i < scaled_data.shape[0]:
            model_input[0, i, :-1] = scaled_data[i:i+1, 0]

    # The last column (current sales) is repeated across all timesteps
    model_input[0, :, -1] = scaled_data[-1, 0]

    # Load and predict using the model
    model = keras.models.load_model(model_path)
    prediction = model.predict(model_input)
    return prediction[0][0]


df_monitoring=pd.read_csv('./Datasetpreprocessed_data.csv')
# Split your df_monitoring into reference and current datasets
split_index = len(df_monitoring) // 2
df_reference = df_monitoring.iloc[:split_index]
df_current = df_monitoring.iloc[split_index:]

# Prepare data for Evidently AI
column_mapping = ColumnMapping()
column_mapping.target = 'Actual Sales'
column_mapping.prediction = 'Predicted Sales'
# Function to generate and save Evidently AI reports
def generate_evidently_report(df_reference, df_current, column_mapping):
    # Generating a regression performance report
    regression_performance_report = Report(metrics=[RegressionPreset()])
    regression_performance_report.run(current_data=df_current, reference_data=df_reference, column_mapping=column_mapping)
    regression_performance_report.save('./templates/regression_performance_report.html')

    # Generating a target drift report
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(current_data=df_current, reference_data=df_reference, column_mapping=column_mapping)
    target_drift_report.save('./templates/target_drift_report.html')

    # Generating a data drift report
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(current_data=df_current, reference_data=df_reference, column_mapping=column_mapping)
    data_drift_report.save('./templates/data_drift_report.html')
    
def get_data_drift_report_content():
    generate_evidently_report(df_reference, df_current, column_mapping)
# Example usage (for testing)
# result = sales_prediction('5000,4500,5200', 'path_to_model.h5')
# print(result)

# Example for generating reports (for testing)
# reference_data = pd.DataFrame(...)  # Replace with your data
# current_data = pd.DataFrame(...)    # Replace with your data
# column_mapping = {...




