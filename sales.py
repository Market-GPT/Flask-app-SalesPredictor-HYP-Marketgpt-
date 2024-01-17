import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
import pandas as pd

# Load the data for scaling
df = pd.read_csv('./Datasetpreprocessed_data.csv')
scaler = MinMaxScaler()
  # Replace 'Net_Sales' with the appropriate column name

def preprocess_input(input_sales, time_steps):
    # Split the input string by commas and strip spaces, then convert each to float
    sales_data = np.array([float(sale.strip()) for sale in input_sales.split(',')])
    sales_data = sales_data[-time_steps:]  # Use the most recent 'time_steps' sales
    sales_data = sales_data.reshape(-1, 1)  # Reshape for scaling
    scaled_data = scaler.fit_transform(sales_data)  # Apply scaling
    return scaled_data.reshape(1, -1, 1)  # Reshape to 3D for LSTM input

def sales_prediction(input_sales, model_path, time_steps=5):
    model = keras.models.load_model(model_path)  # Load the model
    sales_data = preprocess_input(input_sales, time_steps)
    
    prediction = model.predict(sales_data)
    return scaler.inverse_transform(prediction)[0][0]  # Assuming prediction requires inverse transform


# Load monitoring data
df_monitoring = pd.read_csv('./Datasetpreprocessed_data.csv')

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
# column_mapping = {...}

# Uncomment the following line to generate Evidently AI reports
# get_data_drift_report_content()
