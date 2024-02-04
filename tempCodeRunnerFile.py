ef preprocess_input(input_sales, time_steps):
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
    return scaler.inverse_transform(prediction)[0][0]