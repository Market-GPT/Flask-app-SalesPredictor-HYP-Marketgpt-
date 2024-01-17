from flask import Flask, render_template, request
import sales  # Import your sales prediction script

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sub", methods=['POST'])
def submit():
    if request.method == "POST":
        input_sales = request.form["previousSales"]

        # Validate input: Check if it's a comma-separated list of numbers
        if not all(item.strip().replace('.', '', 1).isdigit() for item in input_sales.split(',')):
            return "Invalid input format. Please enter a comma-separated list of numbers."

        try:
            prediction = sales.sales_prediction(input_sales, './model.h5')
        except Exception as e:
            # Improved error handling
            return f"An error occurred: {e}"

        return render_template("sub.html", prediction=prediction)

@app.route("/data_drift_report")
def data_drift_report():
    data_drift_info = sales.get_data_drift_report_content()
    return render_template("data_drift_report.html", 
                           total_columns=data_drift_info['total_columns'], 
                           drifted_columns_count=data_drift_info['drifted_columns_count'], 
                           drifted_columns_names=data_drift_info['drifted_columns_names'])

@app.route("/target_drift_report")
def target_drift_report():
    return render_template("target_drift_report.html")

@app.route("/feature_distribution")
def feature_distribution():
    sales.feature_distribution()
    return render_template("feature_distribution.html")


""" if __name__ == "__main__":
    app.run(debug=True) """







