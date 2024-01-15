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

        try:
            prediction = sales.sales_prediction(input_sales, './model.h5')
        except Exception as e:
            # Simple error message handling if error.html doesn't exist
            return f"An error occurred: {e}"

        return render_template("sub.html", prediction=prediction)

@app.route("/data_drift_report")
def data_drift_report():
    return render_template("data_drift_report.html")

@app.route("/target_drift_report")
def target_drift_report():
    return render_template("target_drift_report.html")

if __name__ == "__main__":
    app.run(debug=True)