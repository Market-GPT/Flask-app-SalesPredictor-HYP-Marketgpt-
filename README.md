This application is built using the Flask framework. Check the 'requirements.txt' file for the necessary libraries. It's recommended to use an older version of Python, such as 3.9.x or 3.11.x (with variation in x), as the latest versions may not support TensorFlow.

To set up the environment, you can either install the required libraries directly or create a virtual environment using conda. Once the dependencies are installed, run the app locally by executing the 'app.py' file in your code editor.

The live web app is hosted on Render, and you can access it through this link: [https://aryan-sales-predictor-r.onrender.com/](https://aryan-sales-predictor-r.onrender.com/).

If you plan to rehost the app after making changes, prepare it for production by removing the following lines from the 'app.py' file:

```python
if __name__ == "__main__":
    app.run(debug=True)
```

Feel free to adapt the Python version and hosting environment according to your system requirements.
