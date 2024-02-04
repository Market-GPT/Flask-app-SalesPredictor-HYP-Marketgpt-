This app is made using flask framework. Also read the description of the app from about section.
All the required libraries are mentioned in the 'requirements.txt' file. Also latest python version don't support tensorflow. So use old python version (like 3.9.x , where you can vary x Or 3.11.x will also work I guess, but not sure.) 
You can use versions which is suitable for your system. But don't use recently published versions of python. You can also run the app by creating the virtual environment using conda.
This app is hosted also using render. Link for the live web-app: [https://aryan-sales-predictor-r.onrender.com/](https://aryan-sales-predictor-r.onrender.com/)
If you want to rehost it after making changes then first you have to make the app ready for production. For that you have to remove '''if __name__ == "__main__":
    app.run(debug=True)''' this which is present at the last of app.py file.
