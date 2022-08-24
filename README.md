# Disaster Response Pipeline Project

This project consists of three parts. First, we are building an ETL Pipeline, where we load data, clean and process it and finally store it in a SQLite database. In the second step we are creating a machine learning pipeline, loading the data, building a machine learning model, where parameters are tuned using GridSearch. In the final step we are providing a Flask Web App, where you can enter your own message to be classified. Additionally you can see a few visuals explaining the data.

## Required Packages
 - pandas
 - numpy
 - sqlalchemy
 - pickle
 - nltk
 - sklearn
 - plotly
 - flask
 - joblib
 - json



## Instructions for running the app:


1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Files structure in the repository
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data with categories of the messages
|- disaster_messages.csv  # data messages and genres
|- process_data.py # file to process the data
|- process_data.ipynb
|- Disaster_reponse.db   # database to save clean data to

- models
|- train_classifier.py # file to generate and train the model
|- classifier.pkl  # saved model 

- README.md

## Acknowledges
I want to thank mentors and reviewers of the Udacity Nanodegree program where this project was part of.