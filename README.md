# Disaster_Response_Pipeline
Udacity data science nanodegree project 2

### Table of contents

1. [Libraries used](#Libraries)
2. [Project Inspiration](#Inspiration)
3. [File Descriptions](#files)
4. [Data Insights](#Insights)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Libraries used <a name="Libraries used"></a>

Python version 3.0.
dependecies: json, plotly, numpy, pandas, nltk, flask, plotly, sklearn, sqlalchemy, os, sys, re, pickle, argparse

## Project Inspiration<a name="Inspiration"></a>

Apply data engineering skills to expand your opportunities and potential as a data scientist. In this project I analyze disaster data 
from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages. The project will include a [web 
app](https://view6914b2f4-3001.udacity-student-workspaces.com/) where an emergency worker can input a new message and get classification 
results in several categories.


## File Descriptions <a name="files"></a>

*__app__* : template has master.html (main page of web app) and go.html (classification result page of web app), also run.py  (Flask file that runs app)

*__data__* : data files, oringial data are in csv forms, and other database form data are cleaned data

*__models__* : saved model for web app and details of ML in train_classifier.py

*__README.md__* : initial techinical README file for the project 

*__asset__* : web app screenshot

*__ETL_Pipeline_Prepation.ipynb__* : extract,transfer and load data. Prepared the data for ML analysis, saved data as database instead of csv.

*__ML_Pipeline_Prepation.ipynb__* : serval time consuming ML algorithm process. After considering running time, I use AdaBoost instead of Random forest

*__all.plk__* :  all the ML models I saved, there are 2 random forest model, 2 AdaBoost models

## Insights<a name="insights"></a>

Result web app [here](https://view6914b2f4-3001.udacity-student-workspaces.com/).

![web app screenshot](asset/pic1.png)

![web app screenshot](asset/pic2.png)


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Data : Figure eight data [here](https://appen.com/)
