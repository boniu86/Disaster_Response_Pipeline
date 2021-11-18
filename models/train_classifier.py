import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier



MODEL_PICKLE_FILENAME = 'trained_classifier.pkl'
DATABASE_FILEPATH = '../db.sqlite3'
TABLE_NAME = 'disaster_message'


def load_data(database_filepath):
    '''
    Load the data from the database
    Args:
        database_filepath: database file location
    Returns:
        X : input
        Y : dataframe containing the categories
        category_names: list containing the categories name
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(TABLE_NAME, engine)
    X = df['message']
    Y = df.iloc[:, 4:] #36 classes
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    '''
    Clean and tokenize the text messages X
    Args:
        text: input text
    Returns:
        clean_tokens: tokens obtained from the input text
    '''
    # normalize text and remove punctuation, lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

    
def build_model(grid_search_cv = False):
    '''
    Build the AdaBoostClassifier model, here I am using AdaBoostclassifier, from my ML jupyternotebook, we know that random forest takes too long to run, and adaboostclassifier is sginificantly way faster.
    Args:
        grid_search_cv (bool): if True after building the pipeline it will be performed an exhaustive search over specified parameter values ti find the best ones
    Returns:
        pipeline (pipeline.Pipeline): model
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultiOutputClassifier(AdaBoostClassifier()))
                    ])

    if grid_search_cv == True:
        print('Searching for best parameters...')
        parameters = {'vect__stop_words': (None,True),
                      'vect__max_features': (None, 5000,10000),
                      'tfidf__use_idf': (True, False)
        }

        pipeline = GridSearchCV(pipeline, param_grid = parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model performances
    Args:
        model: from pipeline
        X_test: test set contain input X
        Y_test: test set contain actual result classes
        category_names: categories name
    '''
    Y_pred = model.predict(X_test)
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
       print('Category: {} '.format(category_names[i]))
       print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
       print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])))


def save_model(model, model_filename):
    '''
    Save the model
    Args:
        model: model to be saved
        model_filename: model.plk file name
    '''
    pickle.dump(model, open(model_filename, 'wb'))


def load_model(model_pickle_filename):
    '''
    Return model from pickle file
    Args:
        model_pickle_filename (str): source pickle filename
    Returns:
        model (pipeline.Pipeline): model readed from pickle file 
    '''
    return pickle.load(open(model_pickle_filename, 'rb'))


def parse_input_arguments():
    '''
    Parse the command line arguments
    Returns:
        database_filename (str): database filename. Default value DATABASE_FILENAME
        model_pickle_filename (str): pickle filename. Default value MODEL_PICKLE_FILENAME
        grid_search_cv (bool): If True perform grid search of the parameters
    '''
    parser = argparse.ArgumentParser(description = "Disaster Response Pipeline Train Classifier")
    parser.add_argument('--database_filename', type = str, default = DATABASE_FILEPATH, help = 'Database filename of the cleaned data')
    parser.add_argument('--model_pickle_filename', type = str, default = MODEL_PICKLE_FILENAME, help = 'Pickle filename to save the model')
    parser.add_argument('--grid_search_cv', action = "store_true", default = False, help = 'Perform grid search of the parameters')
    args = parser.parse_args()
    #print(args)
    return args.database_filename, args.model_pickle_filename, args.grid_search_cv


def train(database_filename, model_pickle_filename, grid_search_cv = False):
    '''
    Train the model and save it in a pickle file
    Args:
        database_filename (str): database filename
        model_pickle_filename (str): pickle filename
        grid_search_cv (bool): if True after building the pipeline it will be performed an exhaustive search over specified parameter values ti find the best ones
    '''
    # print(database_filename)
    # print(model_pickle_filename)
    # print(grid_search_cv)
    # print(os.getcwd())

    print('Download nltk componets if needed...')
    nltk.download(['punkt', 'wordnet'])

    print('Loading data...\n    Database: {}'.format(database_filename))
    X, Y, category_names = load_data(database_filename)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    print('Building model...')
    model = build_model(grid_search_cv)

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    Model: {}'.format(model_pickle_filename))
    save_model(model, model_pickle_filename)

    print('Trained model saved!')


if __name__ == '__main__':
    database_filename, model_pickle_filename, grid_search_cv = parse_input_arguments()
    train(database_filename, model_pickle_filename, grid_search_cv)