# import libraries
import sys
import re
import sys
import time
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords', 'omw-1.4'])
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    load data from database and split into X (feature matrix) and Y (response)
    INPUT:
        database_filepath - data base name
    OUTPUT:
    X - feature matrix
    Y - response
    '''
    # create engine and read df
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM messages_disaster_response", engine)
    # create X and Y
    X = df.message.values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values

    return X, Y


def tokenize(text):
    '''
    tokenize data
    INPUT:
        text - text data
    OUTPUT:
        clean_tokens - lemmatized and tokenized data
    '''
    # tokenize
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text))
    
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    #remove stopwords
    clean_tokens = [w for w in clean_tokens if not w in stopwords.words("english")]

    return clean_tokens

def build_model():
    '''
    build machine learning model pipeline, set hyperparameters using GridSearch
    OUTPUT:
        model - built model
    '''
    #building model pipeline
    pipeline = Pipeline(
        [
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ]
    )

    # select parameters
    parameters = {
        'clf__estimator__n_estimators': [5,7],
        'clf__estimator__min_samples_split': [2,3]
    }

    # GridSearch for hyperparameter tuning
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test):
    '''
    evaluate the model performance
    INPUT:
        model - model to evaluate
        X_test - test set of feature matrix
        Y_test - test set of response
    '''
    # predict on X_test
    Y_pred = model.predict(X_test)

    # classification report for prediction
    report = classification_report(Y_test, Y_pred)

    # print report and best parameters
    print(report)
    print("\nBest Parameters:", model.best_params_)




def save_model(model, model_filepath):
    '''
    save model
    INPUT:
        model - trained model
        model_filepath - path to save the model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()