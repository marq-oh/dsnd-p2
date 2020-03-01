import sys
import nltk
import re
import os
import numpy as np
from numpy import asarray
from numpy import savetxt
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

# Load Data and set feature (y) and target (X) variables
def load_data(database_filepath):
    # load data from database
    cwd = os.getcwd()  # gets current working directory
    dbwd = cwd.replace('/models', '/data/').replace('\\models', '\\data\\')
    engine = create_engine('sqlite:///' + dbwd + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages_Categories_Cleaned", engine)

    X = df.message
    Y = df.iloc[:, 4:]
    category_names = Y.columns

    return X, Y, category_names

# Tokenization function to process data
def tokenize(text):
    # URL detection; replace URLs with urlplaceholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize text
    tokens = word_tokenize(text)

    # Initiate lematizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Build model
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search
    parameters = {'clf__estimator__max_depth': [10, 20, None],
                  'clf__estimator__min_samples_leaf':[2, 5, 10]}

    # create grid search object
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=1)

    return model

# Evaluate model -> Find out what this function should do
def evaluate_model(model, X_test, Y_test, category_names):
    # Get results using classification_report; add to a dataframe.
    y_pred = model.predict(X_test)
    for col1, col2 in zip(targets.T, pred_classes.T):
        # print("col1: ", col1)
        # print("col2: ", col2)
        print("Category: ", target_names[count])
        print("================================")
        print(classification_report(y_true=col1, y_pred=col2))
        print("================================")
        count += 1

# Save model as pickle file
def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))

# Load main function
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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