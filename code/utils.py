import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, recall_score
import requests
import time

def get_data(subreddit):
    """Returns a list of subreddit titles specified by subreddit parameter, with cleaned duplicates.
    Creating this function and for loop due to the API capping data pulled at 100 posts per pull."""

    # url for api
    url = 'https://api.pushshift.io/reddit/search/submission' 
    total_text = []
    # initialize days before day of pull, will be added on later in the function to continue pulling posts further back in time
    days = 1 

    # initialize a start time
    start_time = time.time() 

    for _ in range(80):
        # initialize a current time at start of loop
        current_time = time.time() 

        params = {
            'subreddit' : subreddit,
            # creating multiple iterations that pulls data at different time horizons
            'before' : f'{days}d', 
            'size' : 100 
        }

        res = requests.get(url, params)
        json = res.json()

        # initializing empty list to contain all post text (unfiltered). 
        # text_list is the list of text for 1 iteration.
        text_list = []

        # For loop iterating over json titles and appending them to the text_list
        for dictionary in json['data']: 
            if 'selftext' in dictionary.keys():
                text_list.append(dictionary['selftext'])
            text_list.append(dictionary['title'])

        # Conducting this if statement to prevent overloading the API
        if current_time - start_time >= 5:
            time.sleep(5) 
            start_time = time.time()   

        # remove duplicates
        text_list = list(set(text_list)) 
        # append text_list into titles list to add more titles for each loop. 
        # total_text is the list of titles for all iterations.
        total_text.extend(text_list) 
    
        # arbitrary addition to number of days to try to prevent duplicate posts
        days += 20 

    # removing potential duplicates between each iteration of the for loop (extra precaution)
    return list(set(total_text)) 

def avg_word_len_func(text):
    """Returns an average word length in a string."""
    text = text.split()
    length_list = [len(x) for x in text]
    return np.mean(length_list)

def top_words_plot(series_title, series, color):
    """Returns a plot that shows the top-used words in user-input series."""
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,        
        max_df=0.9
    )

    # Fit / Transform
    X_ft = tfidf.fit_transform(series)

    X_words_df = pd.DataFrame(X_ft.todense(), 
                              columns=tfidf.get_feature_names_out())

    # plot top occuring words
    plt.figure(figsize=(16,9))
    X_words_df.sum().sort_values(ascending=False).head(20).plot(kind='barh', color=color)
    plt.title(f'Top {series_title} Words Used')
    plt.xlabel('Count')
    plt.ylabel('Words');

def run_baseline(model, 
                 X_train, y_train, X_test, y_test,
                 verbose=True):
    """Return a dictionary of model performance indicators."""
    
    results = {}
    
    # 1. Fit the model on the training set
    model.fit(X_train, y_train)

    # 2. Predict on the training and test sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 3. Train & test accuracy
    results['train_recall'] = recall_score(y_train, y_pred_train)
    results['test_recall'] = recall_score(y_test, y_pred_test)
    
    if verbose:
        print('Train Recall: ', results['train_recall'])
        print('Test Recall: ', results['test_recall'])
    
    return results

def test_models(models, X_train, y_train, X_test, y_test,
                verbose=False):
    """Returns DataFrame of baseline results 
       given a dictionary `models` of names/sklearn models."""
    results = {}
    
    # Fit each model and store how it performs on the test set
    for name, model in models.items():
        if verbose:
            print('\nRunning {} - {}'.format(name, model))
        
        results[name] = run_baseline(model, 
                                     X_train, y_train, 
                                     X_test, y_test, 
                                     verbose=False)
        if verbose:
            print('Results: ', results[name])

    return pd.DataFrame.from_dict(
        results, 
        orient='index').sort_values(
            by='test_recall',
            ascending=False
        )

def print_final_results(model_name, model, X_test, y_test):
    """Returns a classification report, confusion matrix, and ROC Curve for a user-input model and test data."""
    
    # Generate Model Predictions
    y_preds = model.predict(X_test)
    
    # Print Classification Report
    print(classification_report(y_test, y_preds))

    # Print Confusion Matrix
    _, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues', ax=ax, normalize='true')
    plt.title(f'{model_name} Confusion Matrix');

    # Print ROC Curve
    _, ax = plt.subplots(figsize=(16, 9))
    ax.title.set_text(f'{model_name} ROC Curve')
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="red")
    RocCurveDisplay.from_predictions(y_test, y_preds, ax=ax);  