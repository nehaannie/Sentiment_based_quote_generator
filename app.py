from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
from random import randrange
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

app = Flask(__name__)

# Load quotes in memory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUOTES_FILE = os.path.join(BASE_DIR, 'quotes.csv')

# Declare global variable
quotes = None

@app.before_request
def prep():
    global quotes
    quotes = pd.read_csv(QUOTES_FILE)  # Load the CSV file with quotes
    sid = SentimentIntensityAnalyzer()  # Initialize the sentiment analyzer
    
    # Calculate sentiment scores for each quote
    all_compounds = [sid.polarity_scores(sentence)['compound'] for sentence in quotes['quote']]
    
    # Add sentiment scores to the DataFrame and sort by these scores
    quotes['sentiment_score'] = all_compounds
    quotes = quotes.sort_values('sentiment_score')
    
    # Add a new index column
    quotes['index'] = [i for i in range(len(quotes))]

def give_a_quote(direction=None, current_index=None):
    max_index_value = len(quotes) - 1  # Correct max_index_value

    # Initialize the random index
    rand_index = randrange(max_index_value + 1)

    if current_index is not None:
        try:
            # Ensure current_index is an integer
            current_index = int(current_index)
        except ValueError:
            current_index = rand_index  # If conversion fails, use a random index
        
        if direction == 'darker':
            # Choose a random index from the lower sentiment scores
            if current_index > 0:
                rand_index = randrange(0, current_index)
        elif direction == 'brighter':
            # Choose a random index from the higher sentiment scores
            if current_index < max_index_value:
                rand_index = randrange(current_index + 1, max_index_value + 1)
    # If current_index is None, or direction is not specified, return a random index
    return rand_index

@app.route("/")
def quote_me():
    global quotes
    quote_stash_tmp = quotes.copy()
    max_index_value = len(quote_stash_tmp) - 1

    darker = request.args.get("darker")
    brighter = request.args.get("brighter")

    if darker is not None:
        try:
            current_index = int(darker)
        except ValueError:
            current_index = randrange(max_index_value + 1)

        new_index = give_a_quote(direction='darker', current_index=current_index)
    
    elif brighter is not None:
        try:
            current_index = int(brighter)
        except ValueError:
            current_index = randrange(max_index_value + 1)

        new_index = give_a_quote(direction='brighter', current_index=current_index)
    
    else:
        # Grab a random value
        new_index = randrange(max_index_value + 1)
    
    random_quote = quote_stash_tmp[quote_stash_tmp['index'] == new_index].iloc[0]
    
    quote = random_quote['quote']
    author = random_quote['author']
    current_id = random_quote['index']

    return render_template("quote.html", quote=quote, author=author, current_id=current_id)

if __name__ == "__main__":
    app.run(debug=True)
