# Twitter Sentiment Analysis Project

## Overview

In this project, I performed sentiment analysis on Twitter data using a logistic regression model. The goal was to classify tweets as either positive or negative based on their content.

## Steps Taken

1. **Data Loading**: 
   - Loaded the Twitter dataset containing tweets and their corresponding sentiment labels.

2. **Text Cleaning**:
   - Implemented a cleaning function (`clean`) to preprocess the text data.
   - The cleaning function performs the following steps:
     - Replace non-alphabetic characters with space.
     - Convert the text to lowercase.
     - Split the text into words.
     - Remove English stopwords using NLTK's `stopwords` module.
     - Apply stemming to the words using NLTK's `PorterStemmer`.
     - Join the cleaned words back into a single string.

3. **Text Vectorization**:
   - Used a vectorizer (e.g., `TfidfVectorizer`) to convert the cleaned text data into numerical features suitable for modeling.
   
4. **Modeling**:
   - Implemented a logistic regression model to classify the tweets into positive or negative sentiment categories.
   - Trained the logistic regression model on the vectorized text data.

## Files

- `twitter_sentiment_analysis.ipynb`: Jupyter Notebook containing the code for the project.
- `twitter_model.pkl`: Pickle file containing the trained logistic regression model.

## Usage

1. **Clone the Repository**:
   ```bash
   git clone <git clone https://github.com/Adham-yacout/Instant_training.git
>
