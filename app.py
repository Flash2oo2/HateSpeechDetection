import pickle
import streamlit as st
from nltk.stem.porter import PorterStemmer
import re
import nltk
import string
import pandas
import numpy


tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


stopwords = nltk.corpus.stopwords.words("english")

# extending the stopwords to include other words used in twitter such as retweet(rt) etc.
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
stemmer = PorterStemmer()


def preprocess(tweet):
    # removal of extra spaces
    regex_pat = re.compile(r"\s+")
    tweet_space = re.sub(regex_pat, " ", tweet)

    # removal of @name[mention]
    regex_pat = re.compile(r"@[\w\-]+")
    tweet_name = re.sub(regex_pat, "", tweet_space)

    # removal of links[https://abc.com]
    giant_url_regex = re.compile(
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
        "[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    tweets = re.sub(giant_url_regex, "", tweet_name)

    # tweets = tweets.apply(
    #     lambda x: [item for item in x if item not in string.punctuation]
    # )

    # removal of punctuations and numbers
    # punc_remove = tweets.str.re("[^a-zA-Z]", " ", regex=True)
    # remove whitespace with a single space
    newtweet = re.sub(r"\s+", " ", tweets)
    # remove leading and trailing whitespace
    newtweet = re.sub(r"^\s+|\s+?$", "", newtweet)
    # replace normal numbers with numbr
    newtweet = re.sub(r"\d+(\.\d+)?", "numbr", newtweet)
    # removal of capitalization
    # tweet_lower = newtweet.str.lower()

    # # tokenizing
    # tokenized_tweet = tweet_lower.apply(lambda x: x.split())
    y = []
    text = newtweet.lower()
    text = nltk.word_tokenize(text)

    # removal of stopwords
    # tokenized_tweet = tokenized_tweet.apply(
    #     lambda x: [item for item in x if item not in stopwords]
    # )

    for i in text:
        if i not in stopwords and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(stemmer.stem(i))
    # # stemming of the tweets
    # tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

    # for i in range(len(tokenized_tweet)):
    #     tokenized_tweet[i] = " ".join(tokenized_tweet[i])
    #     tweets_p = tokenized_tweet

    return " ".join(y)


st.title("Hate Speech Recogniser")

input_msg = st.text_area("Enter the text")

if st.button("Predict"):
    preprocess_msg = preprocess(input_msg)

    vectorize_msg = tfidf.transform([preprocess_msg])

    result = model.predict(vectorize_msg)[0]

    if result == 1:
        st.header("Hate Speech or Offensive Language Detected")

    else:
        st.header("Not a Hate Speech")
