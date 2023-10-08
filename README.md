# sentiment_analysis_of_a_hotel
in this i used python or nltk to predict sentiment of customer
Project Name
Sentiment Analysis with NLTK and Scikit-Learn

Overview
This project utilizes the NLTK library along with Scikit-Learn to perform sentiment analysis on customer reviews. The goal is to determine whether customers liked or disliked the product based on their provided reviews.

Dependencies
pandas
numpy
re
nltk
sklearn
You can install the required packages using the following command:

bash
Copy code
pip install pandas numpy nltk scikit-learn
Usage
Import necessary libraries:
python
Copy code
import pandas as pd
import numpy as np 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import GaussianNB
Load the dataset:
python
Copy code
df = pd.read_csv("a1_RestaurantReviews_HistoricDump.tsv", quoting=3, delimiter='\t')
Preview the dataset:
python
Copy code
df.head()
Define the function to process the reviews:
python
Copy code
def make_fun(df):
    k, p = df.shape
    print(k, p)
    corpus = []
    for i in range(0, k):
        review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    return corpus
Initialize necessary NLTK components:
python
Copy code
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
Perform sentiment analysis:
python
Copy code
# Use the `make_fun` function to process the reviews
corpus = make_fun(df)

# Perform further analysis (e.g., using Scikit-Learn's machine learning models)
Notes
The make_fun function processes the reviews to prepare them for sentiment analysis.
The NLTK library is used for text processing, including stemming and removing stopwords.
The processed data can then be fed into a machine learning model (not shown in this code snippet) for sentiment analysis.
Disclaimer
This project is for educational purposes and may require further customization for real-world applications. 
