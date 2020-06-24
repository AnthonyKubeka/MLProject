import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import nltk
#nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 

def removeStopWords(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered = [word for word in word_tokens if not word in stop_words]
    filtered = []
    
    for word in word_tokens:
        if word not in stop_words:
            filtered.append(word)
            
    return filtered

#read in data, drop unnamed axis, note: data is stored using pandas DataFrames
reddit_posts_df = pd.read_csv('data_cleansed.csv').drop(columns='Unnamed: 0')

features = ['text', 'title']
X = reddit_posts_df[features]
y = reddit_posts_df.target

Y = y.to_numpy().reshape(1693,1)
#Split arrays or matrices into random train and test subsets (train test split is from sklearn)


countVectorizerText = CountVectorizer(stop_words='english')
countVectorizerTitle = CountVectorizer(stop_words='english')
cvtext = countVectorizerText.fit_transform(X.text)
cvtitle = countVectorizerTitle.fit_transform(X.title)

#Build corpus representation
textArr = cvtext.toarray()
titleArr = cvtext.toarray()
print(textArr.shape)
print(titleArr.shape)
#combine text and title word count arrays horizontally, so the title for a row
# is put next to the text of that row
allArr = np.concatenate((titleArr,textArr), axis=1)
print(allArr.shape)

#finally, add the class corresponsding to each row as the last column of each row
corpus_table = np.hstack((allArr,Y))


# Model 1: Naive Bayes

    
#priors

def prob_count()

