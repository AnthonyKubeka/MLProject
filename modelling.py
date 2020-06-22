import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import nltk
nltk.download('punkt')
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

#Split arrays or matrices into random train and test subsets (train test split is from sklearn)


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1,test_size=0.20)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1)
#Prep data for usage in the model


#print(X_train.head(n=3))
#print(X_train[2:3])
#print(X_train.loc[1])
#
#
#print('-----------------')
#
#
#print(X_train.text[1130])


print('---------------')
#remove stop words by iterating over text and title features of training data, also tokenizes
for i in X_train.index:
    X_train.text[i]=removeStopWords(X_train.text[i])
    X_train.title[i]=removeStopWords(X_train.title[i])
    
# Data is now ready for applying models.
    
# Model 1: Logistic Regression
    
def sigmoid(x):
    return 1 / (1+np.exp(-x))

