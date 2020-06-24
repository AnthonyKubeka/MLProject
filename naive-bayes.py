#TESTFILE
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

# Model 1: Naive Bayes
#find no of times a word appears in a class
#CountVectorizer gives Term-Document Matrix for each class (word frequencies appearing in a set of documents i.e. corpus)
#we compute this  TDM for each class / target. The two classes are 1 = gaming, 0 = pcmasterrace

reddit_posts_df['text'] = reddit_posts_df['title']+' '+reddit_posts_df['text']

gaming_docs  = [row['text'] for index, row in reddit_posts_df.iterrows() if row['target']==1]
cv_gaming = CountVectorizer(stop_words='english')
X_gaming = cv_gaming.fit_transform(gaming_docs)

tdm_gaming = pd.DataFrame(X_gaming.toarray(),columns=cv_gaming.get_feature_names())

pcmr_docs  = [row['text'] for index, row in reddit_posts_df.iterrows() if row['target']==0]
cv_pcmr = CountVectorizer(stop_words='english')
X_pcmr = cv_pcmr.fit_transform(pcmr_docs)

tdm_pcmr = pd.DataFrame(X_pcmr.toarray(),columns=cv_pcmr.get_feature_names())

#a cell of the tdm matrix is freq of the word in that sentence
#now, calculate number of times  each word appeared in each class

word_list_gaming = cv_gaming.get_feature_names()
count_list_gaming = X_gaming.toarray().sum(axis=0)
freq_gaming = dict(zip(word_list_gaming, count_list_gaming))

#so freq_gaming is the number of times a word appears in the class gaming(gaming subreddit)

word_list_pcmr = cv_pcmr.get_feature_names()
count_list_pcmr = X_pcmr.toarray().sum(axis=0)
freq_pcmr = dict(zip(word_list_pcmr, count_list_pcmr))

#now calculate the probability that a word will appear in each class (subreddit)

prob_gaming = []
for word, count in zip(word_list_gaming, count_list_gaming):
    prob_gaming.append(count / len(word_list_gaming))
prob_gaming_dict = dict(zip(word_list_gaming, prob_gaming))

prob_pcmr = []
for word, count in zip(word_list_pcmr, count_list_pcmr):
    prob_pcmr.append(count / len(word_list_pcmr))
prob_pcmr_dict = dict(zip(word_list_pcmr, prob_pcmr))


#for Laplace smoothing, get the total count of all features in our training data
all_docs = [row['text'] for index, row in reddit_posts_df.iterrows()]
cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(all_docs)

tot_features = len(cv.get_feature_names())

tot_features_gaming = count_list_gaming.sum(axis=0)
tot_features_pcmr = count_list_pcmr.sum(axis=0)

#now run it for one reddit post
reddit_post = 'What is this video game called? Title' # a gaming post
reddit_post_list = word_tokenize(reddit_post)

#check if it's gaming or pcmr using laplace smoothing
prob_gaming_laplace = []
for word in reddit_post_list:
    if word in freq_gaming.keys():
        count = freq_gaming[word]
    else:
        count = 0
    prob_gaming_laplace.append((count+1)/(tot_features_gaming+tot_features))

#probability that each word is in the gaming class
prob_gaming_dict = dict(zip(reddit_post_list, prob_gaming_laplace))

#probability that the post is in the gaming class

prob_post_in_gaming = 1
for word in reddit_post_list:
    prob_post_in_gaming = prob_post_in_gaming*prob_gaming_dict[word]


#now just compare