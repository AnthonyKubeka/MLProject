import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 

reddit_posts_df = pd.read_csv('data_cleansed.csv').drop(columns='Unnamed: 0')

#features = ['text', 'title']
#X = reddit_posts_df[features]
y = reddit_posts_df.target

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


#tdm_gaming.to_csv('gaming-corpus-table.csv')

#2: Logistic Regression
#setup the data


def no_of_gaming_words_in_post(post):
    post_list = word_tokenize(post)
    count = 0
    for word in post_list:
        if word in freq_gaming.keys():
            count += 1
    return count

def no_of_pcmr_words_in_post(post):
    post_list = word_tokenize(post)
    count = 0
    for word in post_list:
        if word in freq_pcmr.keys():
            count+= 1
    return count

y = y.to_numpy()

#regression function
def h(x, theta):
    return 1 / (1+np.exp(-np.dot(x,theta)))


#turn X into X = (x1, x2) where x1 (class 0) is no of pcmr words in post, x2 (class 1) is no of gaming words in post
X = np.zeros((1693,2))
fullX = reddit_posts_df['text']

for data_post in range(len(y)):
    X[data_post][0] = no_of_pcmr_words_in_post(fullX[data_post])
    X[data_post][1] = no_of_gaming_words_in_post(fullX[data_post])

#make a design matrix with a bias term 1 as x0
design_matrix = np.hstack([np.ones(X.shape[0])[np.newaxis].T, X])
X = design_matrix
#initialise weights as 1 
theta = np.ones(X.shape[1])
#learning rate
alpha = 0.1
theta_old = np.zeros(X.shape[1])

while np.sqrt(np.sum(np.power(theta-theta_old, 2))) > 0.0005:
    theta_old = theta
    for i in range(X.shape[0]):
        theta = theta - alpha*((h(X[i], theta)-y[i])*X[i])
        
print("Model parameters: ", theta)

model_predictions = h(X, theta)

pcmr_category = np.where(model_predictions <0.5) #note: these are indices
gaming_category = np.where(model_predictions >= 0.5)