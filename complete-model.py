import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import math

reddit_posts_df = pd.read_csv('data_cleansed.csv').drop(columns='Unnamed: 0')
reddit_posts_df = shuffle(reddit_posts_df).reset_index(drop=True)
#features = ['text', 'title']
#X = reddit_posts_df[features]


#reddit_posts_df['text'] = reddit_posts_df['title']+' '+reddit_posts_df['text']


#split the sets 60/20/20
train_index = math.floor((60/100)*reddit_posts_df.shape[0])
val_index = train_index+math.floor((20/100)*reddit_posts_df.shape[0])
reddit_posts_df_train = reddit_posts_df[:train_index]
reddit_posts_df_val = reddit_posts_df[train_index:val_index]
reddit_posts_df_test = reddit_posts_df[val_index:]

#training set
reddit_posts_df_train['text'] = reddit_posts_df_train['title'] + ' '+reddit_posts_df_train['text']
X_train = reddit_posts_df_train['text']
y_train = reddit_posts_df_train.target

#test set
reddit_posts_df_test['text'] = reddit_posts_df_test['title'] + ' '+reddit_posts_df_test['text']
X_test = reddit_posts_df_test['text']
y_test = reddit_posts_df_test.target

#validation set
reddit_posts_df_val['text'] = reddit_posts_df_val['title'] + ' '+reddit_posts_df_val['text']
X_val = reddit_posts_df_val['text']
y_val = reddit_posts_df_val.target

#prep training data
gaming_docs  = [row['text'] for index, row in reddit_posts_df_train.iterrows() if row['target']==1]
cv_gaming = CountVectorizer(stop_words='english')
X_gaming = cv_gaming.fit_transform(gaming_docs)

tdm_gaming = pd.DataFrame(X_gaming.toarray(),columns=cv_gaming.get_feature_names())

pcmr_docs  = [row['text'] for index, row in reddit_posts_df_train.iterrows() if row['target']==0]
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

#1: Naive Bayes
def trainNB():
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
    all_docs = [row['text'] for index, row in reddit_posts_df_train.iterrows()]
    cv = CountVectorizer(stop_words='english')
    X = cv.fit_transform(all_docs)
    
    tot_features = len(cv.get_feature_names())

    tot_features_gaming = count_list_gaming.sum(axis=0)
    tot_features_pcmr = count_list_pcmr.sum(axis=0)
    
    return tot_features, tot_features_pcmr, tot_features_gaming

def predict_subredditNB(new_post, tot_features_pcmr, tot_features_gaming, tot_features ):
    #check if it's gaming or pcmr using laplace smoothing
    new_post_list = word_tokenize(new_post)
    prob_gaming_laplace = []
    for word in new_post_list:
        if word in freq_gaming.keys():
            count = freq_gaming[word]
        else:
            count = 0
        prob_gaming_laplace.append((count+1)/(tot_features_gaming+tot_features))
    
    #probability that each word is in the gaming class
    prob_gaming_dict = dict(zip(new_post_list, prob_gaming_laplace))
    
    #probability that the post is in the gaming class
    
    prob_post_in_gaming = 1
    for word in new_post_list:
        prob_post_in_gaming = prob_post_in_gaming*prob_gaming_dict[word]
        
    
    prob_pcmr_laplace = []
    for word in new_post_list:
        if word in freq_pcmr.keys():
            count = freq_pcmr[word]
        else:
            count = 0
        prob_pcmr_laplace.append((count+1)/(tot_features_pcmr+tot_features))
    
    prob_pcmr_dict = dict(zip(new_post_list, prob_pcmr_laplace))
    
    prob_post_in_pcmr = 1
    for word in new_post_list:
        prob_post_in_pcmr = prob_post_in_pcmr*prob_pcmr_dict[word]
        
    
    #now compare probabilities to decide which subreddit the post is in
    if prob_post_in_gaming > prob_post_in_pcmr:
        return 'The post is in the gaming subreddit', prob_gaming_laplace
    else:
        return 'The post is in the pcmasterrace subreddit', prob_pcmr_laplace

tot_features, tot_features_pcmr, tot_features_gaming = trainNB()
print(predict_subredditNB('alienware alienware alienware',tot_features_pcmr,tot_features_gaming,tot_features ))
    
#2: Logistic Regression
#setup the data
def trainLR(X_set, y, alpha):#pass in y as y.to_numpy(), pass in X_set as a fullX dataframe
    #turn X into X = (x1, x2) where x1 (class 0) is no of pcmr words in post, x2 (class 1) is no of gaming words in post
    X = np.zeros((X_set.shape[0],2))
    
    for data_post in range(len(y)):
        X[data_post][0] = no_of_pcmr_words_in_post(X_set[data_post])
        X[data_post][1] = no_of_gaming_words_in_post(X_set[data_post])
    
    #make a design matrix with a bias term 1 as x0
    design_matrix = np.hstack([np.ones(X.shape[0])[np.newaxis].T, X])
    X = design_matrix
    #initialise weights as 1 
    theta = np.ones(X.shape[1])
    theta_old = np.zeros(X.shape[1])
    
    while np.sqrt(np.sum(np.power(theta-theta_old, 2))) > 0.0005:
        theta_old = theta
        for i in range(X.shape[0]):
            theta = theta - alpha*((h(X[i], theta)-y[i])*X[i])
        
    print("Model parameters: ", theta)

    model_predictions = h(X, theta)
    return model_predictions, theta

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


#regression function
def h(x, theta):
    return 1 / (1+np.exp(-np.dot(x,theta)))

y_train_ = y_train.to_numpy()

model_predictions, theta = trainLR(X_train, y_train_, 0.1)

def predict_subredditLR(X_predict, y_predict, theta_predict): #whereas naive bayes predict takes in one post, this function takes in a full testing set of posts
    X = np.zeros((X_predict.shape[0],2))
    
    for data_post in range(len(y_predict)):
        X[data_post][0] = no_of_pcmr_words_in_post(X_predict[data_post+X_predict.keys()[0]])
        X[data_post][1] = no_of_gaming_words_in_post(X_predict[data_post+X_predict.keys()[0]])
        
    design_matrix = np.hstack([np.ones(X.shape[0])[np.newaxis].T, X])
    X = design_matrix
    
    model_predictions = h(X, theta_predict)
    return model_predictions

pcmr_category_train = np.where(model_predictions <0.5) #note: these are indices
gaming_category_train = np.where(model_predictions >= 0.5)

test_predictions = predict_subredditLR(X_test, y_test, theta)
pcmr_category_test = np.where(test_predictions < 0.5)
gaming_category_test = np.where(test_predictions >= 0.5)