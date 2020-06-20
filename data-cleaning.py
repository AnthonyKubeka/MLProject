from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from wordcloud import WordCloud, STOPWORDS

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV

import warnings
warnings.simplefilter(action='ignore')

reddit_posts_df = pd.read_csv('data_raw.csv').drop(columns = 'Unnamed: 0')
reddit_posts_df.drop_duplicates(keep = 'first', inplace = True)
#if a text body is missing it will have 'this is missing text' there
reddit_posts_df.fillna('thisismissingtext', inplace=True)
reddit_posts_df.target.value_counts(normalize=True)

reddit_posts_df.to_csv('data_cleansed.csv')
wc_df = pd.read_csv('data_cleansed.csv')
text = ' '
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for x in wc_df.title: 
      
    # typecaste each val to string 
    x = str(x) 
  
    # split the value 
    values = x.split() 
      
    # Converts each token into lowercase 
    for i in range(len(values)): 
        values[i] = values[i].lower() 
          
    for words in values: 
        text = text + words + ' '
  
  
wc = WordCloud(max_words= 100,
                      width = 744, 
                      height = 544,
                      background_color ='white',
                      stopwords=stopwords, 
                      contour_width=3, 
                      contour_color='steelblue',
                      min_font_size = 10).generate(text) 
  
# plot the WordCloud image                        
plt.figure(figsize = (14, 14)) 
plt.imshow(wc) 
plt.axis("off")
plt.savefig('reddit_wordcloud.png')