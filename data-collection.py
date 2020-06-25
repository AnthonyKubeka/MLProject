#This script scrapes data from reddit posts in designated subreddits

import requests
import time
import pandas as pd
import json
import math

import warnings
warnings.simplefilter(action='ignore')

gaming_url = 'https://www.reddit.com/r/gaming/.json'
headers = {'User-agent': 'Tonys Robot'}

#each request returns 25 posts so will need to loop
gaming_result = requests.get(gaming_url, headers = headers)

print(gaming_result.status_code)

gaming_json = gaming_result.json()
sorted(gaming_json.keys())

#we will pull the desired features: Post Text and Post Title from the json and put them into 
#Pandas Dataframes
ಠ_ಠ = "Status Code Error ಠ_ಠ" #valid python3 lol
print(ಠ_ಠ)

def scrape_posts(subreddit_url, num_of_posts_to_get, subreddit_headers):
    loop_for = math.floor(num_of_posts_to_get / 25)
    posts = []
    post_after = None
    
    for post in range(loop_for):
        print(post)
        if post_after == None:
            post_params = {}
        else:
            post_params = {'after':post_after}
            
        post_res = requests.get(subreddit_url, params = post_params, headers = subreddit_headers)
        if post_res.status_code == 200:
            post_json = post_res.json()
            posts.extend(post_json['data']['children'])
            post_after = post_json['data']['after']
            
        else: 
            print(ಠ_ಠ, post_res.status_code)
            break
    time.sleep(1)
    
    return posts, post_after
#getting data (raw)
    
gaming_posts, gaming_after = scrape_posts(gaming_url,1000,headers)


pcmr_posts, pcmr_after = scrape_posts('https://www.reddit.com/r/pcmasterrace/.json',1000,headers)

#get the features we want from the data, place into dataframes, give class values
gaming_posts_text = [x['data']['selftext'] for x in gaming_posts]
gaming_titles = [x['data']['title'] for x in gaming_posts]

pcmr_posts_text = [x['data']['selftext'] for x in pcmr_posts]
pcmr_titles = [x['data']['title'] for x in pcmr_posts]

gaming_df = pd.DataFrame({'text':gaming_posts_text,'title':gaming_titles})

gaming_df['target'] =1

pcmr_df = pd.DataFrame({'text':pcmr_posts_text,'title':pcmr_titles})

pcmr_df['target'] =0

pd.concat([gaming_df, pcmr_df]).to_csv('data_raw.csv')