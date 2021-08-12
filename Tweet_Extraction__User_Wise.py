#!/usr/bin/env python
# coding: utf-8

# # Tweet Extraction

# #### By Ridhit Bhura

# In[11]:


import pandas as pd
import tweepy
import time
import warnings
warnings.filterwarnings('ignore')


# For a guide to obtain your relevant keys and secret codes, follow instructions on https://support.yapsody.com/hc/en-us/articles/360003291573-How-do-I-get-a-Twitter-Consumer-Key-and-Consumer-Secret-key-

# In[12]:


consumer_key="ADD_YOUR_OWN"
consumer_secret="ADD_YOUR_OWN"
access_token="ADD_YOUR_OWN"
access_token_secret="ADD_YOUR_OWN"


# In[13]:


#Twitter API
auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token=(access_token, access_token_secret)
api= tweepy.API(auth, wait_on_rate_limit= True)


# In[14]:


pd.set_option('display.max_colwidth', -1)


# In[15]:


def get_tweets(username, count):
    try:
        #creating query methods using parameters
        tweets= tweepy.Cursor(api.user_timeline,id= username, lang="en", tweet_mode="extended").items(count)
        tweet_list= [[tweet.created_at, tweet.id, tweet.full_text] for tweet in tweets]
        #creating dataframe from tweets list
        tweets_df=pd.DataFrame(tweet_list, columns=["Date","Tweet_id","Tweet_Text"])
    
    except BaseException as e:
        print('failed on_status', str(e))
        time.sleep(3)
    return tweets_df


# In[16]:


df=get_tweets("narendramodi",30000)
df.shape
df.tail()


# ## Export to CSV

# In[ ]:


df.to_csv('CUSTOM_NAME.csv', index=False)
df.to_csv('FILE_PATH_TO_WHEREVER_YOU_WANT.csv')

