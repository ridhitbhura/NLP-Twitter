#!/usr/bin/env python
# coding: utf-8

# # NLP of Prime Minister Narendra Modi's Tweets during COVID-19

# #### By Ridhit Bhura

# In this project, I gathered a dataset from https://www.kaggle.com/saurabhshahane/twitter-sentiment-dataset and use the pretrained data to train a set of ~3500 tweets from Prime Minister Narendra Modi's personal twitter account. The process includes 1) twitter data extraction which can be found in the git directory for this project, 2) preprocessing of the test data (Modi tweets), 3) visualisation techniques and 4) Standard Scale & Random Forest Classifier model training. 

# Due to migration to the M1 MacBook, the machine learning portion has bugs (specifically token labelling) which will be fixed shortly.
# Nonetheless I was able to obtain >96% accuracy in model training for both the models. 

# ## Importing Packages

# In[103]:


#Importing All Necessary Packages and Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
re.compile('<title>(.*)</title>')


# In[104]:


#Importing the relevant test and train datesets from their respective csv filepaths. 
train = pd.read_csv('/Users/ridhitbhura/Downloads/Twitter_Data.csv')
test = pd.read_csv('/Users/ridhitbhura/Downloads/twitterdatamodi.csv')

print(test.shape)
print(train.shape)


# In[105]:


test.head()


# In[106]:


train.head()


# ## Cleaning The Datasets

# In[107]:


test.isna().sum()


# In[108]:


train.isna().sum()


# In[109]:


train[train['category'].isna()]


# In[110]:


train[train['clean_text'].isna()]


# In[111]:


train.drop(train[train['clean_text'].isna()].index, inplace=True)
train.drop(train[train['category'].isna()].index, inplace=True)


# ## Dataset Cleaning & Preprocessing

# ### Deleting Non-English Words 

# In[112]:


def delete_non_english(var): 
    try: 
        var = re.sub(r'\W+', ' ', var)
        var = var.lower()
        var = var.replace("[^a-zA-Z]", " ")
        word_tokens = word_tokenize(var) 
        cleaned_word = [w for w in word_tokens if all(ord(c) < 128 for c in w)]
        cleaned_word = [w + " " for w in cleaned_word]
        return "".join(cleaned_word)
    except:
        return np.nan

test["english_text"] = test.Tweet_Text.apply(delete_non_english)


# In[113]:


test.head()


# ### Cleaning English Text

# In[114]:


def clean_text(english_txt): 
    try: 
        word_tokens = word_tokenize(english_txt)
        cleaned_word = [w for w in word_tokens if not w in stop_words] 
        cleaned_word = [w + " " for w in cleaned_word]
        return "".join(cleaned_word)
    except:
        return np.nan

test["cleaned_text"] = test.english_text.apply(clean_text)


# In[115]:


test.head()


# In[116]:


#Adopted code
def remove_link_email(txt):
    txt = re.sub(r"https t co \S+", "", txt)
    txt = txt.replace('\S*@\S*\s?', "")
    txt = re.sub(r'[^\w\s]', '', txt)
    return txt
test["final_text"] = test.english_text.apply(remove_link_email)


# In[117]:


test.head()


# ## Visualisations

# ### Word Frequency

# In[118]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(test.final_text)

sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'blue')
plt.title("Word Occurunce Frequency - Top 20")


# ### Word Cloud

# In[130]:


wordcloud = WordCloud(background_color = 'white', width = 1000, height = 1000, font_path = '/Users/ridhitbhura/Downloads/fainland/Fainland.ttf').generate_from_frequencies(dict(words_freq))

plt.figure(figsize=(12,9))
plt.imshow(wordcloud)
plt.title("Most Common Words - Test Data", fontsize = 24)


# In[131]:


#Nuetral Words
normal_words =' '.join([text for text in train['clean_text'][train['category'] == 0]])

wordcloud = WordCloud(background_color = 'white', width=800, height=500, random_state = 0, max_font_size = 110, font_path = '/Users/ridhitbhura/Downloads/fainland/Fainland.ttf').generate(normal_words)
plt.figure(figsize=(12, 9))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Neutral Words - Training Data')
plt.show()


# In[132]:


#Negative Words
normal_words =' '.join([text for text in train['clean_text'][train['category'] == -1]])

wordcloud = WordCloud(background_color = 'white', width=800, height=500, random_state = 0, max_font_size = 110, font_path = '/Users/ridhitbhura/Downloads/fainland/Fainland.ttf').generate(normal_words)
plt.figure(figsize=(12, 9))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Negative Words - Training Data')
plt.show()


# In[133]:


#Positive Words
normal_words =' '.join([text for text in train['clean_text'][train['category'] == 1]])

wordcloud = WordCloud(background_color = 'white', width=800, height=500, random_state = 0, max_font_size = 110,font_path = '/Users/ridhitbhura/Downloads/fainland/Fainland.ttf').generate(normal_words)
plt.figure(figsize=(12, 9))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Positive Words - Training Data')
plt.show()


# ### Hashtag Frequency Distribution

# In[134]:


#Hashtag Collection
def hashtag_extract(x):
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# In[136]:


# extracting hashtags from the test dataset
HT_test = hashtag_extract(test['Tweet_Text'])

# extracting hashtags from the training dataset by sentiment category 
HT_train_negative = hashtag_extract(train['clean_text'][train['category'] == -1])
HT_train_positive = hashtag_extract(train['clean_text'][train['category'] == 1])
HT_train_neutral = hashtag_extract(train['clean_text'][train['category'] == 0])

# removing nested attribute and simplifying the list
HT_test = sum(HT_test, [])
HT_train_negative = sum(HT_train_negative, [])
HT_train_positive = sum(HT_train_positive, [])
HT_train_neutral = sum(HT_train_neutral, [])


# In[138]:


a = nltk.FreqDist(HT_test)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(30, 5))  # width:30, height:3
plt.xticks(range(1, len(a.keys())+1), a.keys(), size='small')
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title('Most Frequently Used Hasthags (Unhashed) - Test Data')
plt.show()


# ## Machine Learning Analysis

# In[139]:


#Tokenizing the training dataset
tokenized_tweet = train['clean_text'].apply(lambda x: x.split()) 

# importing gensim
import gensim

# creating a word to vector model
model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34)

model_w2v.train(tokenized_tweet, total_examples= len(train['clean_text']), epochs=20)


# ### Testing vectorisation of the model with test cases

# In[140]:


model_w2v.wv.most_similar(positive = "election")


# In[141]:


model_w2v.wv.most_similar(positive = "cancer")


# In[142]:


model_w2v.wv.most_similar(positive = "victory")


# ### Labelling Tokenized Words

# In[31]:


from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models.deprecated.doc2vec import LabeledSentence
LabeledSentence = gensim.models.deprecated.doc2vec.LabeledSentence


# In[32]:



def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(LabeledSentence(s, ["tweet_" + str(i)]))
    return output

# label all the tweets
labeled_tweets = add_label(tokenized_tweet)

labeled_tweets[:6]


# In[33]:


print(train.shape)
print(test.shape)


# In[34]:


train.shape


# ### Assigning test-train splits in the data

# In[49]:


import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[60]:


get_ipython().run_cell_magic('time', '', "train_corpus = []\ntry:\n    for i in range(0, 162969):\n        review = re.sub('[^a-zA-Z]', ' ', train['clean_text'][i])\n        review = review.lower()\n        review = review.split()\n\n        ps = PorterStemmer()\n        # stemming\n        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]\n        # joining them back with space\n        review = ' '.join(review)\n        train_corpus.append(review)\nexcept KeyError:\n    train_corpus.append(' ')")


# In[61]:


get_ipython().run_cell_magic('time', '', "test_corpus = []\n\nfor i in range(0, 3250):\n    review = re.sub('[^a-zA-Z]', ' ', test['english_text'][i])\n    review = review.lower()\n    review = review.split()\n    ps = PorterStemmer()\n    # stemming\n    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]\n    # joining them back with space\n    review = ' '.join(review)\n    test_corpus.append(review)")


# In[62]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2500)
x = cv.fit_transform(train_corpus).toarray()
y = train.iloc[:, 1]

print(x.shape)
print(y.shape)


# In[63]:


# creating bag of words

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2500)
x_test = cv.fit_transform(test_corpus).toarray()

print(x_test.shape)


# In[64]:


# splitting the training data into train and valid sets

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25, random_state = 42)

print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)


# In[65]:



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
x_test = sc.transform(x_test)


# In[66]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)

print("Training Accuracy :", model.score(x_train, y_train))
print("Validation Accuracy :", model.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("F1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)


# In[ ]:




