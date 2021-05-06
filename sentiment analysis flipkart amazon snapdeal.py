#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis Program

# In[1]:


import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


# twitter API credentials
consumer_key="xxxxxxxxxxxxxxxxxxxxxx"
consumer_secret="xxxxxxxxxxxxxxxxxxxxxxxxx"
access_token="xxxxxxxxxxxxxxxxxxxxxxxxx"
access_secret="xxxxxxxxxxxxxxxxxxxxxxx"


# In[3]:


# creating authentication object
authenticate= tweepy.OAuthHandler(consumer_key, consumer_secret)

# set the access token
authenticate.set_access_token(access_token, access_secret)

# creating API object
api = tweepy.API(authenticate, wait_on_rate_limit = True)


# In[4]:


# extracting 3000 tweets from Flipkart
args=["flipkart","amazonIN","snapdeal"]
flipkart_tweets=[]
query=args[0]
if len(args)==3:
    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets -filter:replies ",lang="en",result_type= "mixed").items(3000):
        flipkart_tweets.append(status.text)


# In[44]:


df_flkt= pd.DataFrame(flipkart_tweets, columns=['Flipkart_tweets'])
df_flkt.head()


# In[45]:


df_flkt.shape


# In[46]:


df_flkt.to_csv(r'C:\Users\CEA\Documents\Python Scripts\flipkart_tweets.csv',index= True)


# In[47]:


df_flkt= pd.read_csv(r'C:\Users\CEA\Documents\Python Scripts\flipkart_tweets.csv')


# In[48]:


# clean the text
# create a function to clean the text

def cleantxt(text):
    text= text.lower()
    text= re.sub(r"http\S+", "", text)
    text= re.sub('@[^\s]+','',text)
    text= re.sub(r'\W',' ',text)
    text= re.sub(r'\d',' ',text)
    text= re.sub(r'\s+[a-z]\s+',' ',text)
    text= re.sub(r"^[a-z]\s+"," ", text)
    text= text.strip()
    text= re.sub(r'\s+',' ',text)
    
    return text

# cleaning the text
df_flkt['Flipkart_tweets']= df_flkt['Flipkart_tweets'].apply(cleantxt)

# show the cleaned text
df_flkt.head(10)


# In[49]:


# create a function to get subjectivity
def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# create a function to get polarity
def getpolarity(text):
    return TextBlob(text).sentiment.polarity

# create two coulumns in dataframe
df_flkt['subjectivity']= df_flkt['Flipkart_tweets'].apply(getsubjectivity)
df_flkt['Polarity']= df_flkt['Flipkart_tweets'].apply(getpolarity)

# see the new dataframe
df_flkt.head(10)


# In[50]:


df_flkt.iloc[7][3]


# In[51]:


senti_flkt=[]
j=0
for i in df_flkt['Flipkart_tweets']:
    
    if df_flkt.iloc[j][3] >= 0.25 : 
#         print("Positive") 
        senti_flkt.append("Positive")
  
    elif df_flkt.iloc[j][3] <= - 0.25 : 
#         print("Negative") 
        senti_flkt.append("Negative")
  
    else : 
#         print("Neutral")
        senti_flkt.append("Neutral")
    j=j+1


# In[52]:


print(senti_flkt[0:7])


# In[53]:


Sentiment_flkt=pd.DataFrame(senti_flkt)
Sentiment_flkt.columns=["sentiment"]
Sentiment_flkt.head(10)


# In[54]:


Sentiment_flkt.sentiment.value_counts()


# In[55]:


# adding the sentiment column to the dataframe
df_flkt_f=pd.concat([df_flkt,Sentiment_flkt], axis=1)
df_flkt_f.head(7)


# In[59]:


from wordcloud import STOPWORDS
stopwords = set(STOPWORDS) 


# In[60]:


# Plotting the word cloud

allwords= ''.join([twts for twts in df_flkt_f['Flipkart_tweets']])
Wordcloud = WordCloud(width = 800, height= 800, random_state=21, min_font_size=10, background_color='white', stopwords= stopwords).generate(allwords)

plt.figure(figsize=(9,8), facecolor= None)
plt.imshow(Wordcloud, interpolation ='bilinear')
plt.axis('off')
plt.show()


# In[63]:


# WordCloud of "Positive" (Flipkart)
comment_words = '' 
   
for val in df_flkt_f[df_flkt_f.sentiment=="Positive"].Flipkart_tweets: 
    tokens = val.split()       
    comment_words += " ".join(tokens)+" "

wordcloud_p = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                stopwords= stopwords,
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 7), facecolor = None) 
plt.imshow(wordcloud_p) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()


# In[64]:


# WordCloud of "Negative" (Flipkart)
comment_words = '' 
   
for val in df_flkt_f[df_flkt_f.sentiment=="Negative"].Flipkart_tweets: 
    tokens = val.split()       
    comment_words += " ".join(tokens)+" "

wordcloud_n = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                stopwords= stopwords,
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 7), facecolor = None) 
plt.imshow(wordcloud_n) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()


# ## Sentiment analysis for Amazon India Account

# In[67]:


# Extracting 3000 tweets from Amazon India
amazon_tweets=[]        
query=args[1]
if len(args)==3:
    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets -filter:replies",lang="en",result_type= "mixed").items(3000):
        amazon_tweets.append(status.text)


# In[68]:


df_amazon= pd.DataFrame(amazon_tweets, columns=['amazon_tweets'])
df_amazon.head()


# In[69]:


df_amazon.shape


# In[70]:


df_amazon.to_csv(r'C:\Users\CEA\Documents\Python Scripts\amazon_tweets.csv',index= True)


# In[71]:


df_amazon= pd.read_csv(r'C:\Users\CEA\Documents\Python Scripts\amazon_tweets.csv')


# In[72]:


# cleaning the text
df_amazon['amazon_tweets']= df_amazon['amazon_tweets'].apply(cleantxt)

# show the cleaned text
df_amazon.head(10)


# In[73]:


# create two coulumns in dataframe subjectivity and polarity
df_amazon['subjectivity']= df_amazon['amazon_tweets'].apply(getsubjectivity)
df_amazon['Polarity']= df_amazon['amazon_tweets'].apply(getpolarity)

# see the new dataframe
df_amazon.head(10)


# In[74]:


senti_amazon=[]
j=0
for i in df_amazon['amazon_tweets']:
    
    if df_amazon.iloc[j][3] >= 0.25 : 
#         print("Positive") 
        senti_amazon.append("Positive")
  
    elif df_amazon.iloc[j][3] <= - 0.25 : 
#         print("Negative") 
        senti_amazon.append("Negative")
  
    else : 
#         print("Neutral")
        senti_amazon.append("Neutral")
    j=j+1


# In[75]:


Sentiment_amazon=pd.DataFrame(senti_amazon)
Sentiment_amazon.columns=["sentiment"]
Sentiment_amazon.head(10)


# In[77]:


Sentiment_amazon.sentiment.value_counts()


# In[78]:


# adding the sentiment column to the dataframe
df_amazon_f=pd.concat([df_amazon,Sentiment_amazon], axis=1)
df_amazon_f.head(7)


# In[79]:


# Plotting the word cloud

allwords= ''.join([twts for twts in df_amazon_f['amazon_tweets']])
Wordcloud = WordCloud(width = 800, height= 800, random_state=21, min_font_size=10, background_color='white', stopwords= stopwords).generate(allwords)

plt.figure(figsize=(9,8), facecolor= None)
plt.imshow(Wordcloud, interpolation ='bilinear')
plt.axis('off')
plt.show()


# In[80]:


# WordCloud of "Positive" (Amazon)
comment_words = '' 
   
for val in df_amazon_f[df_amazon_f.sentiment=="Positive"].amazon_tweets: 
    tokens = val.split()       
    comment_words += " ".join(tokens)+" "

wordcloud_p = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                stopwords= stopwords,
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 7), facecolor = None) 
plt.imshow(wordcloud_p) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()


# In[81]:


# WordCloud of "Negative" (Amazon)
comment_words = '' 
   
for val in df_amazon_f[df_amazon_f.sentiment=="Negative"].amazon_tweets: 
    tokens = val.split()       
    comment_words += " ".join(tokens)+" "

wordcloud_n = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                stopwords= stopwords,
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 7), facecolor = None) 
plt.imshow(wordcloud_n) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()


# # Sentiment analysis for Snapdeal twiter

# In[93]:


# extracting 3000 tweets from snapdeal
snapdeal_tweets=[]        
query=args[2]
if len(args)==3:
    for status in tweepy.Cursor(api.search,q=query+"  ",lang="en",result_type= "mixed").items(3000):
        snapdeal_tweets.append(status.text)


# In[94]:


df_sdeal= pd.DataFrame(snapdeal_tweets, columns=['snapdeal_tweets'])
df_sdeal.head()


# In[95]:


df_sdeal.shape


# In[96]:


df_sdeal.to_csv(r'C:\Users\CEA\Documents\Python Scripts\snapdeal_tweets.csv',index= True)


# In[97]:


df_sdeal= pd.read_csv(r'C:\Users\CEA\Documents\Python Scripts\snapdeal_tweets.csv')


# In[98]:


# cleaning the text
df_sdeal['snapdeal_tweets']= df_sdeal['snapdeal_tweets'].apply(cleantxt)

# show the cleaned text
df_sdeal.head(10)


# In[99]:


# create two coulumns in dataframe subjectivity and polarity
df_sdeal['subjectivity']= df_sdeal['snapdeal_tweets'].apply(getsubjectivity)
df_sdeal['Polarity']= df_sdeal['snapdeal_tweets'].apply(getpolarity)

# see the new dataframe
df_sdeal.head(10)


# In[100]:


senti_snapdeal=[]
j=0
for i in df_sdeal['snapdeal_tweets']:
    
    if df_sdeal.iloc[j][3] >= 0.25 : 
#         print("Positive") 
        senti_snapdeal.append("Positive")
  
    elif df_sdeal.iloc[j][3] <= - 0.25 : 
#         print("Negative") 
        senti_snapdeal.append("Negative")
  
    else : 
#         print("Neutral")
        senti_snapdeal.append("Neutral")
    j=j+1


# In[101]:


Sentiment_snapdeal=pd.DataFrame(senti_snapdeal)
Sentiment_snapdeal.columns=["sentiment"]
Sentiment_snapdeal.head(10)


# In[102]:


Sentiment_snapdeal.sentiment.value_counts()


# In[103]:


# adding the sentiment column to the dataframe
df_sdeal_f=pd.concat([df_sdeal,Sentiment_snapdeal], axis=1)
df_sdeal_f.head(7)


# In[104]:


# Plotting the word cloud

allwords= ''.join([twts for twts in df_sdeal_f['snapdeal_tweets']])
Wordcloud = WordCloud(width = 800, height= 800, random_state=21, min_font_size=10, background_color='white', stopwords= stopwords).generate(allwords)

plt.figure(figsize=(9,8), facecolor= None)
plt.imshow(Wordcloud, interpolation ='bilinear')
plt.axis('off')
plt.show()


# In[105]:


# WordCloud of "Positive" (snapdeal)
comment_words = '' 
   
for val in df_sdeal_f[df_sdeal_f.sentiment=="Positive"].snapdeal_tweets: 
    tokens = val.split()       
    comment_words += " ".join(tokens)+" "

wordcloud_p = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                stopwords= stopwords,
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 7), facecolor = None) 
plt.imshow(wordcloud_p) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()


# In[106]:


# WordCloud of "Negative" (snapdeal)
comment_words = '' 
   
for val in df_sdeal_f[df_sdeal_f.sentiment=="Negative"].snapdeal_tweets: 
    tokens = val.split()       
    comment_words += " ".join(tokens)+" "

wordcloud_n = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                stopwords= stopwords,
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 7), facecolor = None) 
plt.imshow(wordcloud_n) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()


# In[ ]:




