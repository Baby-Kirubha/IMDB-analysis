#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("IMDB_Movies.csv")


# In[3]:


df.head(2)


# In[4]:


df.dtypes


# In[5]:


df["genres"]= df["genres"].str.split("|")


# In[6]:


df['genres']


# In[7]:


df=df.explode("genres")


# In[8]:


df.head(2)


# In[9]:


df['num_user_for_reviews']=df['num_user_for_reviews'].replace(" ",0)
df['num_user_for_reviews'] = pd.to_numeric(df['num_user_for_reviews'], errors='coerce', downcast='integer')


# In[10]:


df["num_user_for_reviews"].mean()


# In[11]:


(df.isnull().sum())/len(df)


# In[12]:


df['language'].fillna(method='ffill', inplace=True)


# In[13]:


df['country']=df['country'].fillna(df['country'].mode().iloc[0])


# In[14]:


df['actor_1_name'].fillna(method='ffill', inplace=True)
df['actor_3_name'].fillna(method='bfill', inplace=True)
df['plot_keywords'].fillna(method='ffill', inplace=True)
df['content_rating'].fillna(method='bfill', inplace=True)
df['title_year'].fillna(method='ffill', inplace=True)


# In[15]:


len(df['color'].isnull())


# In[16]:


# Calculate the ratio of each unique value
value_counts = df['color'].value_counts(normalize=True)
import numpy as np
# Fill null values based on the ratio
null_indices = df['color'].isnull()
df.loc[null_indices, 'color'] = np.random.choice(value_counts.index, size=null_indices.sum(), p=value_counts.values)


# In[17]:


df.color.value_counts()


# In[18]:


df['director_name'].mode()


# In[19]:


df['director_name']=df['director_name'].fillna(df['director_name'].mode().iloc[0])


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(df["num_critic_for_reviews"])
median=df["num_critic_for_reviews"].median()
mean=df["num_critic_for_reviews"].mean()
plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(mean, color='g', linestyle='dashed', linewidth=2, label='Mean')


# In[21]:


df['num_critic_for_reviews']=df['num_critic_for_reviews'].fillna(df['num_critic_for_reviews'].mean())


# In[22]:


sns.distplot(df["duration"])
median=df["duration"].median()
mean=df["duration"].mean()
plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(mean, color='g', linestyle='dashed', linewidth=2, label='Mean')


# In[23]:


df['duration']=df['duration'].fillna(df['duration'].mean())


# In[24]:


sns.distplot(df["director_facebook_likes"])
median=df["director_facebook_likes"].median()
mean=df["director_facebook_likes"].mean()
plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(mean, color='g', linestyle='dashed', linewidth=2, label='Mean')


# In[25]:


df['director_facebook_likes']=df['director_facebook_likes'].fillna(df['director_facebook_likes'].mean())


# In[26]:


sns.distplot(df["actor_3_facebook_likes"])
median=df["actor_3_facebook_likes"].median()
mean=df["actor_3_facebook_likes"].mean()
plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(mean, color='g', linestyle='dashed', linewidth=2, label='Mean')


# In[27]:


df['actor_3_facebook_likes']=df['actor_3_facebook_likes'].fillna(df['actor_3_facebook_likes'].mean())


# In[28]:


sns.distplot(df["actor_1_facebook_likes"])
median=df["actor_1_facebook_likes"].median()
mean=df["actor_1_facebook_likes"].mean()
plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(mean, color='g', linestyle='dashed', linewidth=2, label='Mean')


# In[29]:


df['actor_1_facebook_likes']=df['actor_1_facebook_likes'].fillna(df['actor_1_facebook_likes'].mean())


# In[30]:


df['actor_2_name']=df['actor_2_name'].ffill()


# In[31]:


sns.distplot(df["gross"])
median=df["gross"].median()
mean=df["gross"].mean()
plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(mean, color='g', linestyle='dashed', linewidth=2, label='Mean')


# In[32]:


df['gross']=df['gross'].fillna(df['gross'].mean())


# In[33]:


sns.distplot(df["facenumber_in_poster"])
median=df["facenumber_in_poster"].median()
mean=df["facenumber_in_poster"].mean()
plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(mean, color='g', linestyle='dashed', linewidth=2, label='Mean')


# In[34]:


fb=df.groupby("director_name").agg({"facenumber_in_poster":"mean"})


# In[35]:


sns.distplot(fb["facenumber_in_poster"])
median=fb["facenumber_in_poster"].median()
mean=fb["facenumber_in_poster"].mean()
plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(mean, color='g', linestyle='dashed', linewidth=2, label='Mean')


# In[36]:


df['facenumber_in_poster']=df['facenumber_in_poster'].fillna(fb['facenumber_in_poster'].mean())


# In[37]:


sns.distplot(df["num_user_for_reviews"])
median=df["num_user_for_reviews"].median()
mean=df["num_user_for_reviews"].mean()
plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(mean, color='g', linestyle='dashed', linewidth=2, label='Mean')


# In[38]:


df['num_user_for_reviews']=df['num_user_for_reviews'].fillna(df['num_user_for_reviews'].mean())


# In[39]:


df['aspect_ratio']=df['aspect_ratio'].fillna(df['aspect_ratio'].median())


# In[40]:


df.isnull().sum()


# In[41]:


sns.distplot(df["actor_2_facebook_likes"])
median=df["actor_2_facebook_likes"].median()
mean=df["actor_2_facebook_likes"].mean()
plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(mean, color='g', linestyle='dashed', linewidth=2, label='Mean')


# In[42]:


f=df.groupby("actor_2_name").agg({"actor_2_facebook_likes":"median"})


# In[43]:


df.columns


# In[44]:


bud=df.groupby("color").agg({'budget':"mean"})


# In[45]:


sns.distplot(bud["budget"])


# In[46]:


df['budget']=df['budget'].fillna(bud['budget'].mean())


# In[47]:


sns.distplot(df['actor_2_facebook_likes'])


# In[48]:


ac=df.groupby('genres').agg({"actor_2_facebook_likes":"mean"})


# In[49]:


sns.distplot(ac['actor_2_facebook_likes'])
median=ac["actor_2_facebook_likes"].median()
mean=ac["actor_2_facebook_likes"].mean()
plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(mean, color='g', linestyle='dashed', linewidth=2, label='Mean')


# In[50]:


df['actor_2_facebook_likes']=df['actor_2_facebook_likes'].fillna(ac['actor_2_facebook_likes'].mean())


# In[51]:


csv_file_path=r"C:\Users\babyk\OneDrive\Documents\projects\imdb\clean_data.csv"


# In[52]:


df.to_csv(csv_file_path, index=False)


# In[53]:


df.isnull().sum()


# In[54]:


df['language']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




