#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('adult_dataset.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


subset_1 = df[['workclass','education','capital-gain']]
subset_1


# In[7]:


subset_2 = df[['race','native-country']]
subset_2


# In[9]:


merging = pd.concat([subset_1,subset_2],axis=1)
merging


# In[10]:


merging.sort_values(by=['capital-gain'],ascending=False)


# In[11]:


merging.transpose()


# In[13]:


df.shape


# In[14]:


pd.melt(df, id_vars =['education'], value_vars =['capital-gain'])


# In[ ]:




