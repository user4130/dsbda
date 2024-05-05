#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('dataset_Facebook.csv',delimiter=';')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


subset_1 = df[['like','share']]
subset_1


# In[10]:


subset_2 = df[['comment','Type','like']]
subset_2


# In[11]:


merging  = pd.merge(subset_1, subset_2)
merging


# In[12]:


merging.sort_values(by=['like'],ascending=False)


# In[13]:


merging.transpose()


# In[18]:


df.shape


# In[15]:


df.Type.unique()


# In[16]:


pd.melt(df, id_vars =['Type'], value_vars =['comment'])


# In[19]:


# Define a dictionary containing employee data
data1 = {
'key': ['K0', 'K1', 'K2', 'K3'],
'Name':['Jai', 'Princi', 'Gaurav', 'Anuj'],
'Age':[27, 24, 22, 32],}
# Define a dictionary containing employee data
data2 = {
'key': ['K0', 'K1', 'K2', 'K3'],
'Address':['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj'],
'Qualification':['Btech', 'B.A', 'Bcom', 'B.hons']}
# Convert the dictionary into DataFrame
data1 = pd.DataFrame(data1)
# Convert the dictionary into DataFrame
data2 = pd.DataFrame(data2)

# print(df, "\n\n", df1)
res = pd.merge(data1, data2, on='key')
res


# In[ ]:




