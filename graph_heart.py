#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Heart.csv')


# In[11]:


df.head()


# In[14]:


plt.figure(figsize = (5, 3))
plt.title('Line Chart')
sns.lineplot(x=df['age'] , y=df['chol'])
plt.show


# In[19]:


plt.figure(figsize=(10,6))
plt.title('Bar Plot')
sns.barplot(x=df['age'],y=df['chol'])
plt.show()


# In[20]:


corr=df.corr()
corr


# In[24]:


plt.figure(figsize = (10,6))
plt.title('Heat Map')
sns.heatmap(corr,annot=True)
plt.show()


# In[34]:


plt.imshow(corr)


# In[26]:


plt.figure(figsize=(5,3))
plt.title('scatter plot')
sns.scatterplot(x=df['age'],y=df['chol'])
plt.show()


# In[33]:


plt.figure(figsize=(5,3))
plt.title('Histogram')
sns.histplot(x=df['age'])
plt.show()


# In[36]:


plt.figure(figsize=(5,3))
plt.title('box plot')
sns.boxplot(y=df['chol'])
plt.show()


# In[39]:


plt.figure(figsize = (5,3))
plt.title('violine plot')
sns.violinplot(y=df['chol'])
plt.show()


# In[46]:





# In[ ]:





# In[ ]:




