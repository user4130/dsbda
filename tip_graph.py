#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv('tip.csv')


# In[5]:


df.head()


# In[6]:


df.info()


# In[8]:


plt.figure(figsize=(10,6))
plt.title('line plot')
sns.lineplot(df,x='total_bill',y='tip')
plt.show()


# In[10]:


plt.figure(figsize=(10,6))
plt.title('Bar Plot')
sns.barplot(x=df['sex'],y=df['tip'])
plt.show()


# In[11]:


corr=df.corr()
corr


# In[12]:


plt.figure(figsize = (10,6))
plt.title('Heat Map')
sns.heatmap(corr,annot=True)
plt.show()


# In[13]:


plt.imshow(corr)


# In[14]:


plt.figure(figsize=(5,3))
plt.title('scatter plot')
sns.scatterplot(x=df['total_bill'],y=df['tip'])
plt.show()


# In[15]:


plt.figure(figsize=(5,3))
plt.title('Histogram')
sns.histplot(x=df['total_bill'])
plt.show()


# In[16]:


plt.figure(figsize=(5,3))
plt.title('box plot')
sns.boxplot(y=df['tip'])
plt.show()


# In[19]:


plt.figure(figsize = (5,3))
plt.title('violine plot')
sns.violinplot(x=df['day'],y=df['tip'])
plt.show()


# In[22]:


df.columns


# In[26]:


plt.title('Time Series Plot of Total Bill')
plt.xlabel('time')
plt.ylabel('total_bill')
sns.lineplot(data=df, x='time', y='total_bill')
plt.show()


# In[ ]:




