#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv("Heart.csv")
df.head()


# In[5]:


df = df.drop_duplicates()


# In[6]:


# Count ,min,max ,etc of each column
df.describe()


# In[7]:


# Information about each column data
df.info()


# In[8]:


#Finding null values in each column
df.isna().sum()


# In[9]:


#Data Integration


# In[10]:


df.head()


# In[11]:


df.fbs.unique()


# In[14]:


subSet1 = df[['age','cp','chol','thalach']]


# In[41]:


subSet2 = df[['exang','slope','target']]


# In[42]:


merging = subSet1.merge(right=subSet2,how='cross')
merging.head()


# In[43]:


#Error Correcting


# In[44]:


df.columns


# In[27]:


def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    threshold = 1.5 * IQR
    outlier_mask = (column < Q1 - threshold) | (column > Q3 + threshold)
    return column[~outlier_mask]


# In[46]:


# Remove outliers for each column using a loop
col_name = ['cp','thalach','exang','oldpeak','slope','ca']
for col in col_name:
    df[col] = remove_outliers(df[col])


# In[33]:


plt.figure(figsize=(10, 6))  # Adjust the figure size if needed

for col in col_name:
    sns.boxplot(data=df[col])
    plt.title(col)
    plt.show()


# In[36]:


#data split


# In[47]:


# splitting data using train test split
x = df[['cp','thalach','exang','oldpeak','slope','ca']]
y = df.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[50]:


#Data transformation


# In[51]:


from sklearn.preprocessing import StandardScaler


# In[52]:


scaler = StandardScaler()


# In[55]:


x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[56]:


#Data model building


# In[57]:


y_train= np.array(y_train).reshape(-1, 1)
y_test= np.array(y_test).reshape(-1, 1)


# In[58]:


y_train.shape


# In[59]:


model = LogisticRegression()
model.fit(x_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test_scaled)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[60]:


#Classification model using Decision Tree
from sklearn.tree import DecisionTreeClassifier
tc=DecisionTreeClassifier(criterion='entropy')
tc.fit(x_train_scaled,y_train)
y_pred=tc.predict(x_test_scaled)

print("Training Accuracy Score :",accuracy_score(y_pred,y_test))
print("Training Confusion Matrix  :",confusion_matrix(y_pred,y_test))


# In[ ]:




