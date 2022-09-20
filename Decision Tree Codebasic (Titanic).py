#!/usr/bin/env python
# coding: utf-8

# ## Exercise: Build decision tree model to predict survival based on certain parameters
# 
# 

# In this file using following columns build a model to predict if person would survive or not,
# 
# 1. Pclass
# 2. Sex
# 3. Age
# 4. Fare
# 
# Calculate score of your model

# ## Loading Data and Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


titanic = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/9_decision_tree/Exercise/titanic.csv')

titanic.head()


# ### Here we have to create model by using some of the variables like(Pclass,Sex,Age,Fare and Survived)

# In[3]:


titanic = titanic.drop(['PassengerId','Name','SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns')

titanic.head()


# In[4]:


titanic = titanic[['Sex', 'Age', 'Pclass', 'Fare', 'Survived']]

titanic.head()


# In[5]:


titanic.shape


# In[6]:


titanic.columns


# In[7]:


print(titanic.isnull().sum())


# ### Replacing missing values with Mean

# In[8]:


titanic["Age"] = titanic["Age"].replace(np.NaN, titanic["Age"].mean())

print(titanic.isnull().sum())


# ### Rounding Age znd Fare column upto 1 decimal

# In[9]:


titanic['Age'] = titanic['Age'].round(1)


# In[10]:


titanic['Fare'] = titanic['Fare'].round(1)


# In[11]:


titanic.head()


# ### Identify independent variable and Target variable

# In[12]:


y = titanic['Survived']

X = titanic.drop('Survived', axis='columns')


# In[13]:


X.head()


# In[14]:


y.head()


# ### Train Test and Split

# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.70, random_state = 0)


# In[16]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[17]:


X_train.head()


# ### Data Preprocessing (As there is one categorical column in ordinal format)

# In[33]:


from sklearn.preprocessing import LabelEncoder

Sex_le = LabelEncoder()

X_train['Sex'] = Sex_le.fit_transform(X_train['Sex'])
X_test['Sex'] = Sex_le.fit_transform(X_test['Sex'])


# In[34]:


X_train.head()


# In[35]:


X_test.head()


# ### Training The Decision Tree Model

# In[20]:


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)


# ### Prediction

# In[36]:


y_pred = classifier.predict(X_test)


# In[ ]:





# - **Query_point from Train Data = Is Female of Age=29.7, Pclass=3, and Fare=14.4583 Survived or not??**

# In[26]:


X_train.head()


# In[28]:


classifier.predict([[0,29.7,3,14.5]])


# In[29]:


titanic.loc[578]


# - **Query_point from Test Data = Is Female of Age=29.0, Pclass=3, and Fare=15.2 Survived or not??**

# In[27]:


X_test.head()


# In[30]:


classifier.predict([[0, 29.7, 1, 146.5]])


# In[31]:


titanic.loc[31]


# ### Accuracy of the Model

# In[38]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[ ]:




