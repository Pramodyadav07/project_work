#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install kaggle')

#install kaggle library


# In[2]:


get_ipython().system(' pip install opendatasets')


# In[3]:


import opendatasets as od 
#Install library to dataset directly from kaggle


# In[4]:


od.download("https://www.kaggle.com/datasets/kazanova/sentiment140")

#download dataset directly from kaggle


# In[19]:


#loading the data from csv file to pandas dataframe
twitter_data = pd.read_csv('sentiment.csv', encoding = 'ISO-8859-1')


# In[20]:


# checking the number of rows and columns
twitter_data.shape


# In[61]:


#print first 5 rows of data
twitter_data.head()


# In[ ]:


# as the names of columns are not read so name the columns and assign to it and read again


# In[22]:


column_names=['target','id','date','flag','user','text']
twitter_data = pd.read_csv('sentiment.csv', names=column_names, encoding = 'ISO-8859-1')


# In[23]:


twitter_data.head()


# In[25]:


# checking for missing null values
twitter_data.isnull().sum()


# In[26]:


#checking the distribution of target column

twitter_data['target'].value_counts()


# In[ ]:


We can see there are equal distribution data 8L positive and 8L negative comments


# In[29]:


#Convert target value 4 to 1

twitter_data.replace({'target':{4:1}},inplace =True)


# In[30]:


twitter_data['target'].value_counts()


# In[ ]:


# zero means negative tweet and 1 means positive tweet


# In[33]:


# now we will import all required libraries for processing
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[34]:


import nltk
nltk.download('stopwords')


# In[ ]:


#stemming- process of reducing the word to root word
# learing, learned --- learn root word


# In[35]:


port_stem = PorterStemmer()


# In[36]:


def steamming(content):
    steammed_content = re.sub('[^a-zA-Z]',' ',content)
    steammed_content = steammed_content.lower()
    steammed_content = steammed_content.split()
    steammed_content = [port_stem.stem(word) for word in steammed_content if not word in stopwords.words('english')]
    steammed_content = ' '.join(steammed_content)
    
    return steammed_content


# In[37]:


twitter_data['steammed_content'] = twitter_data['text'].apply(steamming)


# In[38]:


twitter_data.head()


# In[39]:


#Seperating the data and label
X = twitter_data['steammed_content'].values
Y = twitter_data['target'].values


# In[40]:


print(X)


# In[41]:


print(Y)


# In[42]:


#splitting data to train and test data 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[43]:


print(X.shape, X_train.shape, X_test.shape)


# In[44]:


#Converting the actual data to numerical data

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# In[45]:


print(X_train)


# In[47]:


print(X_test)


# In[50]:


#traing the machine learing model
# logistic Regrassion

model = LogisticRegression(max_iter=1000) #this is used in pickle further to save trained model


# In[51]:


model.fit(X_train, Y_train)


# In[ ]:


#model Evaluation for Accuracy Score


# In[52]:


# Accuracy score on the training data
X_train_prediction =model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)


# In[53]:


print("Accuracy of training data model is:",training_data_accuracy)


# In[ ]:


#Accuracy score of 81% means out of 100, 9 tweets are misjudged by model


# In[54]:


# Accuracy score on the test data
X_test_prediction =model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Accuracy of test data model is:",test_data_accuracy)


# In[ ]:


#Accuracy score of 77% means out of 100, 23 tweets are misjudged by model


# In[ ]:


#overfitting- if there is too much difference in accuracy score of train and test data then the model is overfitted


# In[55]:


#Saving the trained model
import pickle


# In[56]:


filename="trained_model.sav"
pickle.dump(model,open(filename, 'wb'))


# In[57]:


#How to use saved model for future predictions

loaded_model = pickle.load(open('trained_model.sav','rb'))


# In[58]:


X_new = X_test[200]
print(Y_test[200])

prediction = loaded_model.predict(X_new)
print(prediction)

if (prediction[0]==0):
    print ("The Tweet is Negative")
else:
    print ("The Tweet is Positive")


# In[59]:


X_new = X_test[10]
print(Y_test[10])

prediction = loaded_model.predict(X_new)
print(prediction)

if (prediction[0]==0):
    print ("The Tweet is Negative")
else:
    print ("The Tweet is Positive")


# In[ ]:





# In[ ]:




