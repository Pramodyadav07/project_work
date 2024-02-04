#!/usr/bin/env python
# coding: utf-8

# 1. At the begining we will import required packages for the analyasis, and load data using Pandas and display first five rows of data by uding df.head function

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('googleplaystore.csv')
df.head()


# 2. For Checking null values in the data and to get the number of null values for each column we will use following command

# In[2]:


df.isnull().sum()


# 3. To Drop records with nulls in any of the columns present in data. We will write a commaond to remove all null values from all columns and display the shape of data before and after removing null values.

# In[3]:


print("Frame Size before : " , df.shape)
df.dropna(subset=['Rating', 'Type', 'Content Rating', 'Current Ver','Android Ver'],axis=0, inplace=True)
print("Frame Size After : " , df.shape)
df.isnull().sum(axis=0)


# 4. Variables are in incorrect type (string) and inconsistent format. 
#    We need to fix them: Size column has sizes in Kb as well as Mb. 
#    To analyze, we need to convert these to numeric.
#    Extract the numeric value from the column for that we will use the command

# In[4]:


df=df[-df['Size'].str.contains('Var')] #we need to find and drop some unreadble charachetrs from Size column.


# In[5]:


df.loc[:,"SizeNum"] = df.Size.str.rstrip("Mk+")#we use this command to drop Characters from all rows of Size colmn and store it by new variable SizeNum


# In[6]:


df["SizeNum"] #display the SizeNum column where all "Mk+" are removed


# In[6]:


df.SizeNum = pd.to_numeric(df["SizeNum"]) #we change the type of data from str to Numeric for analysis in SizeNum column


# In[7]:


df.SizeNum.dtype #shows the type of SizeNum changed from str to float (Numeric)


# 4. 1.2 Multiply the value by 1,000,if the size is mentioned in Mb.

# In[8]:


df['SizeNum']=np.where(df.Size.str.contains('M'),df.SizeNum*1000, df.SizeNum)


# In[9]:


df.Size=df.SizeNum # Size no more needed, replace it with SizeNum and drop SizeNum
df.drop('SizeNum',axis=1,inplace=True)


# 4. 2. Reviews is a numeric field that is loaded as a string field. Convert it to numeric (int/float). After conversion need to check its data type

# In[10]:


df.Reviews = pd.to_numeric(df.Reviews)


# In[11]:


df.Reviews.dtype


# 4. 3.In Installs column field is currently stored as string and has values like 1,000,000+. 
#    We have tp remove + sign for that we use following commands

# In[12]:


df['Installs']=df.Installs.str.replace("+","")


# In[13]:


df.Installs=df.Installs.str.replace(",","") # remove "," 
df.Installs=pd.to_numeric(df.Installs) #change data type
df.Installs.dtype #display dtype after conversion


# 4. 4 Price field is a string and has '$'symbol. Remove ‘$’ sign, and convert it to numeric.

# In[14]:


df.Price=df.Price.str.replace("$","")
df.Price=pd.to_numeric(df.Price)
df.Price.dtype


# 5. 1. Average rating should be between 1 and 5 as only these values are allowed on the play store. Drop the rows that have a value outside this range. To drop such values outside this range we will apply this command for Ratings column 

# In[15]:


df=df[(df.Rating>=1) & (df.Rating<=5)]


# In[16]:


df["Rating"]


# 5. 2. Reviews should not be more than installs as only those who installed can review the app. If there are any such records, drop them. We will check the length before drop and after drop

# In[17]:


len(df.index)


# In[18]:


df.drop(df.index[df.Reviews>df.Installs],axis=0,inplace=True)
len(df.index)


# For free apps (type = “Free”), the price should not be >0. Drop any such rows. we will check for free apps and drop the rows in which price is more than Zero

# In[19]:


df[(df["Type"]=="Free") & (df["Price"]>0)]


# #There are no free apps with price > 0.

# 5. Performing univariate analysis 
#    we will Boxplot for Price(price on X axis). We will check for are there any outliers? for that we will import seaborn and statistics liabraries.

# In[20]:


import seaborn as sns


# In[21]:


ax = sns.boxplot(x='Price', data=df)


# • Insights: Most of Price values are less than 50 while there is some near concentration around 80 greater than 100 may be considered outliers
# • Consider 3 STD as range of outliers

# In[22]:


import statistics as stc
price_std=stc.stdev(df.Price)
price_std


# In[23]:


price_mean=stc.mean(df.Price)
price_mean


# In[24]:


price_outlier_uplimit=price_mean+3*price_std
price_outlier_uplimit


# In[25]:


print("# of upper outliers is ",len(df[(df.Price>price_outlier_uplimit) ]))


# 5. Performing univariate analysis: 
#    we will Boxplot for Reviews(Revievws on X axis). We will check for Are there any apps with very high number of reviews?

# In[26]:


sns.boxplot(x='Reviews',data=df)


# • Insights: Most Apps get about more than 2M review. Roughly, greater than 2M can be considered outliers
# • Consider 3 STD as range of outliers

# In[28]:


rev_std=stc.stdev(df.Reviews)
rev_std


# In[29]:


rev_mean=stc.mean(df.Reviews)
rev_mean


# In[30]:


rev_outlier_uplimit=rev_mean+3*rev_std
rev_outlier_uplimit


# In[31]:


rev_outlier_downlimit=rev_mean-3*rev_std
rev_outlier_downlimit


# In[32]:


print("# of upper outliers is ",len(df[(df.Reviews>rev_outlier_uplimit)]))


# 5. Histogram for Rating 
#    How are the ratings distributed? We will Boxplot for Ratings ( on X axis) and we will check for Is it more toward higher ratings? 

# In[33]:


sns.histplot(x='Rating',data=df)


# 5.4 Histogram for Size
# We will plot histogram for size using following command.

# In[34]:


sns.histplot(x='Size',data=df,log_scale=True)


# 6. Outlier treatment: 
# 
# Price: From the box plot, it seems like there are some apps with very high price. A price of $200 for an application on the Play Store is very high and suspicious!
# 
# Check out the records with very high price
# 
# Is 200 indeed a high price?
# 
# Drop these as most seem to be junk apps

# In[35]:


df[df.Price>=200]


# In[36]:


print("# of Apps with price >= 200 = ",len(df[(df.Price>=200) ]))


# It is very high and very far than the mean. Drop these as most seem to be junk apps

# In[37]:


df.drop(df.index[(df.Price>=200)], inplace=True)
len(df.index)


# 6.2 Reviews: Very few apps have very high number of reviews. These are all star apps that don’t help with the analysis and, in fact, will skew it. We will Drop records having more than 2 million reviews.

# In[38]:


df.drop(df.index[(df.Reviews>=2000000)], inplace=True)
len(df.index)


# 6.3 Installs: There seems to be some outliers in this field too. Apps having very high number of installs should be dropped from the analysis.
# 6.3.1 We will Find out the different percentiles – 10, 25, 50, 70, 90, 95, 99 after that drop 

# In[39]:


install_10_perc=np.percentile(df.Installs, 10)
install_10_perc


# 6.3.2 Decide a threshold as cutoff for outlier and drop records having values more than that

# In[40]:


install_25_perc=np.percentile(df.Installs, 25)
install_25_perc


# In[41]:


install_50_perc=np.percentile(df.Installs, 50)
install_50_perc


# In[42]:


install_70_perc=np.percentile(df.Installs, 70)
install_70_perc


# In[43]:


install_90_perc=np.percentile(df.Installs,90)
install_90_perc


# In[44]:


install_95_perc=np.percentile(df.Installs,95)
install_95_perc


# In[45]:


install_99_perc=np.percentile(df.Installs,99)
install_99_perc


# In[46]:


sns.histplot(data=df,x='Installs',log_scale=True)


# • My decision is to drop values > percentile of 99(Almost 3 STD)

# In[47]:


print("As result, ",len(df[df.Installs >= install_99_perc])," will be dropped")


# In[48]:


df.drop(df.index[df.Installs >= install_99_perc],inplace=True)
len(df.index)


# 7. Bivariate analysis: Let’s look at how the available predictors relate to the variable of interest, i.e., our target variable rating. Make scatter plots (for numeric features) and box plots (for character features) to assess the relations between rating and the other features For each of the plots , note down your observation.
# 7.1. Make scatter plot/joinplot for Rating vs. Price

# In[49]:


sns.jointplot(data=df,y='Rating',x='Price')


# 7.2. What pattern do you observe? Does rating increase with price? Most of Apps with high price get > 3 Rating but this is because majority of apps are with low price. In addition most apps get rating > 3. Concusion: We cannot consider there is a good relationship between Rating and Price. It seems Price has limited impact on Rating.
# 
# 7.3. Make scatter plot/joinplot for Rating vs. Size

# In[50]:


sns.jointplot(data=df,y='Rating',x='Size')


# 7.4. Are heavier apps rated better? Again if we look to the area where most apps rated (greater than 3) almost the points are evenly distributed The relationship between Size and rating is very weak
# 
# 7.5. Make scatter plot/joinplot for Rating vs. Reviews

# In[51]:


sns.jointplot(data=df,y='Rating',x='Reviews')


# 7.6. Does more review mean a better rating always?
# Although the relationship seems also not so strong, but we can notice that there is some concentration of apps with higher reviews in high rating area. It seems good apps get more reviews than
# others
# 
# 7.7 Make boxplot for Rating vs. Content Rating

# In[52]:


df['Content Rating'].unique()


# In[53]:


sns.boxplot(data=df,x='Rating',y='Content Rating')


# 7.8. Is there any difference in the ratings? Are some types liked better? Apps of Adults only 18+ has higher rating than others while Mature 17+ gets less liks. Others seem to be closed. Content has good impact on 
# 
# 7.9. Make boxplot for Ratings vs. Category

# In[54]:


a4_dims = (11.7, 10.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.boxplot(data=df,x='Rating',y='Category',ax=ax)


# 7.10. Which genre has the best ratings?
# The best genre is Events

# 8.1. Reviews and Install have some values that are still relatively very high. Before building a linear regression model, you need to reduce the skew. Apply log transformation (np.log1p) to Reviews and Installs.

# In[55]:


inp1=df.copy()
inp1.Reviews=inp1.Reviews.apply(np.log1p)


# In[56]:


inp1.Installs=inp1.Installs.apply(np.log1p)


# In[57]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# 8.2.Drop columns App, Last Updated, Current Ver, and Android Ver. These variables are not
# useful for our task.

# In[58]:


inp1.drop(columns=['App','Last Updated','Current Ver','Android Ver'],inplace=True)


# In[59]:


inp1.shape


# 8.3. Get dummy columns for Category, Genres, and Content Rating. This needs to be done as the models do not understand categorical data, and all data should be numeric. Dummy encoding is one way to convert character fields to numeric. Name of dataframe should be inp2.

# In[60]:


inp2= pd.get_dummies(inp1)


# In[61]:


inp2.shape


# 9. Train test split and apply 70-30 split. Name the new dataframes df_train and df_test.

# 10. Separate the dataframes into X_train, y_train, X_test, and y_test.

# In[62]:


data = inp2.drop(columns='Rating')
data.shape


# In[63]:


target = pd.DataFrame(inp2.Rating)
target.shape


# In[64]:


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=3)
print("x_train shape is ", x_train.shape)
print("y_train shape is ", y_train.shape)
print("x_test shape is ", x_test.shape)
print("y_test shape is ", y_test.shape)


# 11. Model building Use linear regression as the technique Report the R2 on the train set

# In[65]:


model=LinearRegression()
model.fit(x_train, y_train)


# In[66]:


train_pred=model.predict(x_train)


# In[67]:


print("R2 value of the model(by train) is ", r2_score(y_train, train_pred))


# 12. Make predictions on test set and report R2.

# In[68]:


test_pred=model.predict(x_test)


# In[69]:


print("R2 value of the model(by test) is ", r2_score(y_test, test_pred))


# In[ ]:




