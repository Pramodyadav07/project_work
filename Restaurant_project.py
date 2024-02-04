#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# loading the dataset in which the restaurent details are given, and display first 10 rows
df= pd.read_excel('G:\simplilearn\Capstone project\Datasets\data.xlsx')
df.head(10)


# In[4]:


#Checking for any duplicate values present in data
print('Any Duplicated Rows ? :' , df.duplicated().any())


# In[5]:


# loading the second dataset in which country info is given
df1= pd.read_excel("G:\simplilearn\Capstone project\Datasets\Country-Code.xlsx")
df1.head()


# In[6]:


# Checking for any duplicate values in data
print('Any Duplicated Rows ? :' , df1.duplicated().any())


# In[7]:


# Merging the two datasets in one dataframe.
df_rest=pd.merge(df,df1,on='Country Code',how='left')
df_rest.head()


# In[8]:


# Renaming the columns name given in dataset we are replacing space by _ here
df_rest.columns = df_rest.columns.str.replace(' ','_')
df_rest.columns


# In[9]:


# Checking for any null values in dataframe
df_rest. isnull().sum()


# In[10]:


#here the restaurant name is missing, we dropped the record and reset the index.
df_rest.dropna(axis=0,subset=['Restaurant_Name'],inplace=True)
df_rest.reset_index(drop=True,inplace=True)


# In[11]:


df_rest. isnull().sum()


# In[12]:


df_rest[df_rest['Cuisines'].isnull()] # we found there are 9 cuisines that have null values


# In[13]:


#As there were 9 records without cuisines, we have replace the null values with Others.
df_rest['Cuisines'].fillna("Others", inplace=True)


# In[15]:


df_rest.isnull().sum()


# In[16]:


df_rest.shape # to get final number of rows and column in dataset
(9550, 20)


# In[19]:


#Explore the geographical distribution of the restaurants.
restaurant_counts = pd.DataFrame(df_rest.Country.value_counts()).rename({'Country':'Freq'}, axis= 1)
restaurant_counts


# In[20]:


# we will now plot the graph on X axis "Country" and on Y axis  restaurant_counts
sns.countplot(x = 'Country', data = df_rest, order = restaurant_counts.index)
plt.xticks(size = 5, rotation = 20)
plt.xlabel('Country',size = 16)
plt.show()


# From the above table and chart We observe that India has the highest number of restaurants with 8651 restaurants and USA is 
# number 2 with 434 restaurants and Canada has the least number of restaurants with 4 restaurents.

# In[21]:


restaurant_counts = pd.Series()
restaurant_counts['India'] = len(df_rest[df_rest.Country == 'India'])
restaurant_counts['Others'] = len(df_rest[df_rest.Country != 'India'])
restaurant_counts.plot.pie(radius = 2,autopct = '%1.1f%%' , textprops = {'size':15 }, explode
= [0.1,0.1], shadow = True, cmap ='Set2')
plt.xticks(size = 12, rotation = 10)
plt.ylabel('')
plt.show()


# From above Pi chart we can say that The country India alone has the 90.6% of total restaurent count and other 14 countries 
# together holds only 9.4%.

# In[22]:


##Finding out the cities with maximum / minimum number of restaurants
city_dist = df_rest.groupby(['Country','City']).agg(Count = ('Restaurant_ID','count'))
city_dist.sort_values(by='Count',ascending=False)

# we see that new Delhi has the maximum restaurant with 5473 we observe that multiple cities have only one restaurant.
# In[23]:


min_cnt_rest = city_dist[city_dist['Count']==1]
min_cnt_rest.info()
min_cnt_rest


# From above table we observe that There are 46 cities in 7 different countries with 1 restaurants

# In[25]:


#Restaurant franchise is a thriving venture. So, it becomes very important to explore the franchise with most national presence.
plt.figure(figsize = (15,10))
vc=df_rest.Restaurant_Name.value_counts()[:10]
g = sns.barplot(y = vc.index, x = vc.values, palette = 'Set2')
g.set_yticklabels(g.get_yticklabels(),fontsize = 13)
for i in range(10):
    value = vc[i]
g.text(x = value - 2,y = i +0.125 , s = value, color='black', ha="center",fontsize = 15)
g.set_xlabel('Count', fontsize = 15)
g.set_title('Restaurant Presence', fontsize = 30, color = 'darkred')
plt.show()


# Cafe Coffe day has most national presesence with count of 83 numbers.

# In[34]:


df_rest1 = df_rest.copy()
df_rest1.columns


# In[35]:


dummy = ['Has_Table_booking','Has_Online_delivery']
df_rest1 = pd.get_dummies(df_rest1,columns=dummy,drop_first=True)
df_rest1.head()
# 0 indicates 'NO'
# 1 indicates 'YES'


# In[36]:


#Ration between restaurants allowing table booking and those which dont
table_booking = df_rest1[df_rest1['Has_Table_booking_Yes']==1]['Restaurant_ID'].count()
table_nbooking =df_rest1[df_rest1['Has_Table_booking_Yes']==0]['Restaurant_ID'].count()
print('Ratio between restaurants that allow table booking vs. those that do not allow table booking: ',
round((table_booking/table_nbooking),2))


# In[46]:


#Pie chart to show percentage of restaurants which allow table booking and those which don't

labels = 'No Table Booking', 'Table Booking'
fig, axes = plt.subplots(figsize=(7, 20))

df_rest1['Has_Table_booking_Yes'].value_counts().plot.pie(ax=axes, labels=labels, autopct='%0.1f%%', radius=1.25,
                                                         wedgeprops={'width': 0.4},
                                                         textprops={'size': 18})

axes.set_title('Table Booking Vs No Table Booking\n', fontsize=16)
axes.set_ylabel('')
plt.show()


# In[ ]:


From observing the above chart we can say that only 12.1% Restaurants provide table booking facility


# In[50]:


# Find out the percentage of restaurants providing online delivery
labels = 'No Online Delivery','Online Delivery'
fig, axes = plt.subplots(figsize=(7, 20))

df_rest1['Has_Online_delivery_Yes'].value_counts().plot.pie(ax=axes, labels=labels, autopct='%0.1f%%', radius=1.25,
                                                         wedgeprops={'width': 0.4},
                                                         textprops={'size': 18})

axes.set_title('Online Delivery\n', fontsize = 16)
axes.set_ylabel('')
plt.show()                                                                                                                                                                                                                                                       


# From observing the above chart we can say that only 25.7% Restaurants provide online deleivery

# In[52]:


dc= df_rest.pivot_table(index = ['Has_Online_delivery'],values = 'Votes', aggfunc = 'sum')
dc


# From observing the above table we can say that the difference between number of votes for restaurents that dont deliver and 
# deliver is 462048 vote.

# In[53]:


dc['Perc'] = (dc.Votes / dc.Votes.sum() *100).round(2)
sns.barplot(x = dc.index, y = dc.Votes,)
plt.xticks( rotation = 0, fontsize = 14)
plt.xlabel('')
for i in range(len(dc)):
    plt.annotate(str(dc.Perc.iloc[i]) + '%',xy = (i-0.15, int(dc.Votes.iloc[i]/2)), fontsize = 12 )
plt.ylabel('No. of Votes',fontsize = 20)


# In[56]:


# What are the top 10 cuisines served across cities?
l = []
for i in df_rest.Cuisines.str.split(','):
    l.extend(i)
s = pd.Series([i.strip() for i in l])
plt.figure(figsize = (15,5))
sns.barplot(x = s.value_counts()[:10].index, y = s.value_counts()[:10] )
for i in range(10):
    plt.annotate(s.value_counts()[i], xy = (i-0.15,s.value_counts()[i]+50),fontsize = 14)
plt.ylim(0, round(s.value_counts()[0]+300))
plt.show()


# In[57]:


#What is the maximum and minimum no. of cuisines that a restaurant serves? Also,what is the relationship between No. of
#cuisines served and Ratings
df_rest['no_cuisines']=df_rest.Cuisines.str.split(',').apply(len)
df_rest['no_cuisines']


# In[59]:


plt.figure(figsize = (15,5))
vc = df_rest.no_cuisines.value_counts()
sns.countplot(x='no_cuisines', data=df_rest, order = vc.index)
for i in range(len(vc)):
    plt.annotate(vc.iloc[i], xy = (i-0.07,vc.iloc[i]+10), fontsize = 12)
plt.show()


# the maximum no of cuisines served by a single restaurant is 8 most of the restaurant are serving atleast 2 or 1 cuisine

# In[60]:


# No of Cuisines vs Rating
df_rest['Rating_cat'] = df_rest['Aggregate_rating'].round(0).astype(int)
fusion_rate = df_rest.loc[df_rest.Aggregate_rating >0,['no_cuisines', 'Rating_cat','Aggregate_rating']].copy()
fusion_rate


# In[61]:


sns.regplot(x='no_cuisines',y='Aggregate_rating',data=fusion_rate)


# In[62]:


fusion_rate[['no_cuisines', 'Aggregate_rating']].corr()


# In[63]:


sns.barplot(x='no_cuisines',y='Aggregate_rating',data=fusion_rate)


# From the above graphs we can observe that the number of cuisines and Aggregate_rating has positive corelation We also observe 
# that higher the number of cuisines higher the rating.

# In[64]:


# Explore how ratings are distributed overall.
plt.figure( figsize = (15, 4))
sns.countplot(x='Aggregate_rating', data = df_rest[df_rest.Aggregate_rating !=0] ,palette = 'magma')
plt.tick_params('x', rotation = 70)


# In[65]:


# Rating Vs Delevery Options(Has_Online_delivery, Yes, No)
plt.figure(figsize=(20,6))
sns.countplot(data=df_rest[df_rest.Aggregate_rating !=0],x='Aggregate_rating',hue='Has_Online_delivery',palette='viridis')
plt.show()


#  From observing the above chart we can say that the delevery options can be a factor to decide the rating of restaurent.

# In[66]:


# Discuss the cost(Average_Cost_for_two) vs the other variables
#Overall Cost Destribution
plt.figure(figsize = (15,5))
sns.distplot(df_rest[df_rest.Average_Cost_for_two != 0].Average_Cost_for_two)
plt.show()


# In[67]:


# cost(Average_Cost_for_two) Vs rating
df_rest['Average_Cost_for_two_cat']= pd.cut(df_rest[df_rest.Average_Cost_for_two != 0].Average_Cost_for_two,
bins = [0, 200, 500, 1000, 3000, 5000,10000],
labels = ['<=200', '<=500', '<=1000', '<=3000', '<=5000', '<=10000'])


# In[68]:


f = plt.figure(figsize = (20,10))
ax = plt.subplot2grid((2,5), (0,0),colspan = 2)
sns.countplot(x=df_rest['Average_Cost_for_two_cat'], ax = ax, palette = sns.color_palette('magma', 7))
ax.set_title('Average Cost for 2')
ax.set_xlabel('')
ax.tick_params('x', rotation = 70)
ax = plt.subplot2grid((2,5), (0,2), colspan = 3)
sns.boxplot(x = 'Average_Cost_for_two_cat', y = 'Aggregate_rating', data =df_rest, ax = ax, palette = sns.color_palette('magma', 7))
plt.show()


# From Observing the above graph we can say that as the average cost for 2 increases the aggregate rating also increases

# In[69]:


# Price Range Vs Rating
count = df_rest['Price_range'].value_counts().reset_index()
count.columns = ['Price_range', 'Count']


# In[70]:


f = plt.figure(figsize = (20,10))
ax = plt.subplot2grid((2,5), (1,0),colspan = 2)
sns.barplot(x = 'Price_range', y = 'Count', data = count, ax=ax, palette = sns.color_palette('magma', 5))
ax.set_title('Price Range')
ax.set_xlabel('')
ax = plt.subplot2grid((2,5), (1,2), colspan = 3)
sns.boxplot(x='Price_range', y ='Aggregate_rating', data = df_rest, ax = ax,palette = sns.color_palette('magma', 5))
plt.subplots_adjust(wspace = 0.3, hspace = 0.4,)
plt.suptitle('Price Range & Rating Distribution', size = 30)
plt.show()


# From Observing the above chart we can say as the price range increases the rating of restaurent is also increases.

# In[71]:


#Aggregate Rating vs Votes
sns.scatterplot(data=df_rest,x='Aggregate_rating',y='Votes', palette ='Set2')
df_rest[['Votes','Aggregate_rating']].corr()


# We see that there is no single variable that affects the rating strongly, however table booking,online delivery,avg price for 
# two and price range,number of votes do play a part in affecting the rating of a restaurant.

# In[72]:


df_rest.to_csv("final_cleaned_data",index=False)


# In[ ]:




