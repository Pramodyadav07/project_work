#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install spotipy')


# In[1]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd


# In[2]:


client_credentials_manager = SpotifyClientCredentials(client_id="get it from spotify", 
                                                      client_secret="get it from spotify")


# In[3]:


sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


# In[15]:


playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DWZNJXX2UeBij"


# In[16]:


# spliting the string to get the required object
playlist_URI = playlist_link.split("/")[-1]


# In[6]:


data = sp.playlist_tracks(playlist_URI)


# In[7]:


#extracting different details from the data like ID, name, release_date, 
#total_tracks etc using indexing and slicing
data['items'][0]['track']['album']['id']


# In[8]:


data['items'][0]['track']['album']['name']


# In[9]:


data['items'][0]['track']['album']['release_date']


# In[10]:


data['items'][0]['track']['album']['total_tracks']


# In[14]:


data['items'][0]['track']['album']['external_urls']


# In[12]:


album_list = []
for row in data['items']:
    album_id = row['track']['album']['id']
    album_name = row['track']['album']['name']
    album_release_date = row['track']['album']['release_date']
    album_total_tracks = row['track']['album']['total_tracks']
    album_url = row['track']['album']['external_urls']['spotify']
    album_element = {'album_id':album_id,'name':album_name,'release_date':album_release_date,
                        'total_tracks':album_total_tracks,'url':album_url}
    album_list.append(album_element)


# In[19]:


album_list


# In[20]:


artist_list = []
for row in data['items']:
    for key, value in row.items():
        if key == "track":
            for artist in value['artists']:
                artist_dict = {'artist_id':artist['id'], 'artist_name':artist['name'], 'external_url': artist['href']}
                artist_list.append(artist_dict)


# In[21]:


artist_list


# In[22]:


song_list = []
for row in data['items']:
    song_id = row['track']['id']
    song_name = row['track']['name']
    song_duration = row['track']['duration_ms']
    song_url = row['track']['external_urls']['spotify']
    song_popularity = row['track']['popularity']
    song_added = row['added_at']
    album_id = row['track']['album']['id']
    artist_id = row['track']['album']['artists'][0]['id']
    song_element = {'song_id':song_id,'song_name':song_name,'duration_ms':song_duration,'url':song_url,
                    'popularity':song_popularity,'song_added':song_added,'album_id':album_id,
                    'artist_id':artist_id
                   }
    song_list.append(song_element)


# In[23]:


song_list


# In[24]:


album_df = pd.DataFrame.from_dict(album_list)


# In[25]:


album_df.head()


# In[26]:


album_df.info()


# In[27]:


album_df = album_df.drop_duplicates(subset=['album_id'])


# In[30]:


artist_df = pd.DataFrame.from_dict(artist_list)


# In[31]:


artist_df = artist_df.drop_duplicates(subset=['artist_id'])


# In[32]:


song_df = pd.DataFrame.from_dict(song_list)


# In[33]:


song_df.head()


# In[34]:


artist_df.head()


# In[35]:


album_df.head()


# In[36]:


#converting to datetime object by using to_datetime 
album_df['release_date'] = pd.to_datetime(album_df['release_date'])


# In[37]:


song_df['song_added'] =  pd.to_datetime(song_df['song_added'])


# In[38]:


song_df.info()


# In[ ]:




